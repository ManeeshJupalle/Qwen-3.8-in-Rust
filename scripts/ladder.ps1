# scripts/ladder.ps1 -- Phase 4.4: the ladder. The same 3 prompts (tests/fixtures/prompts.json), 32 greedy
# tokens each, at memory budgets 6, 8, 12, 16 GiB and fully resident; one line per rung with the layers
# pinned / streamed, bytes per token from disk, s/token, GB/s, % of membw, peak RSS, and the 32 ids.
#
#   powershell -ExecutionPolicy Bypass -File scripts\ladder.ps1 [-Exe target\release\aqueduct.exe]
#       [-Budgets 6G,8G,12G,16G,resident] [-Slots 2] [-Membw 31.6] [-Diskbw 2.5] [-Doctor docs\data\doctor_<machine>.txt]
#       [-Out docs\data\ladder.txt] [-Md docs\ladder.md] [-NoJob] [-FromStats]
#
# -FromStats re-renders both files from the stats / output files of the last run (kept under
# %TEMP%\aqueduct-ladder) without running anything. PowerShell variables are case-insensitive, so the script's
# own names never collide with its parameters (a `$md` list next to the `-Md` parameter is one variable, and a
# `[string]` parameter coerces whatever is assigned to it into a string).
#
# HOW THE CAP IS ENFORCED. Each rung runs `aqueduct run --budget B --job-limit B`. `--budget` sizes the memory
# plan (printed before anything is allocated; refused if it does not fit). `--job-limit` makes the engine put
# ITSELF into a Windows job object with JOB_OBJECT_LIMIT_JOB_MEMORY and JOB_OBJECT_LIMIT_PROCESS_MEMORY set to
# B before it allocates anything, so a commit past B fails and the process aborts: it cannot exceed the cap.
# Doing it from inside the child avoids the CreateProcess-suspended dance from PowerShell and closes the
# window between spawn and assignment. The peak is then read three ways: the job's own PeakJobMemoryUsed
# (committed bytes, what the cap counts), K32GetProcessMemoryInfo's PeakWorkingSetSize from inside the
# engine, and WorkingSet64 sampled from here every 250 ms. The rung's "peak RSS" is the largest of the three
# and the gate is peak <= B.
#
# GATE. The 32 ids must be identical at every rung and equal to tests/fixtures/ladder_expected_ids.json
# (the Phase 3 resident output). A divergence is a bug, not a tolerance.
#
# COST MODEL (4.5). Predicted s/token = bytes_ram / membw + bytes_disk / diskbw + 0.053 with membw and diskbw
# (qd2) taken from the doctor file unless given; the ratio measured / predicted is filed per rung.
param(
    [string]$Exe = "target\release\aqueduct.exe",
    [string]$Model = "C:\models\Qwen3.8-27B-Q4_K_M.gguf",
    [string[]]$Budgets = @("6G", "8G", "12G", "16G", "32G", "resident"),
    [int]$MaxTokens = 32,
    [int]$Slots = 2,
    [int]$Qd = 2,
    [double]$Membw = 0,
    [double]$Diskbw = 0,
    [string]$Doctor = "",
    [string]$Out = "docs\data\ladder.txt",
    [string]$Md = "docs\ladder.md",
    [string]$Expected = "tests\fixtures\ladder_expected_ids.json",
    [switch]$NoJob,
    [switch]$FromStats
)

$ErrorActionPreference = "Stop"
$machine = $env:COMPUTERNAME.ToLower()
if ($Doctor -eq "") { $Doctor = "docs\data\doctor_$machine.txt" }
if (($Membw -le 0 -or $Diskbw -le 0) -and (Test-Path $Doctor)) {
    foreach ($line in Get-Content $Doctor) {
        if ($Membw -le 0 -and $line -match '^membw\s*:\s*([0-9.]+) GB/s') { $Membw = [double]$Matches[1] }
        if ($Diskbw -le 0 -and $line -match '^disk qd2\s*:\s*best ([0-9.]+) GB/s') { $Diskbw = [double]$Matches[1] }
    }
}
if ($Membw -le 0 -or $Diskbw -le 0) { throw "need -Membw and -Diskbw (GB/s), or a doctor file at $Doctor (run: aqueduct doctor)" }

$prompts = (Get-Content "tests\fixtures\prompts.json" -Raw | ConvertFrom-Json).prompts
$expectedIds = (Get-Content $Expected -Raw | ConvertFrom-Json).ids
$tmp = Join-Path $env:TEMP "aqueduct-ladder"
New-Item -ItemType Directory -Force $tmp | Out-Null
$lines = New-Object System.Collections.Generic.List[string]
function Log([string]$s) { $lines.Add($s); Write-Host $s }

function Parse-Size([string]$s) {
    if ($s -match '^([0-9.]+)G$') { return [long]([double]$Matches[1] * 1073741824) }
    if ($s -match '^([0-9.]+)M$') { return [long]([double]$Matches[1] * 1048576) }
    return [long]$s
}

$runStamp = if ($FromStats) { "rendered $(Get-Date -Format 'yyyy-MM-dd HH:mm') from the stats files of the run of $((Get-Item (Join-Path $tmp 'stats_*')).LastWriteTime | Sort-Object | Select-Object -First 1 | Get-Date -Format 'yyyy-MM-dd HH:mm')" } else { Get-Date -Format 'yyyy-MM-dd HH:mm' }
Log "# aqueduct ladder: $machine, $runStamp, exe $Exe, model $Model"
Log "# cap: --budget sizes the plan, --job-limit puts the engine into a job object (JOB_OBJECT_LIMIT_JOB_MEMORY + PROCESS_MEMORY = budget) before it allocates; peak = max(job PeakJobMemoryUsed, engine PeakWorkingSetSize, WorkingSet64 sampled every 250 ms from this script)$(if ($NoJob) { ' [-NoJob: job object NOT applied]' })"
Log "# cost model: t = bytes_ram / $Membw GB/s + bytes_disk / $Diskbw GB/s + 0.053 (membw and disk qd2 from $Doctor); ring $Slots slots, qd $Qd, $MaxTokens tokens per prompt"
Log ("{0,-9} {1,-9} {2,6} {3,8} {4,12} {5,9} {6,8} {7,8} {8,10} {9,12} {10,9} {11,7} {12,9} {13,8}" -f "budget", "prompt", "pinned", "streamed", "GB/tok disk", "s/token", "GB/s", "%membw", "pred s/tok", "ratio", "peakRSS", "under", "read hid", "load s")

$rows = @()
foreach ($b in $Budgets) {
    $budgetBytes = $null
    if ($b -ne "resident") { $budgetBytes = Parse-Size $b }
    foreach ($p in $prompts) {
        $ids = ($p.ids -join ",")
        $stats = Join-Path $tmp ("stats_{0}_{1}.json" -f $b, $p.name)
        $so = Join-Path $tmp ("out_{0}_{1}.txt" -f $b, $p.name)
        $se = Join-Path $tmp ("err_{0}_{1}.txt" -f $b, $p.name)
        $sp = Join-Path $tmp ("peak_{0}_{1}.txt" -f $b, $p.name)
        $sampled = 0
        if ($FromStats) {
            if (-not (Test-Path $stats)) { throw "no stats file for $b $($p.name) at $stats" }
            if (Test-Path $sp) { $sampled = [long](Get-Content $sp) } else { $script:missingSampled = $true }
        } else {
        $args = @("run", "--model", $Model, "--ids", $ids, "--ids-only", "--max-tokens", $MaxTokens, "--slots", $Slots, "--qd", $Qd, "--stats", $stats, "--membw", $Membw, "--diskbw", $Diskbw)
        if ($budgetBytes) {
            $args += @("--budget", $b)
            if (-not $NoJob) { $args += @("--job-limit", $b) }
        }
        Remove-Item -ErrorAction SilentlyContinue $stats, $so, $se
        $proc = Start-Process -FilePath $Exe -ArgumentList $args -PassThru -NoNewWindow -RedirectStandardOutput $so -RedirectStandardError $se
        $null = $proc.Handle  # without this PowerShell 5.1 reports a null ExitCode after the process is gone
        while (-not $proc.HasExited) {
            try { $proc.Refresh(); if ($proc.WorkingSet64 -gt $sampled) { $sampled = $proc.WorkingSet64 } } catch {}
            Start-Sleep -Milliseconds 250
        }
        $proc.WaitForExit()
        if ($proc.ExitCode -ne 0 -or -not (Test-Path $stats)) {
            Log "$b $($p.name): FAILED (exit $($proc.ExitCode)); stderr:"
            Get-Content $se | ForEach-Object { Log "    $_" }
            throw "rung $b failed on $($p.name)"
        }
        Set-Content -Path $sp -Value $sampled
        }
        $st = Get-Content $stats -Raw | ConvertFrom-Json
        $outIds = (Get-Content $so | Select-Object -Last 1).Trim()
        $peaks = @([long]$st.peak_rss, [long]$sampled)
        if ($st.job_peak_job -ne $null) { $peaks += [long]$st.job_peak_job }
        $peak = ($peaks | Measure-Object -Maximum).Maximum
        $under = if ($budgetBytes) { if ($peak -le $budgetBytes) { "yes" } else { "NO" } } else { "n/a" }
        $pct = 100.0 * $st.ram_gbps / $Membw
        $ratio = if ($st.predicted_s_per_token) { $st.s_per_token / $st.predicted_s_per_token } else { 0 }
        $row = [pscustomobject]@{
            budget = $b; prompt = $p.name; pinned = $st.pinned; streamed = $st.streamed; disk_gb = $st.disk_bytes_per_token / 1e9
            s_per_token = $st.s_per_token; gbps = $st.ram_gbps; pct = $pct; predicted = $st.predicted_s_per_token; ratio = $ratio
            peak = $peak; peak_engine = [long]$st.peak_rss; peak_sampled = [long]$sampled; peak_job = $st.job_peak_job; under = $under
            prefetch = $st.prefetch_hidden; hidden_compute = $st.hidden_of_compute; read_gbps = $st.read_gbps; load_s = $st.load_s; prefill_s = $st.prefill_s; ids = $outIds
            read_s = $st.read_s; compute_s = $st.compute_s; wait_s = $st.wait_s; overlap_s = $st.overlap_s; decode_s = $st.decode_s
            plan_total = $st.plan_total; budget_bytes = $budgetBytes; steps = $st.steps; step_secs = $st.step_secs
        }
        $rows += $row
        Log ("{0,-9} {1,-9} {2,6} {3,8} {4,12:F3} {5,9:F3} {6,8:F2} {7,7:F0}% {8,10:F3} {9,12:F2} {10,9:F3} {11,7} {12,8:P0} {13,8:F1}" -f $b, $p.name, $row.pinned, $row.streamed, $row.disk_gb, $row.s_per_token, $row.gbps, $pct, $row.predicted, $ratio, ($peak / 1e9), $under, $row.prefetch, $row.load_s)
    }
}

Log ""
if ($script:missingSampled) { Log "# note: the sampled WorkingSet64 peaks (last column) were not kept by the run being re-rendered and show as 0.00; the sampled working set is bounded by the engine's own PeakWorkingSetSize (middle value) by definition, so the peak RSS column is unaffected" }
Log "# per rung (mean over the 3 prompts; peak RSS = max over prompts, GB; job/engine/sampled peaks GB)"
Log ("{0,-9} {1,6} {2,8} {3,12} {4,9} {5,8} {6,8} {7,10} {8,8} {9,9} {10,7} {11,9} {12,9} {13,8} {14,20}" -f "budget", "pinned", "streamed", "GB/tok disk", "s/token", "GB/s", "%membw", "pred s/tok", "ratio", "peakRSS", "under", "read hid", "comp hid", "read GB/s", "job/engine/sampled")
$summary = @()
foreach ($b in $Budgets) {
    $r = $rows | Where-Object budget -eq $b
    $m = [pscustomobject]@{
        budget = $b; pinned = $r[0].pinned; streamed = $r[0].streamed; disk_gb = ($r | Measure-Object disk_gb -Average).Average
        s_per_token = ($r | Measure-Object s_per_token -Average).Average; gbps = ($r | Measure-Object gbps -Average).Average
        pct = ($r | Measure-Object pct -Average).Average; predicted = ($r | Measure-Object predicted -Average).Average
        ratio = ($r | Measure-Object ratio -Average).Average; peak = ($r | Measure-Object peak -Maximum).Maximum
        under = (($r | ForEach-Object { $_.under }) -join "/"); prefetch = ($r | Measure-Object prefetch -Average).Average; hidden_compute = ($r | Measure-Object hidden_compute -Average).Average
        read_gbps = ($r | Measure-Object read_gbps -Average).Average
        peak_job = ($r | Measure-Object peak_job -Maximum).Maximum; peak_engine = ($r | Measure-Object peak_engine -Maximum).Maximum; peak_sampled = ($r | Measure-Object peak_sampled -Maximum).Maximum
        budget_bytes = $r[0].budget_bytes; plan_total = $r[0].plan_total
    }
    $summary += $m
    Log ("{0,-9} {1,6} {2,8} {3,12:F3} {4,9:F3} {5,8:F2} {6,7:F0}% {7,10:F3} {8,8:F2} {9,9:F3} {10,7} {11,9:P0} {12,9:P0} {13,8:F2} {14,20}" -f $m.budget, $m.pinned, $m.streamed, $m.disk_gb, $m.s_per_token, $m.gbps, $m.pct, $m.predicted, $m.ratio, ($m.peak / 1e9), $m.under, $m.prefetch, $m.hidden_compute, $m.read_gbps, ("{0:F2}/{1:F2}/{2:F2}" -f ($m.peak_job / 1e9), ($m.peak_engine / 1e9), ($m.peak_sampled / 1e9)))
}

Log ""
Log "# identity gate: the 32 ids per prompt at every rung, against tests/fixtures/ladder_expected_ids.json (Phase 3 resident output)"
$allIdentical = $true
$allMatch = $true
foreach ($p in $prompts) {
    $exp = ($expectedIds.($p.name) -join ",")
    $seen = @{}
    foreach ($b in $Budgets) {
        $r = $rows | Where-Object { $_.budget -eq $b -and $_.prompt -eq $p.name }
        $seen[$b] = $r.ids
        if ($r.ids -ne $exp) {
            $allMatch = $false
            $a = $r.ids.Split(","); $e = $exp.Split(",")
            $n = 0; while ($n -lt [Math]::Min($a.Length, $e.Length) -and $a[$n] -eq $e[$n]) { $n++ }
            Log "  $($p.name) @ $b : DIFFERS from Phase 3 at token $n (ours $($a[$n]) vs $($e[$n]))"
        }
    }
    $distinct = ($seen.Values | Sort-Object -Unique).Count
    if ($distinct -ne 1) { $allIdentical = $false }
    Log ("  {0,-9} identical across rungs: {1}; equal to Phase 3: {2}" -f $p.name, $(if ($distinct -eq 1) { "yes" } else { "NO ($distinct distinct)" }), $(if (($seen.Values | Where-Object { $_ -ne $exp }).Count -eq 0) { "yes" } else { "NO" }))
    Log "  $($p.name) ids: $($seen[$Budgets[0]])"
}
Log "identity: ids identical across rungs $(if ($allIdentical) { 'yes' } else { 'NO' }); vs Phase 3 resident $(if ($allMatch) { 'match' } else { 'DIFFER' })"
$worst = $summary | Sort-Object { [Math]::Abs([Math]::Log($_.ratio)) } -Descending | Select-Object -First 1
Log ("cost model: worst ratio {0:F2} at {1}; all rungs within 2x: {2}" -f $worst.ratio, $worst.budget, $(if (($summary | Where-Object { $_.ratio -gt 2 -or $_.ratio -lt 0.5 }).Count -eq 0) { "yes" } else { "NO" }))
$underAll = (($rows | Where-Object { $_.under -eq "NO" }).Count -eq 0)
Log "peak RSS under budget at every rung: $(if ($underAll) { 'yes' } else { 'NO' })"
Log ""
Log "# overlap over the decode phase, per prompt and rung: reading / computing / waiting / wall seconds, overlap seconds"
foreach ($r in $rows) { Log ("  {0} {1}: read {2:F2} compute {3:F2} wait {4:F2} wall {5:F2} overlap {6:F2}" -f $r.budget, $r.prompt, $r.read_s, $r.compute_s, $r.wait_s, $r.decode_s, $r.overlap_s) }
Log ""
Log "# per-step seconds (each prompt, each rung)"
foreach ($r in $rows) { Log ("  {0} {1}: {2}" -f $r.budget, $r.prompt, (($r.step_secs | ForEach-Object { '{0:F2}' -f $_ }) -join " ")) }

[System.IO.File]::WriteAllLines((Resolve-Path -LiteralPath (Split-Path $Out -Parent)).Path + "\" + (Split-Path $Out -Leaf), $lines, (New-Object System.Text.UTF8Encoding $false))  # UTF-8 without the BOM Out-File adds
Write-Host "filed to $Out"

# ---- markdown
$mdLines = New-Object System.Collections.Generic.List[string]
$mdLines.Add("# The ladder (Phase 4)")
$mdLines.Add("")
$mdLines.Add("Qwen3.8-27B Q4_K_M (17.8 GB file) on $machine, $(Get-Date -Format 'yyyy-MM-dd'): the same 3 prompts, $MaxTokens greedy tokens each, under memory budgets enforced by a Windows job object (see ``docs/data/ladder.txt`` for every number and ``scripts/ladder.ps1`` for the mechanism). s/token, GB/s and %membw are the mean over the 3 prompts; peak RSS is the max over the 3 prompts of the largest of the job's committed-memory peak, the engine's peak working set and the working set sampled every 250 ms. Cost model: t = bytes_ram / $Membw GB/s + bytes_disk / $Diskbw GB/s (unbuffered qd2) + 0.053 s.")
$mdLines.Add("")
$mdLines.Add("| budget | layers pinned | layers streamed | GB/token from disk | s/token | tok/s | GB/s (CPU) | % membw | predicted s/token | measured / predicted | peak RSS (GB) | under budget | compute hidden under reads |")
$mdLines.Add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
foreach ($m in $summary) {
    $mdLines.Add(("| {0} | {1} | {2} | {3:F3} | {4:F3} | {5:F2} | {6:F2} | {7:F0}% | {8:F3} | {9:F2} | {10:F3} | {11} | {12:P0} |" -f $m.budget, $m.pinned, $m.streamed, $m.disk_gb, $m.s_per_token, (1 / $m.s_per_token), $m.gbps, $m.pct, $m.predicted, $m.ratio, ($m.peak / 1e9), $m.under, $m.hidden_compute))
}
$mdLines.Add("")
$mdLines.Add("Identity gate: ids identical across rungs **$(if ($allIdentical) { 'yes' } else { 'NO' })**; equal to the Phase 3 resident output **$(if ($allMatch) { 'yes' } else { 'NO' })**. Peak RSS under budget at every rung: **$(if ($underAll) { 'yes' } else { 'NO' })**. Worst cost-model ratio $("{0:F2}" -f $worst.ratio) at $($worst.budget).")
$mdLines.Add("")
$mdLines.Add("The 32 ids (identical at every rung):")
$mdLines.Add("")
foreach ($p in $prompts) {
    $r = $rows | Where-Object { $_.budget -eq $Budgets[0] -and $_.prompt -eq $p.name }
    $mdLines.Add("- ``$($p.name)``: ``$($r.ids)``")
}
[System.IO.File]::WriteAllLines((Resolve-Path -LiteralPath (Split-Path $Md -Parent)).Path + "\" + (Split-Path $Md -Leaf), $mdLines, (New-Object System.Text.UTF8Encoding $false))
Write-Host "filed to $Md"
