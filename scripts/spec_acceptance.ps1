# scripts/spec_acceptance.ps1 -- Phase 5.4: acceptance and speed of MTP speculative decoding, and the identity
# gate. For each budget (default: resident) and each prompt of tests/fixtures/spec_prompts.json, `aqueduct run`
# decodes -MaxTokens greedy tokens plainly (k = 0) and with --spec k for each k in -Ks, with --stats; the
# acceptance per draft position, the mean accepted per round, s/token and tok/s are tabulated per (budget, k),
# and the ids of every --spec run are compared with the plain run's (a difference is a bug, not a tolerance).
# A membw pair brackets the run (the laptop throttles; docs/data/membw.txt).
#
#   powershell -ExecutionPolicy Bypass -File scripts\spec_acceptance.ps1 [-Budgets resident] [-Ks 0,1,2,3,4,5]
#       [-MaxTokens 200] [-Prompts tests\fixtures\spec_prompts.json] [-Out docs\data\spec_acceptance.txt] [-FromStats]
#
# -FromStats re-renders the file from the stats files of the last run (kept under %TEMP%\aqueduct-spec) without
# running anything. Variable names are kept apart from the parameter names (docs/payload-vs-doc.md finding 61).
param(
    [string]$Exe = "target\release\aqueduct.exe",
    [string]$Model = "C:\models\Qwen3.8-27B-Q4_K_M.gguf",
    [string[]]$Budgets = @("resident"),
    [int[]]$Ks = @(0, 1, 2, 3, 4, 5),
    [int]$MaxTokens = 200,
    [string]$Prompts = "tests\fixtures\spec_prompts.json",
    [string[]]$Only = @(),
    [string]$Out = "docs\data\spec_acceptance.txt",
    [switch]$NoJob,
    [switch]$NoMembw,
    [switch]$FromStats
)

$ErrorActionPreference = "Stop"
$machine = $env:COMPUTERNAME.ToLower()
$promptList = (Get-Content $Prompts -Raw | ConvertFrom-Json).prompts
if ($Only.Count -gt 0) { $promptList = $promptList | Where-Object { $Only -contains $_.name } }
$tmp = Join-Path $env:TEMP "aqueduct-spec"
New-Item -ItemType Directory -Force $tmp | Out-Null
$lines = New-Object System.Collections.Generic.List[string]
function Log([string]$s) { $lines.Add($s); Write-Host $s }

function Parse-Size([string]$s) {
    if ($s -match '^([0-9.]+)G$') { return [long]([double]$Matches[1] * 1073741824) }
    if ($s -match '^([0-9.]+)M$') { return [long]([double]$Matches[1] * 1048576) }
    return [long]$s
}

function Membw() {
    if ($NoMembw -or $FromStats) { return "n/a" }
    # the table rows are `threads  best  worst  runs...`; the ceiling is the best over the rows
    $txt = & $Exe bench membw --gib 1 --runs 3 2>&1 | Out-String
    $m = [regex]::Matches($txt, '(?m)^(\d+)\s+([0-9.]+)\s+([0-9.]+)')
    if ($m.Count -gt 0) { return "{0:F2}" -f (($m | ForEach-Object { [double]$_.Groups[2].Value } | Measure-Object -Maximum).Maximum) }
    return "?"
}

$stamp = if ($FromStats) { "rendered $(Get-Date -Format 'yyyy-MM-dd HH:mm') from the stats files of an earlier run" } else { Get-Date -Format 'yyyy-MM-dd HH:mm' }
Log "# aqueduct spec acceptance: $machine, $stamp, exe $Exe, model $Model"
Log "# $MaxTokens greedy tokens per prompt; prompts from $Prompts ($($promptList.Count): $(($promptList | ForEach-Object { $_.name }) -join ', ')); k = 0 is the plain loop"
Log "# membw before: $(Membw) GB/s (bench membw, 1 GiB, best of 3)"

$rows = @()
foreach ($b in $Budgets) {
    $budgetBytes = $null
    if ($b -ne "resident") { $budgetBytes = Parse-Size $b }
    foreach ($p in $promptList) {
        foreach ($k in $Ks) {
            $ids = ($p.ids -join ",")
            $stats = Join-Path $tmp ("stats_{0}_{1}_k{2}.json" -f $b, $p.name, $k)
            $so = Join-Path $tmp ("out_{0}_{1}_k{2}.txt" -f $b, $p.name, $k)
            $se = Join-Path $tmp ("err_{0}_{1}_k{2}.txt" -f $b, $p.name, $k)
            if (-not $FromStats) {
                $runArgs = @("run", "--model", $Model, "--ids", $ids, "--ids-only", "--max-tokens", $MaxTokens, "--stats", $stats)
                if ($k -gt 0) { $runArgs += @("--spec", $k) }
                if ($budgetBytes) {
                    $runArgs += @("--budget", $b)
                    if (-not $NoJob) { $runArgs += @("--job-limit", $b) }
                }
                Remove-Item -ErrorAction SilentlyContinue $stats, $so, $se
                $proc = Start-Process -FilePath $Exe -ArgumentList $runArgs -PassThru -NoNewWindow -RedirectStandardOutput $so -RedirectStandardError $se
                $null = $proc.Handle
                $proc.WaitForExit()
                if ($proc.ExitCode -ne 0 -or -not (Test-Path $stats)) {
                    Log "$b $($p.name) k=$k : FAILED (exit $($proc.ExitCode)); stderr:"
                    Get-Content $se | ForEach-Object { Log "    $_" }
                    throw "run failed"
                }
            }
            if (-not (Test-Path $stats)) { throw "no stats file at $stats" }
            $st = Get-Content $stats -Raw | ConvertFrom-Json
            $outIds = (Get-Content $so | Select-Object -Last 1).Trim()
            $row = [pscustomobject]@{
                budget = $b; prompt = $p.name; k = $k; n = $st.n_generated; s_per_token = $st.s_per_token; tok_s = (1.0 / $st.s_per_token)
                rounds = $st.spec_rounds; drafted = $st.spec_drafted; accepted = $st.spec_accepted; mean_acc = $st.spec_mean_accepted; rate = $st.spec_acceptance_rate
                accepted_at = $st.spec_accepted_at; hist = $st.spec_hist; verify_s = $st.spec_verify_s; refeed_s = $st.spec_refeed_s; refeeds = $st.spec_refeeds
                chain_s = $st.spec_chain_s; chain_steps = $st.spec_chain_steps; replay_s = $st.spec_replay_s; snapshot_s = $st.spec_snapshot_s
                decode_s = $st.decode_s; pinned = $st.pinned; streamed = $st.streamed; disk_gb = $st.disk_bytes_per_token / 1e9; ids = $outIds; load_s = $st.load_s; prefill_s = $st.prefill_s
            }
            $rows += $row
            if ($k -eq 0) {
                Log ("{0,-9} {1,-8} k=0    {2,4} tokens  {3,7:F3} s/token  {4,6:F2} tok/s  (pinned {5}, streamed {6}, {7:F3} GB/token disk)" -f $b, $p.name, $row.n, $row.s_per_token, $row.tok_s, $row.pinned, $row.streamed, $row.disk_gb)
            } else {
                $perpos = ($row.accepted_at | ForEach-Object { '{0:F2}' -f ($_ / [Math]::Max(1, $row.rounds)) }) -join " "
                Log ("{0,-9} {1,-8} k={2}    {3,4} tokens  {4,7:F3} s/token  {5,6:F2} tok/s  rounds {6,4}  accepted/round {7:F3}  rate {8:P0}  at position [{9}]  hist [{10}]  verify {11:F3} s/round  refeed {12:F3} s  chain {13:F3} s/step  replay {14:F3} s/round" -f $b, $p.name, $k, $row.n, $row.s_per_token, $row.tok_s, $row.rounds, $row.mean_acc, $row.rate, $perpos, ($row.hist -join " "), ($row.verify_s / [Math]::Max(1, $row.rounds)), ($row.refeed_s / [Math]::Max(1, $row.refeeds)), ($row.chain_s / [Math]::Max(1, $row.chain_steps)), ($row.replay_s / [Math]::Max(1, $row.rounds)))
            }
        }
    }
}
Log "# membw after: $(Membw) GB/s"
Log ""
Log "# per (budget, k), mean over the prompts: s/token, tok/s, speedup over k=0, mean accepted per round, acceptance rate, accepted at each draft position (fraction of rounds), tokens per round"
Log ("{0,-9} {1,3} {2,9} {3,8} {4,8} {5,12} {6,8} {7,-30} {8,10}" -f "budget", "k", "s/token", "tok/s", "speedup", "acc/round", "rate", "at position", "tok/round")
$summary = @()
foreach ($b in $Budgets) {
    $base = ($rows | Where-Object { $_.budget -eq $b -and $_.k -eq 0 } | Measure-Object s_per_token -Average).Average
    foreach ($k in $Ks) {
        $r = $rows | Where-Object { $_.budget -eq $b -and $_.k -eq $k }
        if (-not $r) { continue }
        $spt = ($r | Measure-Object s_per_token -Average).Average
        $m = [pscustomobject]@{ budget = $b; k = $k; s_per_token = $spt; tok_s = (1.0 / $spt); speedup = ($base / $spt) }
        if ($k -gt 0) {
            $m | Add-Member acc_round (($r | Measure-Object mean_acc -Average).Average)
            $m | Add-Member rate (($r | Measure-Object rate -Average).Average)
            $rounds = ($r | Measure-Object rounds -Sum).Sum
            $at = @(); for ($i = 0; $i -lt $k; $i++) { $at += (($r | ForEach-Object { $_.accepted_at[$i] } | Measure-Object -Sum).Sum / [Math]::Max(1, $rounds)) }
            $m | Add-Member at $at
            $m | Add-Member tok_round ((($r | ForEach-Object { $_.n }) | Measure-Object -Sum).Sum / [Math]::Max(1, $rounds))
            Log ("{0,-9} {1,3} {2,9:F3} {3,8:F2} {4,8:F2} {5,12:F3} {6,7:P0} {7,-30} {8,10:F2}" -f $b, $k, $m.s_per_token, $m.tok_s, $m.speedup, $m.acc_round, $m.rate, (($at | ForEach-Object { '{0:F2}' -f $_ }) -join " "), $m.tok_round)
        } else {
            Log ("{0,-9} {1,3} {2,9:F3} {3,8:F2} {4,8:F2} {5,12} {6,8} {7,-30} {8,10}" -f $b, $k, $m.s_per_token, $m.tok_s, 1.0, "-", "-", "-", "1.00")
        }
        $summary += $m
    }
}
Log ""
Log "# identity gate: the ids of every --spec run against the plain (k=0) run of the same prompt and budget"
$allOk = $true
foreach ($b in $Budgets) {
    foreach ($p in $promptList) {
        $plain = ($rows | Where-Object { $_.budget -eq $b -and $_.prompt -eq $p.name -and $_.k -eq 0 }).ids
        $res = @()
        foreach ($k in ($Ks | Where-Object { $_ -gt 0 })) {
            $r = $rows | Where-Object { $_.budget -eq $b -and $_.prompt -eq $p.name -and $_.k -eq $k }
            if (-not $r) { continue }
            if ($r.ids -eq $plain) { $res += "k=$k identical" } else {
                $allOk = $false
                $a = $r.ids.Split(","); $e = $plain.Split(",")
                $n = 0; while ($n -lt [Math]::Min($a.Length, $e.Length) -and $a[$n] -eq $e[$n]) { $n++ }
                $res += "k=$k DIFFERS at token $n (spec $($a[$n]) vs plain $($e[$n]))"
            }
        }
        Log ("  {0,-9} {1,-8} {2}" -f $b, $p.name, ($res -join "; "))
        Log ("  {0,-9} {1,-8} plain ids: {2}" -f $b, $p.name, $plain)
    }
}
Log "identity: $(if ($allOk) { 'every --spec run equals its plain run' } else { 'DIFFERENCES (a bug)' })"
Log ""
Log "# per-round accepted counts (each prompt, each k)"
foreach ($r in ($rows | Where-Object { $_.k -gt 0 })) { Log ("  {0} {1} k={2}: {3}" -f $r.budget, $r.prompt, $r.k, ((Get-Content (Join-Path $tmp ("stats_{0}_{1}_k{2}.json" -f $r.budget, $r.prompt, $r.k)) -Raw | ConvertFrom-Json).round_accepted -join " ")) }

[System.IO.File]::WriteAllLines((Resolve-Path -LiteralPath (Split-Path $Out -Parent)).Path + "\" + (Split-Path $Out -Leaf), $lines, (New-Object System.Text.UTF8Encoding $false))
Write-Host "filed to $Out"
