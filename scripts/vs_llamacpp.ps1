# scripts/vs_llamacpp.ps1 -- Phase 6.1: llama.cpp on the same GGUF, resident and under the ladder's memory caps.
#
#   powershell -ExecutionPolicy Bypass -Command "& scripts\vs_llamacpp.ps1 [-LlamaDir C:\tools\llama.cpp-b10827]
#       [-Model C:\models\Qwen3.8-27B-Q4_K_M.gguf] [-Caps 11G,8G,5G] [-Variants default,no-repack] [-Threads 6]
#       [-Out docs\data\vs_llamacpp.txt] [-BenchOut <saved llama-bench stdout>] [-SkipBench] [-SkipNoMmap] [-Repetitions 5]"
#
# What it runs, in order, and files to -Out (rewritten after every rung, so a crash loses nothing):
#   1. llama-bench on the resident model (-t 6, the default pp512 / tg128, -r repetitions): the official
#      build's own number, which the README's limitation 12 and finding 21 quote. -BenchOut re-parses a saved
#      llama-bench stdout instead of running it again (tg128 x 5 at 5 s/token is an hour).
#   2. llama-server, resident and then under each cap through scripts\cap_run.ps1 (job object: commit limits,
#      and a hard working-set maximum = cap), once per variant: "default" (the build as shipped: the CPU
#      backend repacks the Q4_K weights into private buffers, which is committed memory) and "no-repack"
#      (--no-repack: the weights stay in the mapped file, so the commit is only the context and compute
#      buffers). The same 3 prompts as the ladder (tests\fixtures\prompts.json), sent as token ids, 32 greedy
#      tokens (temperature 0), the generated ids returned by the server (return_tokens) and compared with
#      tests\fixtures\ladder_expected_ids.json, this engine's output. Per rung: load seconds, s/token from the
#      server's own timings (mean over the prompts that completed), the peak working set and peak commit of the
#      server process, whether they stayed under the cap, the model volume's physical-disk read bytes during
#      the prompts (a raw cumulative counter: the column that says whether a capped run read the drive or was
#      served soft faults from the standby list), and what happened when a prompt failed (the server's last
#      error lines: "bad allocation" is the commit cap).
#   3. llama-server with --no-mmap under the largest cap: the model read into private memory (commit).
param(
    [string]$LlamaDir = "C:\tools\llama.cpp-b10827",
    [string]$Model = "C:\models\Qwen3.8-27B-Q4_K_M.gguf",
    [string[]]$Caps = @("11G", "8G", "5G"),
    [string[]]$Variants = @("default", "no-repack"),
    [int]$Threads = 6,
    [int]$Port = 8089,
    [int]$Ctx = 4096,
    [int]$MaxTokens = 32,
    [int]$Repetitions = 5,
    [string]$Out = "docs\data\vs_llamacpp.txt",
    [string]$BenchOut = "",
    [string]$Prompts = "tests\fixtures\prompts.json",
    [string]$Expected = "tests\fixtures\ladder_expected_ids.json",
    [string]$DiskVolume = "C:",
    [switch]$SkipBench,
    [switch]$SkipNoMmap
)
$ErrorActionPreference = "Stop"
$machine = $env:COMPUTERNAME.ToLower()
$server = Join-Path $LlamaDir "llama-server.exe"
$bench = Join-Path $LlamaDir "llama-bench.exe"
$cli = Join-Path $LlamaDir "llama-cli.exe"
$capRun = Join-Path $PSScriptRoot "cap_run.ps1"
foreach ($f in @($server, $bench, $cli, $capRun, $Model, $Prompts, $Expected)) { if (-not (Test-Path $f)) { throw "missing: $f" } }
$variantArgs = @{ "default" = @(); "no-repack" = @("--no-repack"); "no-mmap" = @("--no-mmap") }
foreach ($v in $Variants) { if (-not $variantArgs.ContainsKey($v)) { throw "unknown variant $v (default, no-repack, no-mmap)" } }
$promptList = (Get-Content $Prompts -Raw | ConvertFrom-Json).prompts
$expectedIds = (Get-Content $Expected -Raw | ConvertFrom-Json).ids
$tmp = Join-Path $env:TEMP "aqueduct-vs-llamacpp"
New-Item -ItemType Directory -Force $tmp | Out-Null
$lines = New-Object System.Collections.Generic.List[string]
$outPath = (Resolve-Path -LiteralPath (Split-Path $Out -Parent)).Path + "\" + (Split-Path $Out -Leaf)
function Log([string]$s) { $lines.Add($s); Write-Host $s }
function Flush() { [System.IO.File]::WriteAllLines($outPath, $lines, (New-Object System.Text.UTF8Encoding $false)) }
function Parse-Size([string]$s) {
    if ($s -match '^([0-9.]+)G$') { return [long]([double]$Matches[1] * 1073741824) }
    if ($s -match '^([0-9.]+)M$') { return [long]([double]$Matches[1] * 1048576) }
    return [long]$s
}
function Disk-ReadBytes([string]$vol) {
    try {
        $d = Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk -ErrorAction Stop | Where-Object { $_.Name -like "*$vol*" } | Select-Object -First 1
        if ($d) { return [long]$d.DiskReadBytesPersec }
    } catch {}
    return $null
}
function Gb($b) { if ($b -eq $null) { return "n/a" } else { return ("{0:F3}" -f ($b / 1e9)) } }
function Match-Count([int[]]$got, [int[]]$exp) {
    $n = 0
    while ($n -lt [Math]::Min($got.Length, $exp.Length) -and $got[$n] -eq $exp[$n]) { $n++ }
    return $n
}
$version = (cmd /c "`"$cli`" --version 2>&1" | Out-String).Trim() -replace "`r?`n", "; "
Log "# llama.cpp versus aqueduct on the same file: $machine, $(Get-Date -Format 'yyyy-MM-dd HH:mm'), model $Model"
Log "# llama.cpp: $LlamaDir ($version); the official release zip llama-b<build>-bin-win-cpu-x64.zip; the CPU backend it picks at load (ggml-cpu-haswell.dll = AVX2 on this i7-9750H) is on the load_backend line"
Log "# caps: scripts\cap_run.ps1 puts llama-server into a job object with JOB_OBJECT_LIMIT_JOB_MEMORY + PROCESS_MEMORY (commit) = cap and sets a hard working-set maximum = cap on the process; the same 5 / 8 / 11 GiB as the ladder (docs\data\ladder.txt), whose cap was the engine's own --job-limit on commit. The working-set cap bounds the mapped file's resident pages but does not take them out of RAM: on this 32 GB machine the trimmed pages stay in the standby list, so 'disk GB' says whether a rung read the drive"
Log "# variants: default = the build as shipped (the CPU backend repacks Q4_K weights into private buffers: committed memory on top of the mapped file); no-repack = --no-repack (weights served from the mapped file)"
Log "# prompts: the ladder's 3 (tests\fixtures\prompts.json) sent as token ids, $MaxTokens greedy tokens (temperature 0), ids compared with tests\fixtures\ladder_expected_ids.json (this engine's output); s/token is the server's own timings.predicted_ms / predicted_n, mean over the prompts that completed; -t $Threads, -c $Ctx, one slot, warmup on, otherwise the build's defaults"
Log ""

# ---- 1. llama-bench, resident
$benchRows = @()
function Parse-Bench([string[]]$text) {
    $rows = @()
    foreach ($l in $text) {
        if ($l -match '^\|.*\|\s*(pp\d+|tg\d+)\s*\|\s*([0-9.]+)\s+\S+\s+([0-9.]+)\s*\|\s*$') {
            $rows += [pscustomobject]@{ test = $Matches[1]; tps = [double]$Matches[2]; sd = [double]$Matches[3] }
        }
    }
    return $rows
}
if (-not $SkipBench) {
    if ($BenchOut -ne "" -and (Test-Path $BenchOut)) {
        $bi = Get-Item $BenchOut
        Log "== llama-bench -m <model> -t $Threads -r $Repetitions (defaults: pp512, tg128), resident: re-parsed from $BenchOut (a run finished $($bi.LastWriteTime.ToString('yyyy-MM-dd HH:mm')))"
        $text = Get-Content $BenchOut -Encoding UTF8
    } else {
        Log "== llama-bench -m <model> -t $Threads -r $Repetitions (defaults: pp512, tg128), resident"
        $bo = Join-Path $tmp "bench_out.txt"; $be = Join-Path $tmp "bench_err.txt"
        $t0 = Get-Date
        $p = Start-Process -FilePath $bench -ArgumentList @("-m", $Model, "-t", $Threads, "-r", $Repetitions) -PassThru -NoNewWindow -RedirectStandardOutput $bo -RedirectStandardError $be
        $null = $p.Handle
        $p.WaitForExit()
        $wall = ((Get-Date) - $t0).TotalSeconds
        $backend = (Get-Content $be -Encoding UTF8 | Where-Object { $_ -match "loaded CPU backend" } | Select-Object -First 1)
        if ($backend) { Log "  $($backend.Trim())" }
        Log ("  (exit {0}, {1:F0} s wall)" -f $p.ExitCode, $wall)
        $text = Get-Content $bo -Encoding UTF8
    }
    foreach ($l in $text) { if ($l -match '^\|') { Log "  $($l -replace [char]0xB1, '+/-')" } }
    $benchRows = Parse-Bench $text
    foreach ($r in $benchRows) { Log ("  {0}: {1:F2} +/- {2:F2} tokens/s = {3:F3} s/token" -f $r.test, $r.tps, $r.sd, (1.0 / $r.tps)) }
    Log ""
    Flush
}

# ---- 2. llama-server: resident, then each cap, per variant
function Start-Server([string]$cap, [string[]]$extra, [string]$tag) {
    $so = Join-Path $tmp "server_${tag}_out.txt"; $se = Join-Path $tmp "server_${tag}_err.txt"
    Remove-Item -ErrorAction SilentlyContinue $so, $se
    $sargs = @("-m", $Model, "-t", $Threads, "-c", $Ctx, "-np", 1, "--port", $Port, "--host", "127.0.0.1") + $extra
    if ($cap -eq "") {
        $p = Start-Process -FilePath $server -ArgumentList $sargs -PassThru -NoNewWindow -RedirectStandardOutput $so -RedirectStandardError $se
        $null = $p.Handle
        return [pscustomobject]@{ proc = $p; id = $p.Id; out = $so; err = $se; peak_ws = [long]0; peak_commit = [long]0 }
    }
    $rep = Join-Path $tmp "cap_${tag}.json"
    $printed = & $capRun -Cap $cap -NoWait -Out $so -Err $se -Report $rep -Exe $server -Arguments ([string[]]$sargs)
    $childPid = [int]($printed | Select-Object -Last 1)
    $p = Get-Process -Id $childPid
    return [pscustomobject]@{ proc = $p; id = $childPid; out = $so; err = $se; peak_ws = [long]0; peak_commit = [long]0 }
}
function Sample-Peaks($s) {
    try {
        $s.proc.Refresh()
        if (-not $s.proc.HasExited) {
            if ([long]$s.proc.PeakWorkingSet64 -gt $s.peak_ws) { $s.peak_ws = [long]$s.proc.PeakWorkingSet64 }
            if ([long]$s.proc.PeakPagedMemorySize64 -gt $s.peak_commit) { $s.peak_commit = [long]$s.proc.PeakPagedMemorySize64 }
        }
    } catch {}
}
function Wait-Ready($s, [int]$timeoutS) {
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $timeoutS) {
        Sample-Peaks $s
        if ($s.proc.HasExited) { return $false }
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5 -ErrorAction Stop
            if ($h.status -eq "ok") { return $true }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    return $false
}
function Stop-Server($s) {
    Sample-Peaks $s
    try { Stop-Process -Id $s.id -Force -ErrorAction Stop } catch {}
    try { $s.proc.WaitForExit(30000) | Out-Null } catch {}
}
function Err-Lines($s, [int]$n) {
    if (-not (Test-Path $s.err)) { return @() }
    $all = Get-Content $s.err -Encoding UTF8
    $bad = $all | Where-Object { $_ -match ' E |GGML_ASSERT|bad alloc|failed|error|cannot|unable' } | Select-Object -Last $n
    if ($bad) { return @($bad | ForEach-Object { $_.Trim() }) }
    return @($all | Select-Object -Last $n | ForEach-Object { $_.Trim() })
}
function Buffer-Lines($s) {
    if (-not (Test-Path $s.err)) { return @() }
    return @(Get-Content $s.err -Encoding UTF8 | Where-Object { $_ -match 'buffer size|repack|CPU_Mapped|load_backend: loaded CPU' } | Select-Object -First 8 | ForEach-Object { $_.Trim() })
}

$rungs = @()
$rungList = @("resident") + $Caps
Log "== llama-server, the ladder's 3 prompts x $MaxTokens greedy tokens, resident then under each cap, per variant"
Log ("{0,-9} {1,-10} {2,7} {3,9} {4,8} {5,10} {6,9} {7,10} {8,7} {9,12} {10,10}  {11}" -f "rung", "variant", "load s", "s/token", "tok/s", "prompt t/s", "peak WS", "peak cmt", "under", "disk GB/prm", "GB/token", "ids vs aqueduct (matching prefix of 32) / failures")
foreach ($rung in $rungList) {
    foreach ($variant in $Variants) {
        $cap = if ($rung -eq "resident") { "" } else { $rung }
        $capBytes = if ($cap -ne "") { Parse-Size $cap } else { $null }
        $tag = "${rung}_${variant}"
        $s = Start-Server $cap $variantArgs[$variant] $tag
        $d0 = Disk-ReadBytes $DiskVolume
        $t0 = Get-Date
        $ready = Wait-Ready $s 1800
        $loadS = ((Get-Date) - $t0).TotalSeconds
        $dLoad = Disk-ReadBytes $DiskVolume
        $diskLoad = if ($d0 -ne $null -and $dLoad -ne $null) { $dLoad - $d0 } else { $null }
        if (-not $ready) {
            $why = (Err-Lines $s 4) -join " | "
            Sample-Peaks $s
            $code = if ($s.proc.HasExited) { "exited $($s.proc.ExitCode)" } else { "still running" }
            Log ("{0,-9} {1,-10} {2,7:F1} FAILED to load ({3}); peak WS {4} GB, peak commit {5} GB; {6}" -f $rung, $variant, $loadS, $code, (Gb $s.peak_ws), (Gb $s.peak_commit), $why)
            Stop-Server $s
            $rungs += [pscustomobject]@{ rung = $rung; variant = $variant; failed = $true; loaded = $false; why = $why; load_s = $loadS; peak_ws = $s.peak_ws; peak_commit = $s.peak_commit; disk_load = $diskLoad; results = @(); buffers = (Buffer-Lines $s) }
            Flush
            continue
        }
        $results = @()
        $failure = ""
        $d1 = Disk-ReadBytes $DiskVolume
        foreach ($p in $promptList) {
            $body = @{ prompt = @($p.ids); n_predict = $MaxTokens; temperature = 0; n_probs = 0; cache_prompt = $false; return_tokens = $true; stream = $false } | ConvertTo-Json -Compress -Depth 5
            $tw0 = Get-Date
            try {
                $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 7200
            } catch {
                $tw = ((Get-Date) - $tw0).TotalSeconds
                Start-Sleep -Milliseconds 1500
                Sample-Peaks $s
                $code = if ($s.proc.HasExited) { "server exited (code $($s.proc.ExitCode))" } else { "server still running" }
                $failure = "{0}: FAILED after {1:F0} s ({2}): {3}; server said: {4}" -f $p.name, $tw, $code, ($_.Exception.Message -replace "`r?`n", " "), ((Err-Lines $s 3) -join " | ")
                Log "    $failure"
                break
            }
            $tw = ((Get-Date) - $tw0).TotalSeconds
            Sample-Peaks $s
            if ($r.tokens -eq $null) { throw "no 'tokens' in the /completion response (return_tokens unsupported?): $($r | ConvertTo-Json -Compress -Depth 3)" }
            $got = [int[]]@($r.tokens)
            $exp = [int[]]@($expectedIds.($p.name))
            $n = Match-Count $got $exp
            $results += [pscustomobject]@{
                name = $p.name; ids = $got; match = $n; total = $exp.Length; identical = ($n -eq $exp.Length -and $got.Length -eq $exp.Length)
                s_per_token = ($r.timings.predicted_ms / 1000.0 / [Math]::Max(1, $r.timings.predicted_n)); predicted_n = $r.timings.predicted_n
                prompt_n = $r.timings.prompt_n; prompt_ms = $r.timings.prompt_ms; prompt_tps = $r.timings.prompt_per_second; wall = $tw; text = [string]$r.content
            }
        }
        $d2 = Disk-ReadBytes $DiskVolume
        $buffers = Buffer-Lines $s
        Stop-Server $s
        $peakWs = $s.peak_ws; $peakCommit = $s.peak_commit
        $diskPrompts = if ($d1 -ne $null -and $d2 -ne $null) { $d2 - $d1 } else { $null }
        $nGen = ($results | Measure-Object predicted_n -Sum).Sum
        $spt = if ($results.Count -gt 0) { ($results | Measure-Object s_per_token -Average).Average } else { 0 }
        $ptps = if ($results.Count -gt 0) { ($results | Measure-Object prompt_tps -Average).Average } else { 0 }
        $under = if ($capBytes) { if ($peakWs -le $capBytes + 64 * 1048576 -and $peakCommit -le $capBytes) { "yes" } else { "NO" } } else { "n/a" }  # the working set may overshoot a hard maximum by a few pages
        $idsSummary = ($results | ForEach-Object { "{0} {1}/{2}" -f $_.name, $_.match, $_.total }) -join ", "
        if ($failure -ne "") { $idsSummary += $(if ($idsSummary) { "; " } else { "" }) + "then " + ($failure -replace ";.*$", "") }
        $gbTok = if ($diskPrompts -ne $null -and $nGen -gt 0) { "{0:F3}" -f ($diskPrompts / 1e9 / $nGen) } else { "n/a" }
        Log ("{0,-9} {1,-10} {2,7:F1} {3,9} {4,8} {5,10:F2} {6,9} {7,10} {8,7} {9,12} {10,10}  {11}" -f $rung, $variant, $loadS, $(if ($spt -gt 0) { "{0:F3}" -f $spt } else { "-" }), $(if ($spt -gt 0) { "{0:F3}" -f (1.0 / $spt) } else { "-" }), $ptps, (Gb $peakWs), (Gb $peakCommit), $under, (Gb $diskPrompts), $gbTok, $idsSummary)
        $rungs += [pscustomobject]@{ rung = $rung; variant = $variant; failed = ($results.Count -eq 0); loaded = $true; why = $failure; cap = $capBytes; load_s = $loadS; s_per_token = $spt; prompt_tps = $ptps; peak_ws = $peakWs; peak_commit = $peakCommit; under = $under; disk_prompts = $diskPrompts; disk_load = $diskLoad; n_gen = $nGen; results = $results; buffers = $buffers }
        Flush
    }
}
Log ""
Log "# per rung and variant: load, disk read during load and during the prompts, the server's buffer lines; per prompt: s/token from the server timings, wall seconds of the request, prompt tokens and prompt tokens/s, the matching prefix against aqueduct's ids, and llama.cpp's ids"
foreach ($g in $rungs) {
    Log ("  {0} {1}: load {2:F1} s, disk read during load {3} GB{4}{5}" -f $g.rung, $g.variant, $g.load_s, (Gb $g.disk_load), $(if ($g.loaded) { ", during the prompts " + (Gb $g.disk_prompts) + " GB" } else { "" }), $(if ($g.why) { "; " + $g.why } else { "" }))
    foreach ($b in $g.buffers) { Log "      $b" }
    foreach ($r in $g.results) {
        Log ("    {0,-9} {1:F3} s/token, wall {2:F1} s, prompt {3} tokens at {4:F2} t/s, ids match {5}/{6}{7}" -f $r.name, $r.s_per_token, $r.wall, $r.prompt_n, $r.prompt_tps, $r.match, $r.total, $(if ($r.identical) { " (identical)" } else { " (diverges at token $($r.match))" }))
        Log ("    {0,-9} ids: {1}" -f "", ($r.ids -join ","))
    }
}
Log ""
Log "# aqueduct's ids for the same prompts (tests\fixtures\ladder_expected_ids.json):"
foreach ($p in $promptList) { Log ("    {0,-9} ids: {1}" -f $p.name, (@($expectedIds.($p.name)) -join ",")) }
Flush

# ---- 3. --no-mmap under the largest cap
if (-not $SkipNoMmap -and $Caps.Count -gt 0) {
    $cap = $Caps[0]
    Log ""
    Log "== llama-server --no-mmap under the $cap cap (the model read into private memory, which the commit cap counts)"
    $s = Start-Server $cap @("--no-mmap") "nommap_$cap"
    $t0 = Get-Date
    $ready = Wait-Ready $s 900
    $secs = ((Get-Date) - $t0).TotalSeconds
    Sample-Peaks $s
    if ($ready) {
        Log ("  became ready after {0:F0} s with peak working set {1} GB and peak commit {2} GB: the cap did not stop it" -f $secs, (Gb $s.peak_ws), (Gb $s.peak_commit))
    } else {
        $code = if ($s.proc.HasExited) { $s.proc.ExitCode } else { "still running" }
        Log ("  did not become ready ({0:F0} s; exit code {1}); peak working set {2} GB, peak commit {3} GB; the server's last error lines:" -f $secs, $code, (Gb $s.peak_ws), (Gb $s.peak_commit))
        foreach ($l in (Err-Lines $s 6)) { Log "    $l" }
    }
    Stop-Server $s
}
Log ""
Log "# summary (s/token = mean over the completed prompts of the server's own timings; the resident llama-bench tg128 figure is the one the README quotes)"
foreach ($g in $rungs) {
    if (-not $g.loaded) { Log ("  {0,-9} {1,-10} did not load: {2}" -f $g.rung, $g.variant, $g.why); continue }
    $idsOk = ($g.results.Count -eq 3) -and (($g.results | Where-Object { -not $_.identical }).Count -eq 0)
    $done = "{0}/3 prompts" -f $g.results.Count
    Log ("  {0,-9} {1,-10} {2} s/token ({3} tok/s) over {4}, peak working set {5} GB, peak commit {6} GB, under cap {7}, disk {8} GB per token, ids {9}{10}" -f $g.rung, $g.variant, $(if ($g.s_per_token -gt 0) { "{0:F3}" -f $g.s_per_token } else { "-" }), $(if ($g.s_per_token -gt 0) { "{0:F3}" -f (1.0 / $g.s_per_token) } else { "-" }), $done, (Gb $g.peak_ws), (Gb $g.peak_commit), $g.under, $(if ($g.disk_prompts -ne $null -and $g.n_gen -gt 0) { "{0:F3}" -f ($g.disk_prompts / 1e9 / $g.n_gen) } else { "n/a" }), $(if ($idsOk) { "identical to aqueduct on all 3 prompts" } elseif ($g.results.Count -gt 0) { (($g.results | ForEach-Object { "{0} {1}/{2}" -f $_.name, $_.match, $_.total }) -join ", ") } else { "none" }), $(if ($g.why) { "; " + ($g.why -replace ";.*$", "") } else { "" }))
}
$tg = $benchRows | Where-Object { $_.test -like "tg*" } | Select-Object -First 1
if ($tg) { Log ("  llama-bench {0} resident: {1:F2} +/- {2:F2} tokens/s = {3:F3} s/token" -f $tg.test, $tg.tps, $tg.sd, (1.0 / $tg.tps)) }
$pp = $benchRows | Where-Object { $_.test -like "pp*" } | Select-Object -First 1
if ($pp) { Log ("  llama-bench {0} resident: {1:F2} +/- {2:F2} tokens/s" -f $pp.test, $pp.tps, $pp.sd) }
Flush
Write-Host "filed to $Out"
