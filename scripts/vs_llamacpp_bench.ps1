# scripts/vs_llamacpp_bench.ps1 -- Phase 6.2: llama.cpp's best case on the resident model, one llama-bench run per
# configuration, every run appended to docs\data\vs_llamacpp.txt with its full command line.
#
#   powershell -ExecutionPolicy Bypass -Command "& scripts\vs_llamacpp_bench.ps1 [-LlamaDir C:\tools\llama.cpp-b10827]
#       [-Model C:\models\Qwen3.8-27B-Q4_K_M.gguf] [-NGen 32] [-Repetitions 3] [-Out docs\data\vs_llamacpp.txt] [-Title ...]"
#
# The configurations are the list below (name, llama-bench arguments after -m <model>); edit it to try others.
# Each run is filed as: the command line, the CPU backend llama.cpp loaded, its markdown table, the parsed
# tokens/s and s/token, exit code and wall seconds. `aqueduct doctor` runs before the first and after the last
# configuration so the memory bandwidth (the laptop's thermal state as the kernels see it) brackets the sweep.
param(
    [string]$LlamaDir = "C:\tools\llama.cpp-b10827",
    [string]$Model = "C:\models\Qwen3.8-27B-Q4_K_M.gguf",
    [string]$Aqueduct = "target\release\aqueduct.exe",
    [int]$NGen = 32,
    [int]$Repetitions = 3,
    [string]$Out = "docs\data\vs_llamacpp.txt",
    [string]$Title = "Phase 6.2: llama.cpp's best case, resident, cool machine"
)
$ErrorActionPreference = "Stop"
$bench = Join-Path $LlamaDir "llama-bench.exe"
foreach ($f in @($bench, $Model, $Aqueduct)) { if (-not (Test-Path $f)) { throw "missing: $f" } }
# name, arguments (after -m <model>); -p 0 skips the prompt test so each run is minutes
$common = @("-p", "0", "-n", "$NGen", "-r", "$Repetitions")
$configs = @(
    @{ name = "baseline: -t 6, as shipped";                 args = @("-t", "6") + $common },
    @{ name = "(a) -t 6, no repack: -ot .*=CPU pins every tensor to the plain CPU buffer type (llama-bench has no --no-repack; the server's --no-repack does the same), so the weights stay Q4_K in the mapped file"; args = @("-t", "6", "-ot", ".*=CPU") + $common },
    @{ name = "(b) -t 6, no mmap (-lm none: the model read into private buffers, no page cache)"; args = @("-t", "6", "-lm", "none") + $common },
    @{ name = "(c) -t 12 (both hardware threads per core)";  args = @("-t", "12") + $common },
    @{ name = "baseline again: -t 6, as shipped (thermal drift over the sweep)"; args = @("-t", "6") + $common }
)
$tmp = Join-Path $env:TEMP "aqueduct-vs-llamacpp"
New-Item -ItemType Directory -Force $tmp | Out-Null
$outPath = (Resolve-Path -LiteralPath (Split-Path $Out -Parent)).Path + "\" + (Split-Path $Out -Leaf)
function Append([string[]]$ls) { [System.IO.File]::AppendAllLines($outPath, [string[]]$ls, (New-Object System.Text.UTF8Encoding $false)); foreach ($l in $ls) { Write-Host $l } }
function Membw([string]$tag) {
    $d = Join-Path $tmp "doctor_$tag.txt"
    # Start-Process with redirects: the doctor's "filed to" line goes to stderr, which PowerShell 5.1 would turn into a terminating error under 2>&1
    $dp = Start-Process -FilePath (Resolve-Path $Aqueduct).Path -ArgumentList @("doctor", "--model", $Model, "--runs", "2", "--out", $d) -PassThru -NoNewWindow -RedirectStandardOutput (Join-Path $tmp "doctor_${tag}_stdout.txt") -RedirectStandardError (Join-Path $tmp "doctor_${tag}_stderr.txt")
    $null = $dp.Handle
    $dp.WaitForExit()
    $m = Get-Content $d | Where-Object { $_ -match '^membw\s*:' } | Select-Object -First 1
    if ($m) { return $m.Trim() } else { return "membw: n/a" }
}
Append @("", ("== {0}: {1}, {2}; llama-bench on the resident model, -p 0 -n {3} -r {4}, one run per configuration, every command line as run" -f $Title, $env:COMPUTERNAME.ToLower(), (Get-Date -Format 'yyyy-MM-dd HH:mm'), $NGen, $Repetitions))
Append @("  aqueduct doctor before the sweep: " + (Membw "before"))
foreach ($c in $configs) {
    $tag = ($c.name -replace '[^a-z0-9]+', '_')
    $so = Join-Path $tmp "bench62_${tag}_out.txt"; $se = Join-Path $tmp "bench62_${tag}_err.txt"
    Remove-Item -ErrorAction SilentlyContinue $so, $se
    $argList = @("-m", $Model) + $c.args
    $cmdline = "llama-bench.exe " + ($argList -join " ")
    $t0 = Get-Date
    $p = Start-Process -FilePath $bench -ArgumentList $argList -PassThru -NoNewWindow -RedirectStandardOutput $so -RedirectStandardError $se
    $null = $p.Handle
    $peakWs = [long]0; $peakCommit = [long]0
    while (-not $p.HasExited) {
        try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peakWs) { $peakWs = $p.PeakWorkingSet64 }; if ($p.PeakPagedMemorySize64 -gt $peakCommit) { $peakCommit = $p.PeakPagedMemorySize64 } } catch {}
        Start-Sleep -Milliseconds 1000
    }
    $p.WaitForExit()
    $wall = ((Get-Date) - $t0).TotalSeconds
    $ls = @("  -- " + $c.name, "     $cmdline", ("     {0}, exit {1}, {2:F0} s wall, peak working set {3:F3} GB, peak commit {4:F3} GB" -f (Get-Date -Format 'HH:mm'), $p.ExitCode, $wall, ($peakWs / 1e9), ($peakCommit / 1e9)))
    $backend = Get-Content $se -Encoding UTF8 | Where-Object { $_ -match 'loaded CPU backend' } | Select-Object -First 1
    if ($backend) { $ls += "     " + $backend.Trim() }
    $rows = @()
    foreach ($l in Get-Content $so -Encoding UTF8) {
        if ($l -match '^\|') { $ls += "     " + ($l -replace [char]0xB1, '+/-') }
        if ($l -match '^\|.*\|\s*(pp\d+|tg\d+)\s*\|\s*([0-9.]+)\s+\S+\s+([0-9.]+)\s*\|\s*$') {
            $rows += [pscustomobject]@{ test = $Matches[1]; tps = [double]$Matches[2]; sd = [double]$Matches[3] }
        }
    }
    foreach ($r in $rows) { $ls += ("     {0}: {1:F2} +/- {2:F2} tokens/s = {3:F3} s/token" -f $r.test, $r.tps, $r.sd, (1.0 / $r.tps)) }
    if ($rows.Count -eq 0) {
        $ls += "     no result row; stderr tail:"
        foreach ($l in (Get-Content $se -Encoding UTF8 | Select-Object -Last 4)) { $ls += "       " + $l.Trim() }
    }
    Append $ls
}
Append @("  aqueduct doctor after the sweep: " + (Membw "after"))
Write-Host "appended to $Out"
