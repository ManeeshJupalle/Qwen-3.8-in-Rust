# Phase 3.5 bench protocol: plugged in, idle for a while, then N rounds of `membw` followed immediately by
# `bench kernels` at 5120 x 5120 and 17408 x 5120 (AVX2 only). Every round is filed, not best-of, and each
# step is bracketed by a measurement of what ELSE the machine is doing, so a reader can tell a slow kernel
# from a busy box.
#
#   powershell -ExecutionPolicy Bypass -File tools/bench_rounds.ps1 [-Exe target\release\aqueduct.exe]
#       [-Baseline <other aqueduct.exe>] [-Rounds 3] [-IdleSeconds 300] [-SampleSeconds 2]
#       [-Out docs\data\bench_rounds.log]
#
# `-Baseline` runs a second binary's kernels at 5120 x 5120 inside each round (same thermal state), for a
# before / after comparison.
#
# HOW THE LOAD IS MEASURED, and why not with performance counters (docs/payload-vs-doc.md finding 47):
# the Windows performance-counter subsystem is broken on this machine. `Win32_PerfFormattedData_PerfProc_Process`
# and `Get-Counter '\Process(*)\% Processor Time'` report a process's CPU time accumulated since boot as though
# it were a rate (one svchost read "900 %" while holding 611 CPU-hours over 561 hours of uptime), and
# `\Processor(_Total)\% Idle Time` is pinned at 0. The first version of this script believed those numbers and
# the Phase 3.5 report wrongly threw away its own throughput results as "taken on a saturated box".
# So: sample each process's `TotalProcessorTime` (the kernel's `GetProcessTimes`, not a counter) twice over a
# fixed interval and divide the delta by the elapsed wall time. That is a real rate, and it agrees with the
# machine: the same svchost measures 0.00 cpu-s per 4 s.
param(
    [string]$Exe = "target\release\aqueduct.exe",
    [string]$Baseline = "",
    [int]$Rounds = 3,
    [int]$IdleSeconds = 300,
    [int]$SampleSeconds = 2,
    [string]$Out = "docs\data\bench_rounds.log"
)

$ErrorActionPreference = "Continue"
$NCPU = [Environment]::ProcessorCount

function Log([string]$s) {
    # Out-File, not Tee-Object: Tee-Object writes UTF-16 in Windows PowerShell 5.1
    $s | Out-File -FilePath $Out -Append -Encoding utf8
    Write-Host $s
}

# CPU seconds consumed by every process, keyed by pid, from GetProcessTimes.
function CpuSnapshot() {
    $h = @{}
    Get-Process | ForEach-Object {
        try { $h[$_.Id] = @{ n = $_.Name; t = $_.TotalProcessorTime.TotalSeconds } } catch {}
    }
    $h
}

function State([string]$label) {
    $before = CpuSnapshot
    $t0 = Get-Date
    Start-Sleep -Seconds $SampleSeconds
    $dt = ((Get-Date) - $t0).TotalSeconds
    $after = CpuSnapshot
    $rows = @()
    $totalCpuSeconds = 0.0
    foreach ($id in $after.Keys) {
        if (-not $before.ContainsKey($id)) { continue }
        $name = $after[$id].n
        $d = $after[$id].t - $before[$id].t
        if ($d -le 0) { continue }
        $totalCpuSeconds += $d
        # our own binary is not "other load"; it is not running between steps anyway
        if ($name -like "aqueduct*") { continue }
        $rows += [pscustomobject]@{ n = $name; id = $id; pct = $d / $dt * 100 }
    }
    $top = ($rows | Sort-Object pct -Descending | Select-Object -First 3 | ForEach-Object { "{0} {1:N0} %" -f $_.n, $_.pct }) -join ", "
    if (-not $top) { $top = "none above 0" }
    $ac = "n/a"
    try { $ac = (Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction Stop | Select-Object -First 1).PowerOnline } catch {}
    Log ("# state [{0}] {1}: other load {2:N2} of {3} cores ({4:N1} % of the machine) over {5:N1} s; busiest: {6}; ac power {7}" -f `
        $label, (Get-Date -Format "HH:mm:ss"), ($totalCpuSeconds / $dt), $NCPU, (100 * $totalCpuSeconds / $dt / $NCPU), $dt, $top, $ac)
}

function Run([string]$exe, [string[]]$argv) {
    # ($args is PowerShell's automatic variable and cannot be a parameter name)
    Log ("# > {0} {1}" -f $exe, ($argv -join " "))
    & $exe @argv 2>&1 | ForEach-Object { Log ([string]$_) }
}

Log ("# bench_rounds {0}: exe {1}{2}, rounds {3}, idle {4} s, load sample {5} s" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Exe, $(if ($Baseline) { ", baseline $Baseline" } else { "" }), $Rounds, $IdleSeconds, $SampleSeconds)
Log ("# cpu: {0} ({1} logical processors)" -f (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name), $NCPU)
Log "# 'other load' is a GetProcessTimes delta over the sample window, NOT a performance counter (see the header of this script)."
State "before idle"
if ($IdleSeconds -gt 0) {
    Start-Sleep -Seconds $IdleSeconds
}
State "after idle"
for ($r = 1; $r -le $Rounds; $r++) {
    Log ("# ==== round {0} of {1} ====" -f $r, $Rounds)
    Run $Exe @("bench", "membw", "--runs", "3")
    State "after membw"
    Run $Exe @("bench", "kernels", "--rows", "5120", "--cols", "5120", "--no-scalar")
    State "after kernels 5120"
    if ($Baseline) {
        Run $Baseline @("bench", "kernels", "--rows", "5120", "--cols", "5120", "--no-scalar")
        State "after baseline kernels 5120"
    }
    Run $Exe @("bench", "kernels", "--rows", "17408", "--cols", "5120", "--no-scalar")
    State "after kernels 17408"
}
Log "# done"
