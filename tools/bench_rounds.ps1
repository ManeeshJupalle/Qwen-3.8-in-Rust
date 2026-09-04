# Phase 3.5 bench protocol: plugged in, idle 5 minutes, then N rounds of `membw` followed immediately by
# `bench kernels` at 5120 x 5120 and 17408 x 5120 (AVX2 only); every round is filed, not best-of. The CPU's
# ACPI thermal zone, the clock ratio (% Processor Performance, 100 = 2.6 GHz nominal) and total CPU busy are
# sampled before and after every step, so a reader can see the machine's state next to every number.
#
#   powershell -ExecutionPolicy Bypass -File tools/bench_rounds.ps1 [-Exe target\release\aqueduct.exe]
#       [-Baseline <other aqueduct.exe>] [-Rounds 3] [-IdleSeconds 300] [-Out docs\data\bench_rounds.log]
#
# `-Baseline` runs a second binary's kernels at 5120 x 5120 inside each round (same thermal state), for a
# before / after comparison of the kernels.
param(
    [string]$Exe = "target\release\aqueduct.exe",
    [string]$Baseline = "",
    [int]$Rounds = 3,
    [int]$IdleSeconds = 300,
    [string]$Out = "docs\data\bench_rounds.log"
)

$ErrorActionPreference = "Continue"

function Log([string]$s) {
    # Out-File, not Tee-Object: Tee-Object writes UTF-16 in Windows PowerShell 5.1
    $s | Out-File -FilePath $Out -Append -Encoding utf8
    Write-Host $s
}

function State([string]$label) {
    $tz = "n/a"
    try {
        $t = (Get-Counter '\Thermal Zone Information(*)\Temperature' -ErrorAction Stop).CounterSamples | Select-Object -First 1 -ExpandProperty CookedValue
        $tz = "{0:N1} C" -f ($t - 273.15)
    } catch {}
    $perf = "n/a"; $busy = "n/a"
    try {
        $p = (Get-Counter '\Processor Information(_Total)\% Processor Performance' -ErrorAction Stop).CounterSamples | Select-Object -First 1 -ExpandProperty CookedValue
        $perf = "{0:N0} %" -f $p
    } catch {}
    try {
        $b = (Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction Stop).CounterSamples | Select-Object -First 1 -ExpandProperty CookedValue
        $busy = "{0:N0} %" -f $b
    } catch {}
    $ac = "n/a"
    try { $ac = (Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction Stop | Select-Object -First 1).PowerOnline } catch {}
    # the three busiest other processes (% of one core), so a runaway service shows up next to the numbers
    $top = "n/a"
    try {
        $top = (Get-CimInstance Win32_PerfFormattedData_PerfProc_Process -ErrorAction Stop | Where-Object { $_.Name -notin @('_Total', 'Idle') -and $_.Name -notlike 'aqueduct*' -and $_.PercentProcessorTime -gt 0 } | Sort-Object PercentProcessorTime -Descending | Select-Object -First 3 | ForEach-Object { "{0} {1}%" -f $_.Name, $_.PercentProcessorTime }) -join ", "
        if (-not $top) { $top = "none" }
    } catch {}
    Log ("# state [{0}] {1}: thermal zone {2}, clock {3} of nominal, cpu busy {4}, ac power {5}, other load: {6}" -f $label, (Get-Date -Format "HH:mm:ss"), $tz, $perf, $busy, $ac, $top)
}

function Run([string]$exe, [string[]]$argv) {
    # ($args is PowerShell's automatic variable and cannot be a parameter name)
    Log ("# > {0} {1}" -f $exe, ($argv -join " "))
    & $exe @argv 2>&1 | ForEach-Object { Log ([string]$_) }
}

Log ("# bench_rounds {0}: exe {1}{2}, rounds {3}, idle {4} s" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Exe, $(if ($Baseline) { ", baseline $Baseline" } else { "" }), $Rounds, $IdleSeconds)
Log ("# cpu: {0}" -f (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name))
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
