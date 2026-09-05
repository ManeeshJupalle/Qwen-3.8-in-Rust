# scripts/chat_16g.ps1 -- Phase 5.3 gate: a real multi-turn chat at a 16 GiB budget, end to end through
# `aqueduct chat` (template, prefix reuse across turns, streaming, stop set, sampling from
# generation_config.json), with --spec K; the transcript and the per-turn tok/s and acceptance lines are filed.
#
#   powershell -ExecutionPolicy Bypass -File scripts\chat_16g.ps1 [-Budget 16G] [-Spec 3] [-Out docs\data\chat_16g.txt] [-Greedy] [-NoThink]
param(
    [string]$Exe = "target\release\aqueduct.exe",
    [string]$Budget = "16G",
    [int]$Spec = 3,
    [int]$MaxTokens = 400,
    [string]$Out = "docs\data\chat_16g.txt",
    [switch]$Greedy,
    [switch]$NoThink,
    [switch]$NoJob
)
$ErrorActionPreference = "Stop"
$turns = @(
    "Hi! In one sentence, what does a memory plan do in an inference engine?",
    "Now give the same idea as a three-item list, and end the list with the word DONE.",
    "What did I ask you first? Quote my question exactly."
)
$chatArgs = @("chat", "--budget", $Budget, "--spec", $Spec, "--max-tokens", $MaxTokens, "--seed", "7")
if (-not $NoJob) { $chatArgs += @("--job-limit", $Budget) }
if ($Greedy) { $chatArgs += "--greedy" }
if ($NoThink) { $chatArgs += "--no-think" }
$stdin = ($turns -join "`n") + "`n/quit`n"
$tmp = Join-Path $env:TEMP "aqueduct-chat"
New-Item -ItemType Directory -Force $tmp | Out-Null
$inFile = Join-Path $tmp "turns.txt"
[System.IO.File]::WriteAllText($inFile, $stdin, (New-Object System.Text.UTF8Encoding $false))
$so = Join-Path $tmp "chat_out.txt"
$se = Join-Path $tmp "chat_err.txt"
$t0 = Get-Date
$proc = Start-Process -FilePath $Exe -ArgumentList $chatArgs -PassThru -NoNewWindow -RedirectStandardInput $inFile -RedirectStandardOutput $so -RedirectStandardError $se
$null = $proc.Handle
$proc.WaitForExit()
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# aqueduct chat at $Budget with --spec $Spec, $($env:COMPUTERNAME.ToLower()), $(Get-Date -Format 'yyyy-MM-dd HH:mm'); exit code $($proc.ExitCode); wall $([int]((Get-Date) - $t0).TotalSeconds) s")
$lines.Add("# command: $Exe $($chatArgs -join ' ')")
$lines.Add("# stdin (three user turns, then /quit):")
foreach ($t in $turns) { $lines.Add("#   > $t") }
$lines.Add("")
$lines.Add("# ---- stdout (the transcript; per-turn stats in brackets)")
Get-Content $so -Encoding UTF8 | ForEach-Object { $lines.Add($_) }
$lines.Add("")
$lines.Add("# ---- stderr (load and plan)")
Get-Content $se -Encoding UTF8 | ForEach-Object { $lines.Add($_) }
[System.IO.File]::WriteAllLines((Resolve-Path -LiteralPath (Split-Path $Out -Parent)).Path + "\" + (Split-Path $Out -Leaf), $lines, (New-Object System.Text.UTF8Encoding $false))
Write-Host "filed to $Out (exit $($proc.ExitCode))"
Get-Content $so | Select-String -Pattern "^\[" | ForEach-Object { Write-Host $_ }
