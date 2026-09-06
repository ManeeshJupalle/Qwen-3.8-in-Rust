# scripts/demo.ps1 -- the one-take demo: `aqueduct doctor` (what the machine has and the verdict), the memory
# plan for the 16 GB laptop's budget with --spec 3 (the decision the engine makes before it allocates), then one
# chat turn at that budget, streaming. Record the terminal while it runs.
#
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 [-Budget 11G] [-Spec 3] [-MaxTokens 80] [-Prompt "..."]
param(
    [string]$Exe = "target\release\aqueduct.exe",
    [string]$Model = "C:\models\Qwen3.8-27B-Q4_K_M.gguf",
    [string]$Budget = "11G",
    [int]$Spec = 3,
    [int]$MaxTokens = 80,
    [string]$Prompt = "In three short sentences: why is reading a file sequentially faster than reading it randomly?"
)
$ErrorActionPreference = "Stop"
$t0 = Get-Date

Write-Host "`$ aqueduct doctor --model $Model" -ForegroundColor Cyan
& $Exe doctor --model $Model
if ($LASTEXITCODE -ne 0) { throw "doctor failed ($LASTEXITCODE)" }

Write-Host ""
Write-Host "`$ aqueduct plan --budget $Budget --spec $Spec --model $Model" -ForegroundColor Cyan
& $Exe plan --budget $Budget --spec $Spec --model $Model
if ($LASTEXITCODE -ne 0) { throw "plan refused ($LASTEXITCODE)" }

Write-Host ""
Write-Host "`$ aqueduct chat --model $Model --budget $Budget --spec $Spec --no-think --max-tokens $MaxTokens --seed 7" -ForegroundColor Cyan
Write-Host "> $Prompt" -ForegroundColor Yellow
$tmp = Join-Path $env:TEMP "aqueduct-demo"
New-Item -ItemType Directory -Force $tmp | Out-Null
$inFile = Join-Path $tmp "turn.txt"
[System.IO.File]::WriteAllText($inFile, $Prompt + "`n/quit`n", (New-Object System.Text.UTF8Encoding $false))
Get-Content $inFile | & $Exe chat --model $Model --budget $Budget --spec $Spec --no-think --max-tokens $MaxTokens --seed 7
if ($LASTEXITCODE -ne 0) { throw "chat failed ($LASTEXITCODE)" }

Write-Host ""
Write-Host ("demo done in {0} s" -f [int]((Get-Date) - $t0).TotalSeconds) -ForegroundColor Cyan
