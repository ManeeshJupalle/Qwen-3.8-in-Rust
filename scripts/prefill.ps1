# scripts/prefill.ps1 -- Phase 5.5: prefill tok/s per prompt length, the blocked GEMM against the Phase 3.3 path.
# For each budget (resident and 11 GiB, the free RAM of a 16 GB laptop) and each prompt of
# tests/fixtures/prefill_prompts.json (32 / 128 / 512 raw ids, prefixes of one text), `aqueduct run --max-tokens 1`
# is timed three ways, back to back so the pairs share their thermal state:
#   phase3   AQUEDUCT_MATMUL_T=0 AQUEDUCT_MATMUL_THREADS=6  the Phase 3.3 per-row matmul on the physical cores
#   blocked6 AQUEDUCT_MATMUL_THREADS=6                       the blocked GEMM on the physical cores
#   blocked  (defaults)                                      the blocked GEMM on every hardware thread
# The stats file's prefill_s is the prompt's wall time through the layers, the final norm and the head. A membw
# reading brackets the run (the laptop throttles, docs/data/membw.txt).
#
#   powershell -ExecutionPolicy Bypass -File scripts\prefill.ps1 [-Budgets resident,11G] [-Out docs\data\prefill.txt]
param(
    [string]$Exe = "target\release\aqueduct.exe",
    [string]$Model = "C:\models\Qwen3.8-27B-Q4_K_M.gguf",
    [string[]]$Budgets = @("resident", "11G"),
    [string]$Prompts = "tests\fixtures\prefill_prompts.json",
    [string]$Out = "docs\data\prefill.txt",
    [switch]$NoJob
)

$ErrorActionPreference = "Stop"
$machine = $env:COMPUTERNAME.ToLower()
$promptList = (Get-Content $Prompts -Raw | ConvertFrom-Json).prompts
$tmp = Join-Path $env:TEMP "aqueduct-prefill"
New-Item -ItemType Directory -Force $tmp | Out-Null
$lines = New-Object System.Collections.Generic.List[string]
function Log([string]$s) { $lines.Add($s); Write-Host $s }

function Membw() {
    $txt = & $Exe bench membw --gib 1 --runs 3 2>&1 | Out-String
    $m = [regex]::Matches($txt, '(?m)^(\d+)\s+([0-9.]+)\s+([0-9.]+)')
    if ($m.Count -gt 0) { return "{0:F2}" -f (($m | ForEach-Object { [double]$_.Groups[2].Value } | Measure-Object -Maximum).Maximum) }
    return "?"
}

$configs = @(
    @{ name = "phase3"; tile = "0"; threads = "6" },
    @{ name = "blocked6"; tile = ""; threads = "6" },
    @{ name = "blocked"; tile = ""; threads = "" }
)

Log "# aqueduct prefill: $machine, $(Get-Date -Format 'yyyy-MM-dd HH:mm'), exe $Exe, model $Model"
Log "# prompts from $Prompts (raw ids, prefixes of one text); --max-tokens 1, so prefill_s is the prompt through the layers, the final norm and the head"
Log "# phase3 = AQUEDUCT_MATMUL_T=0 (the Phase 3.3 per-row matmul) on 6 threads; blocked6 = the blocked GEMM on 6 threads; blocked = the blocked GEMM on every hardware thread (the default)"
Log "# membw before: $(Membw) GB/s"
Log ("{0,-9} {1,-5} {2,5} {3,-9} {4,9} {5,8} {6,9} {7,8} {8,7}" -f "budget", "prompt", "n", "config", "prefill s", "tok/s", "x phase3", "load s", "pinned")
$rows = @()
foreach ($b in $Budgets) {
    foreach ($p in $promptList) {
        $base = $null
        foreach ($c in $configs) {
            $ids = ($p.ids -join ",")
            $stats = Join-Path $tmp ("stats_{0}_{1}_{2}.json" -f $b, $p.name, $c.name)
            $so = Join-Path $tmp ("out_{0}_{1}_{2}.txt" -f $b, $p.name, $c.name)
            $se = Join-Path $tmp ("err_{0}_{1}_{2}.txt" -f $b, $p.name, $c.name)
            $runArgs = @("run", "--model", $Model, "--ids", $ids, "--ids-only", "--max-tokens", "1", "--stats", $stats)
            if ($b -ne "resident") {
                $runArgs += @("--budget", $b)
                if (-not $NoJob) { $runArgs += @("--job-limit", $b) }
            }
            if ($c.tile -ne "") { $env:AQUEDUCT_MATMUL_T = $c.tile } else { Remove-Item Env:AQUEDUCT_MATMUL_T -ErrorAction SilentlyContinue }
            if ($c.threads -ne "") { $env:AQUEDUCT_MATMUL_THREADS = $c.threads } else { Remove-Item Env:AQUEDUCT_MATMUL_THREADS -ErrorAction SilentlyContinue }
            Remove-Item -ErrorAction SilentlyContinue $stats, $so, $se
            $proc = Start-Process -FilePath $Exe -ArgumentList $runArgs -PassThru -NoNewWindow -RedirectStandardOutput $so -RedirectStandardError $se
            $null = $proc.Handle
            $proc.WaitForExit()
            if ($proc.ExitCode -ne 0 -or -not (Test-Path $stats)) {
                Log "$b $($p.name) $($c.name): FAILED (exit $($proc.ExitCode)); stderr:"
                Get-Content $se | ForEach-Object { Log "    $_" }
                throw "run failed"
            }
            $st = Get-Content $stats -Raw | ConvertFrom-Json
            if ($c.name -eq "phase3") { $base = $st.prefill_s }
            $row = [pscustomobject]@{ budget = $b; prompt = $p.name; n = $st.prompt_len; config = $c.name; prefill_s = $st.prefill_s; tok_s = ($st.prompt_len / $st.prefill_s); speedup = ($base / $st.prefill_s); load_s = $st.load_s; pinned = $st.pinned; first = (Get-Content $so | Select-Object -Last 1).Trim() }
            $rows += $row
            Log ("{0,-9} {1,-5} {2,5} {3,-9} {4,9:F3} {5,8:F2} {6,9:F2} {7,8:F1} {8,7}" -f $row.budget, $row.prompt, $row.n, $row.config, $row.prefill_s, $row.tok_s, $row.speedup, $row.load_s, $row.pinned)
        }
    }
}
Remove-Item Env:AQUEDUCT_MATMUL_T -ErrorAction SilentlyContinue
Remove-Item Env:AQUEDUCT_MATMUL_THREADS -ErrorAction SilentlyContinue
Log "# membw after: $(Membw) GB/s"
Log ""
Log "# first generated id per (budget, prompt) and config: the three configs must agree (the blocked kernels are bit-identical to the per-row path)"
$ok = $true
foreach ($b in $Budgets) {
    foreach ($p in $promptList) {
        $r = $rows | Where-Object { $_.budget -eq $b -and $_.prompt -eq $p.name }
        $firsts = ($r | ForEach-Object { $_.first } | Sort-Object -Unique)
        if ($firsts.Count -ne 1) { $ok = $false }
        Log ("  {0,-9} {1,-5} first id {2} : {3}" -f $b, $p.name, ($firsts -join " / "), $(if ($firsts.Count -eq 1) { "identical" } else { "DIFFER" }))
    }
}
Log "identity of the first token across configs: $(if ($ok) { 'yes' } else { 'NO' })"
[System.IO.File]::WriteAllLines((Resolve-Path -LiteralPath (Split-Path $Out -Parent)).Path + "\" + (Split-Path $Out -Leaf), $lines, (New-Object System.Text.UTF8Encoding $false))
Write-Host "filed to $Out"
