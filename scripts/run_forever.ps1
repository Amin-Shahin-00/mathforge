# Keep MathForge running around the clock on Windows (the twin of run_forever.sh):
# if the process ever exits abnormally (crash, out-of-memory), restart it. All progress
# is checkpointed in the run directory, so every restart resumes where it left off.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_forever.ps1 configs\laptop.yaml
#   powershell -ExecutionPolicy Bypass -File scripts\run_forever.ps1 configs\gpu_small.yaml --set run.dir=runs\big
#
# Ctrl+C once: MathForge finishes the current step, saves, and the loop ends.
param([string]$Config = "configs\laptop.yaml", [Parameter(ValueFromRemainingArguments = $true)] $Rest)

$mathforge = Join-Path $PSScriptRoot "..\.venv\Scripts\mathforge.exe"
if (-not (Test-Path $mathforge)) { $mathforge = "mathforge" }   # installed some other way

while ($true) {
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  starting mathforge with $Config"
    & $mathforge run --config $Config @Rest
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  mathforge finished normally (max_rounds reached or stopped)."
        break
    }
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  mathforge exited with code $code; restarting in 30 s (Ctrl+C to stop)"
    Start-Sleep -Seconds 30
}
