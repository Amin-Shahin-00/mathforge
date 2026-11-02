# One-time setup for the Lean 4 verifier on Windows (the twin of setup_lean.sh):
# installs Lean (via elan) and creates a Lake project with Mathlib, downloading
# Mathlib's pre-built cache. Needs ~5 GB of disk and git on PATH.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_lean.ps1               # ./lean_workspace
#   powershell -ExecutionPolicy Bypass -File scripts\setup_lean.ps1 -Dir D:\lean  # elsewhere
#   ... -ModifyPath   also add elan to your user PATH (otherwise MathForge finds it in ~\.elan\bin)
param([string]$Dir = "lean_workspace", [switch]$ModifyPath)
$ErrorActionPreference = "Stop"

function Invoke-Native([string]$What) {
    # native tools report failure through the exit code, not exceptions
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

$elanHome = if ($env:ELAN_HOME) { $env:ELAN_HOME } else { Join-Path $HOME ".elan" }
$elanBin = Join-Path $elanHome "bin"
if (-not (Get-Command elan -ErrorAction SilentlyContinue) -and -not (Test-Path (Join-Path $elanBin "elan.exe"))) {
    Write-Host ">> Installing elan (the Lean version manager)"
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "mathforge-elan"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    Invoke-WebRequest -UseBasicParsing -OutFile (Join-Path $tmp "elan.zip") `
        -Uri "https://github.com/leanprover/elan/releases/latest/download/elan-x86_64-pc-windows-msvc.zip"
    Expand-Archive (Join-Path $tmp "elan.zip") -DestinationPath $tmp -Force
    $initArgs = @("-y", "--default-toolchain", "none")
    if (-not $ModifyPath) { $initArgs += "--no-modify-path" }
    & (Join-Path $tmp "elan-init.exe") @initArgs; Invoke-Native "elan-init"
    Remove-Item -Recurse -Force $tmp
}
$env:PATH = "$elanBin;$env:PATH"

if (-not (Test-Path $Dir)) {
    Write-Host ">> Creating a Mathlib project in $Dir"
    lake "+leanprover-community/mathlib4:lean-toolchain" new $Dir math; Invoke-Native "lake new"
}

Push-Location $Dir
try {
    Write-Host ">> Downloading pre-built Mathlib (this is the slow part)"
    lake exe cache get; Invoke-Native "lake exe cache get"
    lake build; Invoke-Native "lake build"

    Write-Host ">> Smoke test"
    $smoke = "import Mathlib`ntheorem mf_smoke (a b : " + [char]0x211D + ") : 2 * a * b " + [char]0x2264 +
             " a ^ 2 + b ^ 2 := by nlinarith [sq_nonneg (a - b)]`n#print axioms mf_smoke`n"
    [System.IO.File]::WriteAllText((Join-Path (Get-Location) ".mf_smoke.lean"), $smoke, (New-Object System.Text.UTF8Encoding $false))
    lake env lean .mf_smoke.lean; Invoke-Native "smoke test"
    Remove-Item .mf_smoke.lean
    Write-Host ">> Lean + Mathlib ready in $(Get-Location). Point engines.selfplay.lean.project_dir here."
} finally {
    Pop-Location
}
