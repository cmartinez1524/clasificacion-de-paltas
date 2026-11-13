<#: Uso: powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1
Instala un entorno separado para RTX 50; no cambia el Python del sistema.
#>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$envPath = Join-Path $env:LOCALAPPDATA 'paltas-gpu/.venv'
$python = Join-Path $envPath 'Scripts/python.exe'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Falta uv. Instala uv y vuelve a ejecutar este script.'
}
if (-not (Test-Path -LiteralPath $python)) {
    uv venv --python 3.11 $envPath
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno Python 3.11.' }
}
uv pip install --python $python -r requirements.txt --override requirements-gpu-rtx50.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias CUDA.' }
& $python scripts/check_gpu.py
if ($LASTEXITCODE -ne 0) { throw 'La comprobacion CUDA fallo.' }
Write-Host "Entorno listo: $python"
Write-Host 'Para E: powershell -ExecutionPolicy Bypass -File scripts/run_asimetricas.ps1'
exit 0
