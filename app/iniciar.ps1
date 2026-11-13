[CmdletBinding()]
param([switch]$CPU)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$runtime = Join-Path $env:LOCALAPPDATA 'paltas-demo'
$pointer = Join-Path $runtime 'python.txt'
if (-not (Test-Path -LiteralPath $pointer)) { throw 'Ejecuta INSTALAR.cmd una vez en este computador.' }
$python = (Get-Content -LiteralPath $pointer -Raw).Trim()
if (-not (Test-Path -LiteralPath $python)) { throw 'No se encuentra Python. Vuelve a ejecutar INSTALAR.cmd.' }
$info = Get-Content -LiteralPath modelo.json -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $info.checkpoint -Algorithm SHA256).Hash -ne $info.sha256) {
    throw 'El modelo esta corrupto. Vuelve a extraer el ZIP completo.'
}
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:HF_HUB_OFFLINE = '1'
$extra = @()
if ($CPU) { $extra += '--cpu' }
Write-Host 'Abriendo http://localhost:7860. Deja esta ventana abierta durante la demostracion.'
& $python -u app/gradio_app.py --host 127.0.0.1 --open-browser @extra
exit $LASTEXITCODE
