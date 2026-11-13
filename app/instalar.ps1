[CmdletBinding()]
param([ValidateSet('CPU', 'NVIDIA')][string]$Dispositivo = 'CPU')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$info = Get-Content -LiteralPath (Join-Path $root 'modelo.json') -Raw | ConvertFrom-Json
$checkpoint = Join-Path $root $info.checkpoint
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash -ne $info.sha256) {
    throw 'El modelo esta incompleto o corrupto. Vuelve a extraer el ZIP completo.'
}
$runtime = Join-Path $env:LOCALAPPDATA 'paltas-demo'
New-Item -ItemType Directory -Force $runtime | Out-Null
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uv = $uvCommand.Source
} else {
    $uv = Join-Path $runtime 'uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        $zip = Join-Path $runtime 'uv.zip'
        Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip -UseBasicParsing
        Expand-Archive -LiteralPath $zip -DestinationPath $runtime -Force
    }
}
$envPath = Join-Path $runtime ("venv-" + $Dispositivo.ToLower())
$python = Join-Path $envPath 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & $uv venv --python 3.11 $envPath
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo preparar Python 3.11.' }
}
$build = if ($Dispositivo -eq 'NVIDIA') { 'cu128' } else { 'cpu' }
$override = Join-Path $runtime 'torch-overrides.txt'
@("torch==2.7.1+$build", "torchvision==0.22.1+$build") | Set-Content -LiteralPath $override -Encoding UTF8
& $uv pip install --python $python -r (Join-Path $PSScriptRoot 'requirements-local.txt') --override $override --extra-index-url "https://download.pytorch.org/whl/$build" --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias. Revisa la conexion a Internet.' }
if ($Dispositivo -eq 'NVIDIA') {
    & $python -c "import torch; assert torch.cuda.is_available(), 'CUDA no disponible; instala el modo CPU'; print(torch.ones(1, device='cuda').sum().item())"
    if ($LASTEXITCODE -ne 0) { throw 'La GPU no es compatible con este entorno. Ejecuta INSTALAR.cmd para usar CPU.' }
}
$python | Set-Content -LiteralPath (Join-Path $runtime 'python.txt') -Encoding UTF8
Write-Host 'Instalacion completa. Abre INICIAR.cmd. No hace falta reentrenar.'
exit 0
