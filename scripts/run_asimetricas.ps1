<#: Uso: powershell -ExecutionPolicy Bypass -File scripts/run_asimetricas.ps1 [-RawDir 'C:/datos/Hass Avocado Ripening Photographic Dataset'] [-CheckOnly]
Corre las semillas 1 y 2 de ambos modelos asimetricos. No sobrescribe corridas.
#>
[CmdletBinding()]
param(
    [string]$RawDir = (Join-Path $env:LOCALAPPDATA 'paltas-gpu/dataset'),
    [ValidateRange(0, 32)][int]$NumWorkers = 4,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$python = Join-Path $env:LOCALAPPDATA 'paltas-gpu/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Ejecuta scripts/setup_gpu.ps1 primero.' }
if ($RawDir) { $env:PALTAS_RAW_DIR = $RawDir }
& $python scripts/check_gpu.py
if ($LASTEXITCODE -ne 0) { throw 'CUDA no supera la comprobacion.' }
# Valida todos los archivos que usaran las particiones antes de comenzar.
@'
import sys
from pathlib import Path
sys.path.insert(0, 'src')
import pandas as pd
from paltas.paths import CACHE_DIR, IMAGES_DIR, SPLITS_CSV
root = CACHE_DIR if any(CACHE_DIR.glob('*.jpg')) else IMAGES_DIR
files = pd.read_csv(SPLITS_CSV)['image_file']
missing = [f for f in files if not (root / f).is_file()]
if missing:
    raise SystemExit(f'Faltan {len(missing)} imagenes en {root}. Usa -RawDir o genera data/cache/. Primera: {missing[0]}')
print(f'Dataset listo: {len(files)} imagenes en {root}')
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw 'Dataset incompleto; no se inicio ningun entrenamiento.' }
if ($CheckOnly) { return }
New-Item -ItemType Directory -Force logs | Out-Null
& $python -c "import platform,torch; print(platform.platform()); print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0))" | Tee-Object -FilePath 'logs/asimetricas_entorno.log'
if ($LASTEXITCODE -ne 0) { throw 'No se pudo registrar el entorno.' }
foreach ($seed in 1, 2) {
    foreach ($model in 'resnet50_asimetrico', 'vit_small_asimetrico') {
        $name = "${model}_s$seed"
        if ((Test-Path "checkpoints/$name.pt") -or (Get-ChildItem "reports/metrics/${name}_*" -ErrorAction SilentlyContinue)) {
            throw "Ya hay salidas de $name. No se sobrescriben; revisalas antes de volver a ejecutar."
        }
    }
}
foreach ($seed in 1, 2) {
    foreach ($model in 'resnet50_asimetrico', 'vit_small_asimetrico') {
        $name = "${model}_s$seed"
        $ErrorActionPreference = 'Continue'
        & $python -u scripts/train.py --config "configs/$model.yaml" --seed $seed --name $name --num-workers $NumWorkers 2>&1 | Tee-Object -FilePath "logs/$name.log"
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -ne 0) { throw "Fallo $name. Revisa logs/$name.log." }
    }
}
$ErrorActionPreference = 'Continue'
& $python scripts/11_resumen_semillas.py 2>&1 | Tee-Object -FilePath 'logs/asimetricas_resumen.log'
$ErrorActionPreference = 'Stop'
if ($LASTEXITCODE -ne 0) { throw 'Fallo el resumen; las corridas quedaron guardadas.' }
Write-Host 'Tarea E calculada. Revisar y actualizar el informe y la presentacion con las nuevas salidas.'
