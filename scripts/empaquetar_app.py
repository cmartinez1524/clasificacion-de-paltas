"""Crea dist/PaltasDemo.zip con la app, pesos y ejemplos; no incluye entornos ni dataset."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelo', default='resnet50_app', help='nombre del experimento ResNet-50 entrenado')
    args = parser.parse_args(argv)
    experiment = args.modelo
    if Path(experiment).name != experiment:
        parser.error('El modelo debe ser un nombre, no una ruta')
    checkpoint = ROOT / 'checkpoints' / f'{experiment}.pt'
    if not checkpoint.is_file():
        raise SystemExit(f'Falta {checkpoint}. Copia el checkpoint entrenado antes de empaquetar.')
    metrics = json.loads((ROOT / 'reports/metrics' / f'{experiment}_test.json').read_text(encoding='utf-8'))
    if metrics['config']['model'] != 'resnet50':
        raise SystemExit('Este paquete de demostracion requiere un checkpoint ResNet-50.')
    examples = json.loads((ROOT / 'reports/metrics/resnet50_app_mapas.json').read_text(encoding='utf-8'))
    filename = 'PaltasDemo.zip' if experiment == 'resnet50_app' else f'PaltasDemo-{experiment}.zip'
    output = ROOT / 'dist' / filename
    output.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='paltas-demo-') as temp:
        package = Path(temp) / 'PaltasDemo'
        (package / 'app/ejemplos').mkdir(parents=True)
        (package / 'src/paltas').mkdir(parents=True)
        (package / 'checkpoints').mkdir()
        for name in ('gradio_app.py', 'requirements-local.txt', 'instalar.ps1', 'iniciar.ps1'):
            shutil.copy2(ROOT / 'app' / name, package / 'app' / name)
        for name in ('INSTALAR.cmd', 'INICIAR.cmd'):
            shutil.copy2(ROOT / 'app' / name, package / name)
        shutil.copy2(ROOT / 'app/LEEME-local.md', package / 'README.md')
        if experiment != 'resnet50_app':
            readme = package / 'README.md'
            readme.write_text(f'> **Variante experimental:** `{experiment}`. Seleccionada en validación; '
                              'no implica una mejora demostrada en cámaras reales. El paquete anterior se conserva.\n\n'
                              + readme.read_text(encoding='utf-8'), encoding='utf-8')
        shutil.copy2(ROOT / 'LICENSE', package / 'LICENSE')
        for path in (ROOT / 'src/paltas').glob('*.py'):
            shutil.copy2(path, package / 'src/paltas' / path.name)
        shutil.copy2(checkpoint, package / 'checkpoints/resnet50_app.pt')
        for example in examples:
            name = example['imagen']
            if Path(name).name != name:
                raise ValueError('Nombre de ejemplo fuera de la carpeta de imagenes')
            shutil.copy2(ROOT / 'data/cache' / name, package / 'app/ejemplos' / name)
        with checkpoint.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        info = {'nombre': 'resnet50_app', 'experimento_origen': experiment, 'arquitectura': 'resnet50',
                'checkpoint': 'checkpoints/resnet50_app.pt', 'sha256': digest,
                'best_epoch': metrics['best_epoch'],
                'test': {k: metrics['test'][k] for k in ('accuracy', 'qwk', 'mae')}}
        (package / 'modelo.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(package.parent))
    print(f'{output} ({output.stat().st_size / 1e6:.1f} MB), checkpoint SHA256 {digest}')


if __name__ == '__main__':
    main()
