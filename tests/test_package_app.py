import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile


def test_package_contains_weights_and_no_training_dataset(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('package_app', source / 'scripts/empaquetar_app.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    for folder in ('app', 'src/paltas', 'checkpoints', 'reports/metrics', 'data/cache'):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    for name in ('gradio_app.py', 'requirements-local.txt', 'instalar.ps1', 'iniciar.ps1',
                 'INSTALAR.cmd', 'INICIAR.cmd', 'LEEME-local.md'):
        (tmp_path / 'app' / name).write_text('fixture', encoding='utf-8')
    (tmp_path / 'src/paltas/__init__.py').write_text('')
    (tmp_path / 'LICENSE').write_text('fixture')
    weights = b'fixed trained checkpoint fixture'
    (tmp_path / 'checkpoints/resnet50_app.pt').write_bytes(weights)
    (tmp_path / 'data/cache/example.jpg').write_bytes(b'image fixture')
    (tmp_path / 'reports/metrics/resnet50_app_test.json').write_text(json.dumps(
        {'best_epoch': 8, 'test': {'accuracy': .7, 'qwk': .9, 'mae': .3}}))
    (tmp_path / 'reports/metrics/resnet50_app_mapas.json').write_text(json.dumps([{'imagen': 'example.jpg'}]))
    mod.main()
    with zipfile.ZipFile(tmp_path / 'dist/PaltasDemo.zip') as archive:
        names = archive.namelist()
        info = json.loads(archive.read('PaltasDemo/modelo.json'))
        assert info['sha256'] == hashlib.sha256(weights).hexdigest()
        assert archive.read('PaltasDemo/checkpoints/resnet50_app.pt') == weights
        assert 'PaltasDemo/app/ejemplos/example.jpg' in names
        assert 'PaltasDemo/INICIAR.cmd' in names
        assert not any('/data/' in p or '/reports/' in p or '/.venv/' in p for p in names)
