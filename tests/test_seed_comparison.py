import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1]


def cargar():
    spec = importlib.util.spec_from_file_location('seed_comparison', SOURCE / 'scripts/comparar_semillas.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_has_25_final_models_and_five_independent_base_trainings(tmp_path):
    mod = cargar()
    runs = mod.crear_corridas(SOURCE, tmp_path)
    finals = [r for r in runs if r['final']]
    assert len(runs) == 30 and len(finals) == 25
    assert len({r['config']['name'] for r in runs}) == 30
    for receta in mod.RECETAS:
        assert sorted(r['config']['seed'] for r in finals if r['receta'] == receta) == [1, 2, 3, 4, 5]
    for run in finals:
        if run['receta'] == 'resnet101_refinada':
            seed = run['config']['seed']
            assert run['config']['init_checkpoint'] == str(tmp_path / f'checkpoints/resnet101_app_base_s{seed}_v6.pt')


@pytest.mark.parametrize('candidate_wins', [False, True])
def test_selection_external_outputs_and_completed_resume(tmp_path, monkeypatch, candidate_wins):
    mod = cargar()
    repo = tmp_path / 'repo'
    output = tmp_path / 'external-disk/results'
    for folder in ('configs', 'data', 'checkpoints', 'reports/metrics', 'scripts', 'src/paltas'):
        (repo / folder).mkdir(parents=True)
    shutil.copy2(SOURCE / 'scripts/train.py', repo / 'scripts/train.py')
    shutil.copy2(SOURCE / 'scripts/comparar_semillas.py', repo / 'scripts/comparar_semillas.py')
    shutil.copy2(SOURCE / 'src/paltas/engine.py', repo / 'src/paltas/engine.py')
    for name in {source for source, stem in mod.RECETAS.values()} | {'resnet101_app_v3'}:
        shutil.copy2(SOURCE / f'configs/{name}.yaml', repo / f'configs/{name}.yaml')
    (repo / 'data/splits.csv').write_text('fixed split')
    (repo / f'checkpoints/{mod.BASELINE}.pt').write_bytes(b'baseline weights')
    reference = repo / 'reports/metrics'
    csv = 'file_name,sample_id,y_true,y_pred\nexample.jpg,1,1,{}\n'
    for split in ('val', 'test'):
        (reference / f'{mod.BASELINE}_{split}_preds.csv').write_text(csv.format(5))
        (reference / f'{mod.BASELINE}_{split}.json').write_text(json.dumps({'config': {'model': 'vit_base'}}))
    monkeypatch.setattr(mod, 'ROOT', repo)
    monkeypatch.setattr(mod, 'SEEDS', (1, 2))
    monkeypatch.setattr(mod.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(mod.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 2**30))
    monkeypatch.setattr(mod, 'compute_metrics', lambda true, pred: {
        'qwk': float(pred[0]), 'accuracy': float(pred[0])/10, 'mae': 10-float(pred[0])})
    calls = []

    def ejecutar(command, **kwargs):
        cfg = mod.TrainConfig.from_yaml(Path(command[command.index('--config') + 1])).to_dict()
        name, seed = cfg['name'], cfg['seed']
        assert kwargs['env']['PALTAS_OUTPUT_DIR'] == str(output)
        assert '--require-cuda' in command
        calls.append(name)
        metrics = output / 'reports/metrics'
        if command[-1] == '--eval-only':
            plan = json.loads((metrics / 'app_semillas_v6.json').read_text())
            assert plan['elegido_en_validacion'] == name == 'vit_base_app_s2_v6'
            assert len(plan['validacion']) == 10
            split, prediction = 'test', 9
        else:
            assert command[-1] == '--val-only'
            assert not list(metrics.glob('*_test.json'))
            if cfg['init_checkpoint']:
                init = Path(cfg['init_checkpoint'])
                assert init == output / f'checkpoints/resnet101_app_base_s{seed}_v6.pt'
                assert init.is_file()
            split = 'val'
            prediction = 10 if '_base_' in name and name.startswith('resnet101') else 2 + seed
            if name == 'vit_base_app_s2_v6' and candidate_wins:
                prediction = 9
            (output / f'checkpoints/{name}.pt').write_bytes(b'trained weights')
        (metrics / f'{name}_{split}.json').write_text(json.dumps({'config': cfg}))
        (metrics / f'{name}_{split}_preds.csv').write_text(csv.format(prediction))

    monkeypatch.setattr(mod.subprocess, 'run', ejecutar)
    mod.main(['--salida', str(output)])
    report = output / 'reports/metrics/app_semillas_v6.json'
    plan = json.loads(report.read_text())
    assert plan['estado'] == 'completo'
    assert len(plan['validacion']) == 10
    assert all(stats['n'] == 2 for stats in plan['resumen_por_receta'].values())
    assert plan['resumen_por_receta']['resnet101_refinada']['qwk']['media'] == 3.5
    assert plan['elegido_en_validacion'] == ('vit_base_app_s2_v6' if candidate_wins else mod.BASELINE)
    assert len(calls) == 12 + int(candidate_wins)
    assert not list(reference.glob('*_v6*'))
    assert (output / 'reports/metrics/comparacion_semillas_v6.csv').is_file()
    with pytest.raises(SystemExit, match='plan ya existe'):
        mod.main(['--salida', str(output)])
    plan['estado'] = 'entrenando'
    report.write_text(json.dumps(plan))
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **kw: pytest.fail('No debe repetir corridas completas ni test'))
    mod.main(['--salida', str(output), '--reanudar'])
    assert json.loads(report.read_text())['estado'] == 'completo'
    plan['estado'] = 'entrenando'
    report.write_text(json.dumps(plan))
    (repo / 'src/paltas/engine.py').write_text('changed code')
    with pytest.raises(SystemExit, match='Cambio el codigo'):
        mod.main(['--salida', str(output), '--reanudar'])


def test_output_environment_does_not_move_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv('PALTAS_OUTPUT_DIR', str(tmp_path))
    spec = importlib.util.spec_from_file_location('isolated_paths', SOURCE / 'src/paltas/paths.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.METRICS_DIR == tmp_path / 'reports/metrics'
    assert mod.CHECKPOINT_DIR == tmp_path / 'checkpoints'
    assert mod.CACHE_DIR == SOURCE / 'data/cache'
    assert mod.SPLITS_CSV == SOURCE / 'data/splits.csv'
