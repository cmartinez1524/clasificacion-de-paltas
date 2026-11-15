import importlib.util
from pathlib import Path
import json
import pytest


def test_selection_uses_validation_qwk_then_accuracy():
    path = Path(__file__).resolve().parents[1] / 'scripts/mejorar_modelo_app.py'
    spec = importlib.util.spec_from_file_location('improvement', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [
        {'nombre': 'baseline', 'val': {'qwk': .94, 'accuracy': .75}, 'test': {'accuracy': 1.0}},
        {'nombre': 'more_accuracy', 'val': {'qwk': .93, 'accuracy': .85}, 'test': {'accuracy': 1.0}},
        {'nombre': 'selected', 'val': {'qwk': .95, 'accuracy': .76}, 'test': {'accuracy': 0.0}},
    ]
    assert mod.elegir(rows) == 'selected'
    rows[0]['val']['qwk'] = .95
    rows[0]['val']['accuracy'] = .77
    assert mod.elegir(rows) == 'baseline'


@pytest.mark.parametrize('candidate_wins', [False, True])
@pytest.mark.parametrize('mode', ['--grandes', '--ajustar-resnet101'])
def test_large_models_only_test_validation_winner(tmp_path, monkeypatch, candidate_wins, mode):
    tuning = mode == '--ajustar-resnet101'
    candidates = ('resnet101_app_refinado_v4', 'resnet101_app_regularizado_v4') if tuning else ('resnet101_app_v3', 'vit_base_app_v3')
    baseline = 'vit_base_app_v3' if tuning else 'resnet50_app_s1_v2'
    report = 'app_resnet101_v4.json' if tuning else 'app_grandes_v3.json'
    source = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('improvement', source / 'scripts/mejorar_modelo_app.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    for folder in ('reports/metrics', 'configs', 'data'):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / 'data/splits.csv').write_text('fixed split')
    for name in candidates:
        (tmp_path / f'configs/{name}.yaml').write_text((source / f'configs/{name}.yaml').read_text())
    metrics = tmp_path / 'reports/metrics'
    csv = 'sample_id,y_true,y_pred\n1,1,{}\n'
    for split in ('val', 'test'):
        (metrics / f'{baseline}_{split}_preds.csv').write_text(csv.format(2))
    if tuning:
        (metrics / 'resnet101_app_v3_val_preds.csv').write_text(csv.format(1))
    monkeypatch.setattr(mod, 'compute_metrics', lambda true, pred: {'qwk': float(pred[0]), 'accuracy': .7})
    monkeypatch.setattr(mod, 'paired_bootstrap_diff', lambda *a, **kw: {'checked': True})
    calls = []

    def execute(command, **kwargs):
        name = Path(command[command.index('--config') + 1]).stem
        calls.append((name, command[-1]))
        assert '--require-cuda' in command
        if '--eval-only' in command:
            plan = json.loads((metrics / report).read_text())
            assert plan['elegido_en_validacion'] == name == candidates[0]
            assert len(calls) == 3
            split = 'test'
        else:
            assert command[-1] == '--val-only'
            split = 'val'
        prediction = 3 if candidate_wins and name == candidates[0] else 1
        (metrics / f'{name}_{split}_preds.csv').write_text(csv.format(prediction))

    monkeypatch.setattr(mod.subprocess, 'run', execute)
    mod.main([mode])
    plan = json.loads((metrics / report).read_text())
    assert plan['baseline'] == baseline
    assert plan['elegido_en_validacion'] == (candidates[0] if candidate_wins else baseline)
    assert sum(flag == '--eval-only' for _, flag in calls) == int(candidate_wins)
    assert not (metrics / f'{candidates[1]}_test_preds.csv').exists()


def test_require_cuda_prevents_cpu_training(monkeypatch):
    source = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('train_gpu_only', source / 'scripts/train.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(mod.sys, 'argv', ['train.py', '--config', str(source / 'configs/resnet101_app_refinado_v4.yaml'), '--require-cuda'])
    with pytest.raises(SystemExit, match='Se requiere CUDA'):
        mod.main()
