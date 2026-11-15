import importlib.util
import json
from pathlib import Path


def monitor():
    path = Path(__file__).resolve().parents[1] / 'scripts/vigilar_experimentos.py'
    spec = importlib.util.spec_from_file_location('monitor', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_monitor_uses_validation_and_same_best_epoch(tmp_path):
    mod = monitor()
    metrics = tmp_path / 'reports/metrics'
    metrics.mkdir(parents=True)
    (tmp_path / 'logs').mkdir()
    (metrics / 'app_resnet101_v4.json').write_text(json.dumps({
        'baseline': 'baseline_app', 'candidatos': [{'name': 'new_app', 'epochs': 12}, {'name': 'queued_app', 'epochs': 25}]}))
    (metrics / 'baseline_app_test.json').write_text(json.dumps({
        'config': {'epochs': 15}, 'best_epoch': 8,
        'val': {'qwk': .94, 'accuracy': .75, 'mae': .25}, 'test': {'qwk': .8, 'accuracy': .7}}))
    (metrics / 'new_app_history.json').write_text(json.dumps([
        {'epoch': 1, 'qwk': .93, 'accuracy': .74, 'mae': .27, 'train_loss': .8, 'train_acc': .8},
        {'epoch': 2, 'qwk': .92, 'accuracy': .80, 'mae': .26, 'train_loss': .7, 'train_acc': .85}]))
    (metrics / 'app_regla_val.json').write_text(json.dumps({'val': {'qwk': .99}}))
    (tmp_path / 'logs/new_app_entrenar.log').write_text('new_app ep3/12:  50%| progress')
    plan, rows, warnings = mod.capturar(tmp_path)
    assert not warnings
    assert [r['nombre'] for r in rows] == ['baseline_app', 'new_app', 'queued_app']
    new = rows[1]
    assert new['mejor_epoca'] == 1 and new['val']['accuracy'] == .74
    assert new['epoca'] == 2 and new['progreso'] == ('3', '12', '50')
    assert rows[0]['estado'] == 'Finalizado' and rows[2]['estado'] == 'Pendiente'
    assert new['test'] is None
    output = mod.mostrar(plan, rows, warnings)
    assert '75.00%' in output and 'TEST DISPONIBLE' in output


def test_monitor_survives_partial_writes_and_reports_tracebacks(tmp_path):
    mod = monitor()
    metrics = tmp_path / 'reports/metrics'
    metrics.mkdir(parents=True)
    (tmp_path / 'logs').mkdir()
    (metrics / 'broken_app_history.json').write_text('[{"epoch":')
    (tmp_path / 'logs/broken_app_entrenar.log').write_text(
        'Traceback (most recent call last):\nRuntimeError: CUDA out of memory')
    plan, rows, warnings = mod.capturar(tmp_path)
    assert rows[0]['estado'] == 'ERROR'
    assert any('JSONDecodeError' in a for a in warnings)
    assert any('CUDA out of memory' in a for a in warnings)
    assert 'AVISO:' in mod.mostrar(plan, rows, warnings)


def test_monitor_empty_directory(tmp_path):
    mod = monitor()
    assert 'Todavia no hay experimentos' in mod.mostrar(*mod.capturar(tmp_path))
