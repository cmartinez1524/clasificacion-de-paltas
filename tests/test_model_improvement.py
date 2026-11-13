import importlib.util
from pathlib import Path


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
