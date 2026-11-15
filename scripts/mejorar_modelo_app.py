"""Dos candidatos predefinidos, seleccion en validacion y test solo del elegido."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paltas.config import TrainConfig
from paltas.metrics import compute_metrics
from paltas.compare import paired_bootstrap_diff

BASELINE = 'resnet50_app'
CANDIDATOS = ('resnet50_app_ce_v2', 'resnet50_app_ordinal_v2')


def elegir(filas):
    """QWK primero, accuracy solo para desempatar; baseline participa."""
    return max(filas, key=lambda f: (f['val']['qwk'], f['val']['accuracy']))['nombre']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--refinar', action='store_true', help='refina el baseline conservando sus pesos de clase')
    mode.add_argument('--semillas', action='store_true', help='compara dos inicializaciones de la receta original')
    mode.add_argument('--grandes', action='store_true', help='ResNet-101 y ViT-Base frente a la mejor ResNet-50 actual')
    mode.add_argument('--ajustar-resnet101', action='store_true', help='refinamiento y regularizacion de ResNet-101 en GPU')
    args = parser.parse_args(argv)
    baseline = 'vit_base_app_v3' if args.ajustar_resnet101 else 'resnet50_app_s1_v2' if args.grandes else BASELINE
    candidatos = (('resnet101_app_refinado_v4', 'resnet101_app_regularizado_v4') if args.ajustar_resnet101 else
                  ('resnet101_app_v3', 'vit_base_app_v3') if args.grandes else
                  ('resnet50_app_s1_v2', 'resnet50_app_s2_v2') if args.semillas else
                  ('resnet50_app_refinado_v2',) if args.refinar else CANDIDATOS)
    metrics = ROOT / 'reports/metrics'
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    report = metrics / ('app_resnet101_v4.json' if args.ajustar_resnet101 else 'app_grandes_v3.json' if args.grandes else 'app_semillas_v2.json' if args.semillas else
                        'app_refinado_v2.json' if args.refinar else 'app_mejoras_v2.json')
    if report.exists():
        raise SystemExit(f'Ya existe {report.name}; no se sobrescribe el experimento.')
    for name in candidatos:
        if (ROOT / f'checkpoints/{name}.pt').exists() or list(metrics.glob(name + '_*')):
            raise SystemExit(f'Ya existen resultados de {name}; no se sobrescriben.')
    plan = {
        'baseline': baseline, 'baseline_semilla': 1 if args.grandes else 42,
        'criterio': 'Mayor QWK en validacion; desempate accuracy; baseline incluido. Test no participa.',
        'candidatos': [TrainConfig.from_yaml(ROOT / f'configs/{n}.yaml').to_dict() for n in candidatos],
        'splits_sha256': hashlib.sha256((ROOT / 'data/splits.csv').read_bytes()).hexdigest(),
        'limitacion': 'Test solo del elegido; seleccionar un checkpoint no demuestra mejora de la receta ni generalizacion a camaras reales.',
    }
    report.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')

    def ejecutar(name, *flags):
        with (logs / f'{name}_{"evaluar" if "--eval-only" in flags else "entrenar"}.log').open('w', encoding='utf-8') as output:
            subprocess.run([sys.executable, '-u', 'scripts/train.py', '--config',
                            f'configs/{name}.yaml', *(['--require-cuda'] if args.grandes or args.ajustar_resnet101 else []), *flags], cwd=ROOT, stdout=output,
                           stderr=subprocess.STDOUT, check=True)

    def metricas(name, split):
        df = pd.read_csv(metrics / f'{name}_{split}_preds.csv')
        return compute_metrics(df.y_true.to_numpy(), df.y_pred.to_numpy())

    rows = [{'nombre': baseline, 'val': metricas(baseline, 'val')}]
    if args.ajustar_resnet101:
        rows.append({'nombre': 'resnet101_app_v3', 'val': metricas('resnet101_app_v3', 'val')})
    for name in candidatos:
        print(f'Entrenando {name}: solo train/validacion', flush=True)
        ejecutar(name, '--val-only')
        rows.append({'nombre': name, 'val': metricas(name, 'val')})
        print(f'{name}: QWK val {rows[-1]["val"]["qwk"]:.5f}, accuracy {rows[-1]["val"]["accuracy"]:.4f}', flush=True)
    selected = elegir(rows)
    plan.update({'validacion': rows, 'elegido_en_validacion': selected})
    report.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Elegido antes de evaluar test: {selected}', flush=True)
    if selected != baseline:
        ejecutar(selected, '--eval-only')
    plan['test'] = metricas(selected, 'test')
    plan['costo_reportado'] = {'subestimacion': 2.0, 'sobreestimacion': 1.0}
    if selected != baseline:
        a = pd.read_csv(metrics / f'{selected}_test_preds.csv')
        b = pd.read_csv(metrics / f'{baseline}_test_preds.csv')
        plan['diferencias_test_vs_baseline'] = {
            k: paired_bootstrap_diff(a, b, k, n_boot=2000) for k in ('qwk', 'accuracy', 'mae')}
    report.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'elegido': selected, 'test': plan['test']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
