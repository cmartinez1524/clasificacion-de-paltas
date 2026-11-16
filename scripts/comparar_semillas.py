"""Cinco semillas de cinco recetas en CUDA, con salidas en una carpeta independiente."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paltas.config import TrainConfig
from paltas.metrics import compute_metrics

SEEDS = (1, 2, 3, 4, 5)
BASELINE = 'vit_base_app_v3'
RECETAS = {
    'resnet50': ('resnet50_app_s1_v2', 'resnet50_app'),
    'vit_base': ('vit_base_app_v3', 'vit_base_app'),
    'resnet101_refinada': ('resnet101_app_refinado_v4', 'resnet101_app'),
    'resnet101_parcial': ('resnet101_app_parcial_v5', 'resnet101_app_parcial'),
    'vit_base_capas': ('vit_base_app_capas_v5', 'vit_base_app_capas'),
}


def crear_corridas(root, salida):
    corridas = []
    for seed in SEEDS:
        for receta, (source, stem) in RECETAS.items():
            cfg = TrainConfig.from_yaml(root / f'configs/{source}.yaml').override(
                name=f'{stem}_s{seed}_v6', seed=seed)
            if receta == 'resnet101_refinada':
                base = TrainConfig.from_yaml(root / 'configs/resnet101_app_v3.yaml').override(
                    name=f'resnet101_app_base_s{seed}_v6', seed=seed)
                corridas.append({'receta': receta, 'final': False, 'config': base.to_dict()})
                cfg.init_checkpoint = str(salida / f'checkpoints/{base.name}.pt')
            cfg.notes = f'Cinco semillas v6: receta {receta}, semilla {seed}; sin fotos nuevas.'
            corridas.append({'receta': receta, 'final': True, 'config': cfg.to_dict()})
    return corridas


def resumir(filas):
    resumen = {}
    for receta in RECETAS:
        runs = [r for r in filas if r['receta'] == receta]
        if not runs:
            continue
        result = {'n': len(runs)}
        for key in ('qwk', 'accuracy', 'mae'):
            values = [r['val'][key] for r in runs]
            result[key] = {'media': statistics.mean(values),
                           'desviacion': statistics.stdev(values) if len(values) > 1 else None}
        resumen[receta] = result
    return resumen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=Path('E:/PaltasExperimentos/semillas_v6'))
    parser.add_argument('--reanudar', action='store_true', help='continua corridas finalizadas; no retoma epocas parciales')
    args = parser.parse_args(argv)
    salida = args.salida.resolve()
    if salida == Path(salida.anchor):
        parser.error('--salida debe ser una carpeta de experimentos, no la raiz del disco')
    metrics = salida / 'reports/metrics'
    checkpoints = salida / 'checkpoints'
    report = metrics / 'app_semillas_v6.json'
    configs_dir = salida / 'configs'
    logs = salida / 'logs'
    split_hash = hashlib.sha256((ROOT / 'data/splits.csv').read_bytes()).hexdigest()

    def codigo():
        files = [ROOT / 'scripts/train.py', ROOT / 'scripts/comparar_semillas.py',
                 *sorted((ROOT / 'src/paltas').glob('*.py'))]
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}

    if args.reanudar:
        plan = json.loads(report.read_text(encoding='utf-8'))
        if plan['splits_sha256'] != split_hash or Path(plan['salida']) != salida:
            raise SystemExit('Splits o carpeta distintos: no se mezclan experimentos.')
        if plan['estado'] == 'completo':
            print(f'Ya termino: {report}')
            return
        if plan['codigo_sha256'] != codigo():
            raise SystemExit('Cambio el codigo de entrenamiento; no se mezclan corridas.')
    else:
        if report.exists():
            raise SystemExit('El plan ya existe. Usa --reanudar para continuar sin sobrescribirlo.')
        corridas = crear_corridas(ROOT, salida)
        for run in corridas:
            name = run['config']['name']
            if (configs_dir / f'{name}.yaml').exists() or (checkpoints / f'{name}.pt').exists() or list(metrics.glob(name + '_*')):
                raise SystemExit(f'Ya hay archivos de {name}; no se sobrescriben.')
        parent = salida
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < 10 * 2**30:
            raise SystemExit('Se requieren al menos 10 GiB libres para las 30 etapas.')
        plan = {
            'estado': 'preparado', 'semillas': list(SEEDS), 'baseline': BASELINE,
            'salida': str(salida), 'repo': str(ROOT), 'splits_sha256': split_hash,
            'codigo_sha256': codigo(),
            'criterio': 'Receta por media QWK val; checkpoint por QWK val, accuracy desempata, baseline incluido. Test solo del elegido.',
            'candidatos': [r['config'] for r in corridas if r['final']],
            'etapas_auxiliares': [r['config'] for r in corridas if not r['final']],
            'corridas': [{'nombre': r['config']['name'], 'receta': r['receta'], 'final': r['final']} for r in corridas],
            'limitacion': 'La desviacion entre semillas no mide incertidumbre ante frutos nuevos. Validacion y test usados antes; sin fotos nuevas.',
        }
    if not torch.cuda.is_available():
        raise SystemExit('Se requiere CUDA; no se entrena en CPU.')
    configs = {c['name']: c for c in plan['candidatos'] + plan['etapas_auxiliares']}
    for directory in (metrics, checkpoints, configs_dir, logs):
        directory.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs.items():
        path = configs_dir / f'{name}.yaml'
        if path.exists():
            if TrainConfig.from_yaml(path).to_dict() != cfg:
                raise SystemExit(f'Configuracion modificada: {name}')
        else:
            path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    for relative in (f'checkpoints/{BASELINE}.pt',
                     f'reports/metrics/{BASELINE}_val.json', f'reports/metrics/{BASELINE}_val_preds.csv'):
        target = salida / relative
        if not target.exists():
            shutil.copy2(ROOT / relative, target)

    def guardar():
        report.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')

    def ejecutar(name, evaluar=False):
        log = logs / f'{name}_{"evaluar" if evaluar else "entrenar"}.log'
        if log.exists():
            raise RuntimeError(f'Ya existe {log.name}; revisar la corrida parcial antes de reiniciar.')
        env = os.environ.copy()
        env['PALTAS_OUTPUT_DIR'] = str(salida)
        env['HF_HUB_OFFLINE'] = '1'
        with log.open('w', encoding='utf-8') as output:
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/train.py'), '--config',
                            str(configs_dir / f'{name}.yaml'), '--require-cuda',
                            '--eval-only' if evaluar else '--val-only'], cwd=ROOT, env=env,
                           stdout=output, stderr=subprocess.STDOUT, check=True)

    reference = pd.read_csv(metrics / f'{BASELINE}_val_preds.csv')

    def metricas(name, split):
        df = pd.read_csv(metrics / f'{name}_{split}_preds.csv')
        if split == 'val' and not reference[['file_name', 'sample_id', 'y_true']].equals(df[['file_name', 'sample_id', 'y_true']]):
            raise RuntimeError(f'Ejemplos o etiquetas distintos en {name}')
        return compute_metrics(df.y_true.to_numpy(), df.y_pred.to_numpy())

    guardar()
    filas = []
    try:
        for index, run in enumerate(plan['corridas'], 1):
            name = run['nombre']
            plan.update(estado='entrenando', corrida_actual=name, etapa=index, total_etapas=len(plan['corridas']))
            guardar()
            summary = metrics / f'{name}_val.json'
            if not summary.exists():
                if codigo() != plan['codigo_sha256'] or TrainConfig.from_yaml(configs_dir / f'{name}.yaml').to_dict() != configs[name]:
                    raise RuntimeError('Cambio el codigo o la configuracion; no se mezclan corridas.')
                if (checkpoints / f'{name}.pt').exists() or list(metrics.glob(name + '_*')):
                    raise RuntimeError(f'Corrida parcial {name}; no se sobrescribe.')
                print(f'Etapa {index}/{len(plan["corridas"])}: {name}', flush=True)
                ejecutar(name)
            saved = json.loads(summary.read_text(encoding='utf-8'))
            if saved['config'] != configs[name] or not (checkpoints / f'{name}.pt').is_file():
                raise RuntimeError(f'Corrida inconsistente: {name}')
            val = metricas(name, 'val')
            if run['final']:
                filas.append({'nombre': name, 'receta': run['receta'], 'semilla': configs[name]['seed'], 'val': val})
            plan.update(validacion=filas, resumen_por_receta=resumir(filas))
            guardar()
        assert all(len([r for r in filas if r['receta'] == receta]) == len(plan['semillas']) for receta in RECETAS)
        means = plan['resumen_por_receta']
        family = max(means, key=lambda key: (means[key]['qwk']['media'], means[key]['accuracy']['media']))
        baseline = {'nombre': BASELINE, 'val': metricas(BASELINE, 'val')}
        selected = max([baseline, *filas], key=lambda r: (r['val']['qwk'], r['val']['accuracy']))['nombre']
        plan.update(estado='evaluando', mejor_receta_por_promedio=family,
                    mejor_checkpoint_nuevo=max(filas, key=lambda r: (r['val']['qwk'], r['val']['accuracy']))['nombre'],
                    elegido_en_validacion=selected)
        guardar()
        print(f'Elegido antes de test: {selected}; mejor receta media: {family}', flush=True)
        if selected == BASELINE:
            for suffix in ('test.json', 'test_preds.csv'):
                source = ROOT / f'reports/metrics/{BASELINE}_{suffix}'
                target = metrics / source.name
                if not target.exists():
                    shutil.copy2(source, target)
        elif not (metrics / f'{selected}_test.json').exists():
            ejecutar(selected, evaluar=True)
        plan.update(test=metricas(selected, 'test'), estado='completo', corrida_actual=None)
        guardar()
        pd.DataFrame([{'nombre': r['nombre'], 'receta': r['receta'], 'semilla': r['semilla'],
                       **{k: r['val'][k] for k in ('qwk', 'accuracy', 'mae')}} for r in filas]).to_csv(
                           metrics / 'comparacion_semillas_v6.csv', index=False)
        print(f'COMPLETO: {report}', flush=True)
    except Exception as error:
        plan.update(estado='error', error=str(error))
        guardar()
        raise


if __name__ == '__main__':
    main()
