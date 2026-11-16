"""Monitor de solo lectura. Ejecutar desde cualquier carpeta con Python 3.10+."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
if ROOT.name == 'scripts':
    ROOT = ROOT.parent


def leer_json(path, avisos):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        avisos.append(f'{path.name}: no se pudo leer; se reintentara ({type(error).__name__}).')
        return None


def cola(path):
    """Los logs de tqdm crecen mucho: solo interesa su final."""
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 12000))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def capturar(root, plan_name='app_semillas_v6.json', solo_plan=False):
    metrics = root / 'reports/metrics'
    avisos = []
    plan = leer_json(metrics / plan_name, avisos) or {}
    configs = {c['name']: c for c in plan.get('candidatos', []) + plan.get('etapas_auxiliares', [])
               if isinstance(c, dict) and 'name' in c}
    names = set(configs)
    if plan.get('baseline'):
        names.add(plan['baseline'])
    for row in plan.get('validacion', []):
        names.add(row['nombre'])
    if not solo_plan:
        for suffix in ('_history.json', '_val.json', '_test.json'):
            names.update(p.name.removesuffix(suffix) for p in metrics.glob('*app*' + suffix))
    rows = []
    for name in sorted(names):
        # Los nombres del plan tambien pueden venir de un archivo externo.
        if Path(name).name != name:
            avisos.append('Nombre de experimento fuera de la carpeta: omitido.')
            continue
        history = leer_json(metrics / f'{name}_history.json', avisos) or []
        test_result = leer_json(metrics / f'{name}_test.json', avisos) or {}
        val_result = leer_json(metrics / f'{name}_val.json', avisos) or {}
        result = test_result or val_result
        if not (metrics / f'{name}_history.json').is_file() and not result.get('config') and name not in configs and name != plan.get('baseline'):
            continue
        config = result.get('config', configs.get(name, {}))
        best = max(history, key=lambda r: (r['qwk'], -r['epoch'])) if history else {}
        val = result.get('val', best)
        epoch = history[-1]['epoch'] if history else 0
        best_epoch = result.get('best_epoch', best.get('epoch'))
        log = cola(root / 'logs' / f'{name}_entrenar.log')
        matches = re.findall(r'ep(\d+)/(\d+):\s*(\d+)%', log)
        progress = matches[-1] if matches else None
        state = 'Finalizado' if result else 'Sin resumen' if history else 'Pendiente'
        if log and not result:
            state = 'En proceso*'
        if 'Traceback (most recent call last)' in log:
            state = 'ERROR'
            avisos.append(f'{name}: ' + log.strip().splitlines()[-1][:200])
        rows.append({'nombre': name, 'estado': state, 'epoca': epoch,
                     'max_epocas': config.get('epochs', int(progress[1]) if progress else '?'),
                     'mejor_epoca': best_epoch, 'val': val, 'test': test_result.get('test'),
                     'ultima': history[-1] if history else {}, 'progreso': progress})
    rows.sort(key=lambda r: r['val'].get('qwk', float('-inf')), reverse=True)
    stderr = cola(root / 'logs' / (Path(plan_name).stem.replace('_', '-') + '.stderr.log'))
    if stderr.strip():
        avisos.append('Log del orquestador: ' + stderr.strip()[-1000:])
    return plan, rows, avisos


def numero(value, percent=False):
    if value is None:
        return '--'
    return f'{100 * value:.2f}%' if percent else f'{value:.4f}'


def mostrar(plan, rows, avisos):
    lines = [f'EXPERIMENTOS PALTAS | {datetime.now():%Y-%m-%d %H:%M:%S}',
             'Orden: mejor QWK de VALIDACION. Accuracy y MAE corresponden al mismo checkpoint.',
             f"Elegido por el protocolo: {plan.get('elegido_en_validacion', 'pendiente')}", '',
             f"{'Experimento':36} {'Estado':13} {'Epocas':8} {'Mejor ep':8} {'QWK val':9} {'Acc val':9} {'MAE val':9}"]
    if 'estado' in plan:
        lines[1:1] = [f"Estado: {plan['estado']} | etapa {plan.get('etapa', 0)}/{plan.get('total_etapas', len(plan.get('corridas', [])))}",
                      f"Corrida actual: {plan.get('corrida_actual') or '--'}"]
    for r in rows:
        v = r['val']
        lines.append(f"{r['nombre']:36} {r['estado']:13} {str(r['epoca']) + '/' + str(r['max_epocas']):8} "
                     f"{str(r['mejor_epoca'] or '--'):8} {numero(v.get('qwk')):9} "
                     f"{numero(v.get('accuracy'), True):9} {numero(v.get('mae')):9}")
    if not rows:
        lines.append('Todavia no hay experimentos. Esperando archivos en reports/metrics/.')
    if plan.get('resumen_por_receta'):
        lines.extend(['', 'PROMEDIO +/- DESVIACION ENTRE SEMILLAS (VALIDACION):'])
        for name, stats in plan['resumen_por_receta'].items():
            lines.append(f"{name:24} n={stats['n']}/{len(plan.get('semillas', []))} | "
                         f"QWK {numero(stats['qwk']['media'])} +/- {numero(stats['qwk']['desviacion'])} | "
                         f"Acc {numero(stats['accuracy']['media'], True)} +/- {numero(stats['accuracy']['desviacion'], True)} | "
                         f"MAE {numero(stats['mae']['media'])} +/- {numero(stats['mae']['desviacion'])}")
        lines.append(f"Mejor receta por promedio: {plan.get('mejor_receta_por_promedio', 'pendiente hasta completar las cinco semillas')}")
    lines.extend(['', 'TEST DISPONIBLE (no se usa para ordenar ni elegir):'])
    for r in rows:
        if r['test']:
            t = r['test']
            lines.append(f"{r['nombre']}: QWK {numero(t.get('qwk'))} | "
                         f"accuracy {numero(t.get('accuracy'), True)} | MAE {numero(t.get('mae'))}")
    lines.extend(['', 'ULTIMA EPOCA COMPLETADA:'])
    for r in rows:
        if r['estado'] == 'En proceso*':
            last = r['ultima']
            progress = r['progreso']
            detail = f' | log: epoca {progress[0]}/{progress[1]}, {progress[2]}%' if progress else ''
            lines.append(f"{r['nombre']}: train loss {numero(last.get('train_loss'))}, "
                         f"train acc {numero(last.get('train_acc'), True)}, "
                         f"val QWK {numero(last.get('qwk'))}{detail}")
    lines.extend(['', '* En proceso indica log sin resumen final; no comprueba que el proceso siga vivo.',
                  'Las metricas se actualizan al completar cada epoca. -- significa dato no disponible.',
                  'Ctrl+C para cerrar el monitor; los entrenamientos continuan.'])
    lines.extend('AVISO: ' + a for a in avisos)
    if plan.get('error'):
        lines.append('ERROR DEL EXPERIMENTO: ' + plan['error'])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--una-vez', action='store_true', help='mostrar una captura y salir')
    parser.add_argument('--intervalo', type=float, default=10, help='segundos entre actualizaciones (10)')
    parser.add_argument('--plan', default='app_semillas_v6.json', help='nombre del plan en reports/metrics/')
    parser.add_argument('--carpeta', type=Path, default=ROOT, help='carpeta que contiene checkpoints, logs y reports')
    parser.add_argument('--solo-plan', action='store_true', help='mostrar solo las corridas del plan seleccionado')
    args = parser.parse_args(argv)
    if not (0 < args.intervalo < float('inf')):
        parser.error('--intervalo debe ser positivo y finito')
    if Path(args.plan).name != args.plan:
        parser.error('--plan debe ser un nombre de archivo dentro de reports/metrics/')
    try:
        while True:
            snapshot = capturar(args.carpeta, args.plan, args.solo_plan)
            if not args.una_vez and sys.stdout.isatty():
                os.system('cls' if os.name == 'nt' else 'clear')
            print(mostrar(*snapshot), flush=True)
            if args.una_vez:
                return
            time.sleep(args.intervalo)
    except KeyboardInterrupt:
        print('\nMonitor cerrado. Los entrenamientos no se detienen.')


if __name__ == '__main__':
    main()
