#!/usr/bin/env bash
# Experimentos posteriores al feedback del avance.
#   1. Exporta predicciones de validacion de los modelos ya entrenados (semilla 42)
#      para poder elegir variantes sin mirar el conjunto de prueba.
#   2. Modelos imagen + metadatos (temperatura y dia), semilla 42.
#   3. Dos semillas extra (1 y 2) para los baselines y los modelos con metadatos.
# Uso: bash scripts/run_semillas.sh  (unos 90 min en una RTX 4070 Laptop)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

for m in resnet50 vit_small resnet50_scratch vit_small_scratch hybrid_fusion hybrid_vit_r26; do
  python scripts/train.py --config configs/$m.yaml --eval-only > logs/${m}_eval.log 2>&1
done

for m in resnet50_meta vit_small_meta; do
  python scripts/train.py --config configs/$m.yaml > logs/$m.log 2>&1
done

for s in 1 2; do
  for m in resnet50 vit_small resnet50_meta vit_small_meta; do
    python scripts/train.py --config configs/$m.yaml --seed $s --name ${m}_s$s > logs/${m}_s$s.log 2>&1
  done
done
echo "listo"
