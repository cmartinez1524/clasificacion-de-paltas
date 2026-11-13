#!/usr/bin/env bash
# Pipeline completo de punta a punta. Requiere el dataset descargado desde
# Mendeley (ver README). Tiempo aproximado en una RTX 4070 Laptop: ~2 h.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5  Preparacion de datos =="
python scripts/00_build_manifest.py
python scripts/01_make_splits.py
python scripts/03_cache_images.py
python scripts/02_eda.py

echo "== 2/5  Baselines equiparables =="
python scripts/train.py --config configs/resnet50.yaml
python scripts/train.py --config configs/vit_small.yaml
python scripts/compare_models.py --models resnet50 vit_small --tag avance

echo "== 3/5  Ablacion sin pre-entrenamiento =="
python scripts/train.py --config configs/resnet50_scratch.yaml
python scripts/train.py --config configs/vit_small_scratch.yaml

echo "== 4/5  Hibridos y entrenamiento asimetrico =="
python scripts/train.py --config configs/hybrid_fusion.yaml
python scripts/train.py --config configs/hybrid_vit_r26.yaml
python scripts/train.py --config configs/resnet50_asimetrico.yaml
python scripts/train.py --config configs/vit_small_asimetrico.yaml

echo "== 5/5  Analisis y entregables =="
python scripts/13_evaluar_asimetria.py
python scripts/compare_models.py \
  --models resnet50 vit_small resnet50_asimetrico vit_small_asimetrico resnet50_scratch vit_small_scratch hybrid_fusion hybrid_vit_r26 \
  --tag final
python scripts/04_interpretability.py --mode disagreement

echo "== Entrega final: referencia sin imagen, metadatos y tres semillas =="
python scripts/10_referencia_metadatos.py
bash scripts/run_semillas.sh
python scripts/11_resumen_semillas.py
python scripts/09_mejoras_inferencia.py
python scripts/12_ejemplos_errores.py
python scripts/08_figuras_didacticas.py
python scripts/07_deck_latex.py

echo "Listo. Demo:  python app/gradio_app.py"
