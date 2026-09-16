#!/usr/bin/env bash
# Entrena los dos baselines del avance, secuencialmente (una sola GPU).
set -e
python scripts/train.py --config configs/resnet50.yaml
python scripts/train.py --config configs/vit_small.yaml
