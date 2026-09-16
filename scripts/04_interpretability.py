"""Grad-CAM (ResNet) y attention rollout (ViT) sobre imagenes de test.

Uso:
    python scripts/04_interpretability.py
    python scripts/04_interpretability.py --mode disagreement   # casos donde difieren
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.data import build_transforms  # noqa: E402
from paltas.interpret import GradCAM, attention_rollout, denormalize, resnet_target_layer  # noqa: E402
from paltas.models import build_model  # noqa: E402
from paltas.paths import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    FIGURES_DIR,
    IMAGES_DIR,
    METRICS_DIR,
    SPLITS_CSV,
    ensure_dirs,
)
from paltas.viz import save, set_style  # noqa: E402


def cargar(nombre: str, arquitectura: str, device):
    ckpt = CHECKPOINT_DIR / f"{nombre}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Falta {ckpt}; entrena {arquitectura} primero.")
    model = build_model(arquitectura, pretrained=False)
    estado = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(estado)
    return model.to(device).eval()


def elegir_imagenes(mode: str, n_por_clase: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_CSV)
    test = df[df["split"] == "test"]

    if mode == "disagreement":
        pr, pv = (METRICS_DIR / "resnet50_test_preds.csv", METRICS_DIR / "vit_small_test_preds.csv")
        if pr.exists() and pv.exists():
            a = pd.read_csv(pr).sort_values("file_name").reset_index(drop=True)
            b = pd.read_csv(pv).sort_values("file_name").reset_index(drop=True)
            dif = a[a["y_pred"] != b["y_pred"]]["file_name"]
            test = test[test["file_name"].isin(dif)]
            print(f"{len(dif)} imagenes donde ResNet y ViT predicen distinto")

    rng = np.random.default_rng(seed)
    partes = []
    for c in range(1, 6):
        sub = test[test["ripening_index"] == c]
        if len(sub) == 0:
            continue
        k = min(n_por_clase, len(sub))
        partes.append(sub.iloc[rng.choice(len(sub), k, replace=False)])
    return pd.concat(partes).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "disagreement"], default="random")
    ap.add_argument("--n-por-clase", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--resnet", default="resnet50")
    ap.add_argument("--vit", default="vit_small")
    args = ap.parse_args()

    ensure_dirs()
    set_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet = cargar(args.resnet, "resnet50", device)
    vit = cargar(args.vit, "vit_small", device)

    img_dir = CACHE_DIR if any(CACHE_DIR.glob("*.jpg")) else IMAGES_DIR
    tf = build_transforms(224, train=False)
    seleccion = elegir_imagenes(args.mode, args.n_por_clase, args.seed)

    filas = len(seleccion)
    fig, axes = plt.subplots(filas, 3, figsize=(8.4, 2.85 * filas))
    axes = np.atleast_2d(axes)

    cam = GradCAM(resnet, resnet_target_layer(resnet))
    try:
        for i, (_, r) in enumerate(seleccion.iterrows()):
            pil = Image.open(img_dir / r["image_file"]).convert("RGB")
            x = tf(pil).unsqueeze(0).to(device)
            base = denormalize(x[0])

            mapa_cam, pred_r = cam(x)
            mapa_att, pred_v = attention_rollout(vit, x)
            real = int(r["ripening_index"])

            for j, (m, titulo, pred) in enumerate([
                (None, "Entrada", None),
                (mapa_cam, "ResNet-50 · Grad-CAM", pred_r + 1),
                (mapa_att, "ViT-S/16 · attention rollout", pred_v + 1),
            ]):
                ax = axes[i, j]
                ax.imshow(base)
                if m is not None:
                    ax.imshow(m, cmap="jet", alpha=0.45)
                    ok = "✓" if pred == real else "✗"
                    ax.set_xlabel(f"pred {pred} {ok}", fontsize=9,
                                  color="#1a7f37" if pred == real else "#b3261e")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if i == 0:
                    ax.set_title(titulo, fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"real: {real}\n{CLASS_NAMES[real].split(' (')[0]}",
                                  fontsize=8.5, rotation=0, ha="right", va="center", labelpad=40)
    finally:
        cam.remove()

    sufijo = "desacuerdo" if args.mode == "disagreement" else "aleatorio"
    fig.suptitle("Donde mira cada arquitectura", fontsize=12, y=1.005)
    save(fig, FIGURES_DIR / f"20_interpretabilidad_{sufijo}.png")


if __name__ == "__main__":
    main()
