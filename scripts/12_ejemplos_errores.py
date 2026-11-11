"""Ejemplos de errores con sus fotos (pedido del feedback del avance).

Una tabla de metricas no muestra COMO se equivoca el modelo. Esta figura pone
fotos reales de test en tres grupos:

1. Errores de la ResNet-50 en paltas "fuera de calendario" (donde la referencia
   sin imagen falla por construccion): que ve la red cuando el calendario engana.
2. Errores de la ResNet-50 en paltas "en calendario": casos que el calendario
   resolvia y la foto no.
3. Errores que comete incluso el modelo con temperatura y dia.

Se usa la semilla 42 de cada modelo. Los ejemplos se eligen al azar (semilla
fija) dentro de cada grupo, no a mano, para no seleccionar los mas vistosos.

Uso:
    python scripts/12_ejemplos_errores.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.console import use_utf8  # noqa: E402
from paltas.paths import CACHE_DIR, FIGURES_DIR, IMAGES_DIR, METRICS_DIR  # noqa: E402
from paltas.viz import RIPENESS_COLORS, save, set_style  # noqa: E402

use_utf8()

POR_FILA = 5
GRUPO = {"T10": "10 °C", "T20": "20 °C", "Tam": "ambiente"}


def cargar() -> pd.DataFrame:
    r = pd.read_csv(METRICS_DIR / "resnet50_test_preds.csv")
    m = pd.read_csv(METRICS_DIR / "resnet50_meta_test_preds.csv")
    ref = pd.read_csv(METRICS_DIR / "referencia_metadatos_test_preds.csv")
    df = r[["file_name", "storage_group", "day", "y_true", "y_pred"]].rename(
        columns={"y_pred": "pred_img"})
    df = df.merge(m[["file_name", "y_pred"]].rename(columns={"y_pred": "pred_meta"}),
                  on="file_name")
    df = df.merge(ref[["file_name", "y_pred"]].rename(columns={"y_pred": "pred_cal"}),
                  on="file_name")
    return df


def foto(file_name: str) -> Image.Image:
    for base in (CACHE_DIR, IMAGES_DIR):
        p = base / f"{file_name}.jpg"
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(file_name)


def main() -> None:
    set_style()
    df = cargar()
    grupos = [
        ("La palta va fuera de ritmo y la foto tampoco acierta",
         df[(df.pred_cal != df.y_true) & (df.pred_img != df.y_true)], "pred_img"),
        ("Va a ritmo normal: el calendario acierta y la foto no",
         df[(df.pred_cal == df.y_true) & (df.pred_img != df.y_true)], "pred_img"),
        ("Se equivoca incluso el modelo con foto + temperatura y día",
         df[df.pred_meta != df.y_true], "pred_meta"),
    ]

    fig, axes = plt.subplots(len(grupos), POR_FILA, figsize=(11, 7.4))
    for fila, (titulo, sub, col) in enumerate(grupos):
        muestra = sub.sample(min(POR_FILA, len(sub)), random_state=0)
        for k, ax in enumerate(axes[fila]):
            ax.axis("off")
            if k >= len(muestra):
                continue
            f = muestra.iloc[k]
            ax.imshow(foto(f.file_name))
            real, pred = int(f.y_true) + 1, int(f[col]) + 1
            ax.set_title(f"real {real}  →  dice {pred}", fontsize=9.5,
                         color=RIPENESS_COLORS[pred - 1], fontweight="bold")
            ax.text(0.5, -0.06, f"día {int(f.day)}, {GRUPO.get(f.storage_group)}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8,
                    color="#555")
        axes[fila][0].text(-0.08, 1.28, f"{titulo}  ({len(sub)} fotos de 2.262)",
                           transform=axes[fila][0].transAxes, fontsize=10.5,
                           fontweight="bold", ha="left")
    fig.subplots_adjust(hspace=0.55, wspace=0.08)
    save(fig, FIGURES_DIR / "37_ejemplos_errores.png")

    print("Errores de la ResNet-50 (solo imagen):",
          int((df.pred_img != df.y_true).sum()))
    print("  de esos, fuera de calendario:", len(grupos[0][1]),
          "| en calendario:", len(grupos[1][1]))
    print("Errores del modelo con temperatura y dia:", len(grupos[2][1]))
    print("Distancia de los errores (ResNet):",
          (df.pred_img - df.y_true)[df.pred_img != df.y_true].abs().value_counts().to_dict())


if __name__ == "__main__":
    main()
