"""Analisis exploratorio: genera las figuras de reports/figures/.

Uso:
    python scripts/02_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.paths import (  # noqa: E402
    CLASS_NAMES,
    FIGURES_DIR,
    IMAGES_DIR,
    SPLITS_CSV,
    ensure_dirs,
)
from paltas.viz import (  # noqa: E402
    GROUP_COLORS,
    GROUP_LABEL,
    RIPENESS_COLORS,
    miles,
    save,
    set_style,
)

ORDEN_SPLIT = ["train", "val", "test"]
GRUPOS = ["T10", "T20", "Tam"]


def fig_class_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    counts = df["ripening_index"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color=RIPENESS_COLORS, edgecolor="white")
    for i, v in zip(counts.index, counts.values):
        axes[0].text(i, v + 60, miles(v), ha="center", fontsize=9)
    axes[0].set_title("Distribucion global de clases")
    axes[0].set_xlabel("Indice de madurez")
    axes[0].set_ylabel("Numero de imagenes")
    axes[0].set_xticks(range(1, 6))
    axes[0].set_ylim(0, counts.max() * 1.15)

    ct = pd.crosstab(df["storage_group"], df["ripening_index"], normalize="index").mul(100)
    bottom = np.zeros(len(ct))
    for c in range(1, 6):
        axes[1].barh(range(len(ct)), ct[c], left=bottom, color=RIPENESS_COLORS[c - 1],
                     edgecolor="white", label=str(c))
        bottom += ct[c].to_numpy()
    axes[1].set_yticks(range(len(ct)))
    axes[1].set_yticklabels([GROUP_LABEL[g] for g in ct.index])
    axes[1].set_title("Composicion de clases por grupo de almacenamiento")
    axes[1].set_xlabel("% de imagenes del grupo")
    axes[1].set_xlim(0, 100)
    axes[1].legend(title="Indice", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    axes[1].grid(axis="y", visible=False)

    fig.suptitle("Desbalance moderado (1,6x) y perfiles distintos por temperatura",
                 fontsize=12, y=1.03)
    save(fig, FIGURES_DIR / "01_distribucion_clases.png")


def fig_ripening_curves(df: pd.DataFrame) -> None:
    """Indice medio de madurez en funcion del dia, por grupo de almacenamiento."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for g in GRUPOS:
        sub = df[df["storage_group"] == g]
        m = sub.groupby("day")["ripening_index"].agg(["mean", "sem", "size"])
        m = m[m["size"] >= 10]
        ax.plot(m.index, m["mean"], marker="o", ms=4, lw=2,
                color=GROUP_COLORS[g], label=GROUP_LABEL[g])
        ax.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"],
                        color=GROUP_COLORS[g], alpha=0.18, lw=0)
    ax.set_xlabel("Dia del experimento")
    ax.set_ylabel("Indice de madurez medio")
    ax.set_ylim(0.8, 5.2)
    ax.set_yticks(range(1, 6))
    ax.axhline(4.5, ls="--", lw=1, color="#999")
    ax.text(ax.get_xlim()[1], 4.55, "fin de vida util  ", ha="right", va="bottom",
            fontsize=8, color="#666")
    ax.set_title("La refrigeracion desacopla el tiempo del estado de madurez")
    ax.legend(frameon=False, loc="lower right")
    save(fig, FIGURES_DIR / "02_curvas_maduracion.png")


def fig_examples(df: pd.DataFrame, n: int = 4, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(5, n, figsize=(2.0 * n, 10.4))
    for c in range(1, 6):
        sub = df[df["ripening_index"] == c]
        picks = sub.iloc[rng.choice(len(sub), n, replace=False)]
        for j, (_, r) in enumerate(picks.iterrows()):
            ax = axes[c - 1, j]
            ax.imshow(Image.open(IMAGES_DIR / r["image_file"]).resize((224, 224)))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(RIPENESS_COLORS[c - 1])
                s.set_linewidth(2.5)
            if j == 0:
                ax.set_ylabel(f"{c}\n{CLASS_NAMES[c]}", fontsize=8.5, rotation=0,
                              ha="right", va="center", labelpad=44)
    fig.suptitle("Ejemplos por clase · fondo uniforme, encuadre fijo, luz controlada",
                 fontsize=12, y=0.997)
    save(fig, FIGURES_DIR / "03_ejemplos_por_clase.png")


def fig_splits(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    frutas = df.drop_duplicates("sample_id")
    img = df["split"].value_counts().reindex(ORDEN_SPLIT)
    frt = frutas["split"].value_counts().reindex(ORDEN_SPLIT)
    x = np.arange(3)
    axes[0].bar(x - 0.2, img, 0.38, color="#1f6fb2")
    ax2 = axes[0].twinx()
    ax2.bar(x + 0.2, frt, 0.38, color="#d1495b")
    ax2.grid(False)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ORDEN_SPLIT)
    axes[0].set_ylabel("Imagenes", color="#1f6fb2")
    ax2.set_ylabel("Frutas (unidad de particion)", color="#d1495b")
    axes[0].set_title("Split 70/15/15 aplicado sobre frutas")
    for i, (a, b) in enumerate(zip(img, frt)):
        axes[0].text(i - 0.2, a + 120, miles(a), ha="center", fontsize=8)
        ax2.text(i + 0.2, b + 4, str(b), ha="center", fontsize=8)

    dist = pd.crosstab(df["split"], df["ripening_index"],
                       normalize="index").mul(100).reindex(ORDEN_SPLIT)
    bottom = np.zeros(3)
    for c in range(1, 6):
        axes[1].bar(ORDEN_SPLIT, dist[c], bottom=bottom, color=RIPENESS_COLORS[c - 1],
                    edgecolor="white", label=str(c))
        bottom += dist[c].to_numpy()
    axes[1].set_ylabel("% de la particion")
    axes[1].set_title("Distribucion de clases homogenea entre particiones")
    axes[1].legend(title="Indice", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    save(fig, FIGURES_DIR / "04_particiones.png")


def fig_photos_per_fruit(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    datos = [df[df["storage_group"] == g].groupby("sample_id").size() for g in GRUPOS]
    bp = ax.boxplot(datos, tick_labels=[GROUP_LABEL[g] for g in GRUPOS],
                    patch_artist=True, widths=0.55, medianprops=dict(color="black"))
    for patch, g in zip(bp["boxes"], GRUPOS):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.65)
    ax.set_ylabel("Fotos por fruta")
    ax.set_title("Las frutas refrigeradas aportan muchas mas fotos")
    save(fig, FIGURES_DIR / "05_fotos_por_fruta.png")


def main() -> None:
    ensure_dirs()
    set_style()
    df = pd.read_csv(SPLITS_CSV)
    print(f"EDA sobre {miles(len(df))} imagenes / {df['sample_id'].nunique()} frutas")
    fig_class_distribution(df)
    fig_ripening_curves(df)
    fig_examples(df)
    fig_splits(df)
    fig_photos_per_fruit(df)
    print("Listo.")


if __name__ == "__main__":
    main()
