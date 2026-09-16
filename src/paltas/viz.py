"""Estilo comun para todas las figuras del proyecto."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paleta secuencial: el indice de madurez es ordinal, asi que el color debe
# progresar de verde a cafe en lugar de usar categorias arbitrarias.
RIPENESS_COLORS = ["#4C9A2A", "#8FB339", "#C9A227", "#8C5E2A", "#4A3222"]
MODEL_COLORS = {"resnet50": "#1f6fb2", "vit_small": "#d1495b",
                "hybrid_fusion": "#6a4c93", "hybrid_vit_r26": "#2a9d8f"}
GROUP_LABEL = {"T10": "T10 · 10 °C, 85 % HR",
               "T20": "T20 · 20 °C, 85 % HR",
               "Tam": "Tamb · ambiente"}
GROUP_COLORS = {"T10": "#2a6f97", "T20": "#e07a5f", "Tam": "#81b29a"}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })


def miles(n) -> str:
    """Formatea un entero con punto como separador de miles (convencion es-CL)."""
    return f"{int(n):,}".replace(",", ".")


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figura -> {path.name}")
