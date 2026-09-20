"""Dos mejoras que no requieren reentrenar nada.

1. TTA con volteo horizontal: se evalua tambien la foto espejada y se promedian
   las dos salidas. El parametro ya existia en el codigo (`eval_tta_hflip`) pero
   nunca se habia activado.

2. Fusion de las dos caras: el dataset fotografia cada palta por ambos lados el
   mismo dia, y en test los 1.131 pares a/b estan completos. Hoy clasificamos
   cada foto por separado; promediando las probabilidades de las dos caras se
   obtiene una prediccion por fruta-dia, que ademas es lo que se haria en una
   planta real.

Ojo con la comparacion: las filas "por fruta-dia" se calculan sobre 1.131
observaciones y las "por foto" sobre 2.262. No es la misma poblacion, asi que
no se comparan entre si directamente; lo que interesa es que la de fruta-dia
es la que refleja el uso operativo.

Uso:
    python scripts/09_mejoras_inferencia.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.console import use_utf8  # noqa: E402
from paltas.data import build_dataloaders  # noqa: E402
from paltas.metrics import compute_metrics  # noqa: E402
from paltas.models import build_model  # noqa: E402
from paltas.paths import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_DIR,
    IMAGES_DIR,
    METRICS_DIR,
    SPLITS_CSV,
    ensure_dirs,
)

use_utf8()

ARQUITECTURA = {
    "resnet50": "resnet50",
    "vit_small": "vit_small",
    "hybrid_fusion": "hybrid_fusion",
    "hybrid_vit_r26": "hybrid_vit_r26",
}
ETIQUETA = {
    "resnet50": "ResNet-50",
    "vit_small": "ViT-S/16",
    "hybrid_fusion": "Híbrido fusión",
    "hybrid_vit_r26": "ViT-Hybrid R26",
}


@torch.no_grad()
def probabilidades(model, loader, device, tta: bool) -> np.ndarray:
    """Softmax de cada imagen del loader, en el orden del DataFrame."""
    model.eval()
    salida = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x).float()
            if tta:
                logits = (logits + model(torch.flip(x, dims=[3])).float()) / 2
        salida.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(salida).numpy()


def metricas_por_foto(df: pd.DataFrame, probs: np.ndarray) -> dict:
    return compute_metrics(df["ripening_index"].to_numpy() - 1, probs.argmax(1))


def metricas_por_fruta_dia(df: pd.DataFrame, probs: np.ndarray) -> tuple[dict, int]:
    """Promedia las probabilidades de las dos caras y clasifica la fruta-dia."""
    aux = df[["sample_id", "day", "ripening_index"]].copy().reset_index(drop=True)
    for c in range(probs.shape[1]):
        aux[f"p{c}"] = probs[:, c]
    cols = [f"p{c}" for c in range(probs.shape[1])]
    agg = aux.groupby(["sample_id", "day"], as_index=False).agg(
        {**{c: "mean" for c in cols}, "ripening_index": "first"})
    y_pred = agg[cols].to_numpy().argmax(1)
    return compute_metrics(agg["ripening_index"].to_numpy() - 1, y_pred), len(agg)


def fila(nombre: str, m: dict, n: int) -> dict:
    return {"variante": nombre, "n": n,
            "accuracy": round(m["accuracy"], 4),
            "qwk": round(m["qwk"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "mae": round(m["mae"], 4),
            "off_by_one": round(m["off_by_one"], 4)}


def fig_mejoras(todo: dict) -> None:
    """Tres barras por modelo: linea base, fusion de caras, y ademas TTA."""
    import matplotlib.pyplot as plt

    from paltas.paths import FIGURES_DIR
    from paltas.viz import save, set_style

    set_style()
    modelos = list(todo)
    variantes = [("por foto", "Una foto a la vez\n(línea base)", "#9AA5AE"),
                 ("por fruta-día", "Promediando las dos\ncaras de la palta", "#2E6B2F"),
                 ("por fruta-día + TTA", "Las dos caras\n+ foto espejada", "#8FB339")]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(modelos))
    ancho = 0.26
    for k, (clave, etiqueta, color) in enumerate(variantes):
        vals = [next(f["accuracy"] for f in todo[m] if f["variante"] == clave)
                for m in modelos]
        barras = ax.bar(x + (k - 1) * ancho, vals, ancho, label=etiqueta, color=color,
                        edgecolor="white")
        for b, val in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.004,
                    f"{100*val:.1f}".replace(".", ","), ha="center", fontsize=8.5)

    ax.set_xticks(x, [ETIQUETA[m] for m in modelos], fontsize=10)
    ax.set_ylabel("Acierta la clase exacta")
    ax.set_ylim(0.68, 0.84)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f} %"))
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Dos mejoras que no requieren reentrenar nada", fontsize=12.5)
    save(fig, FIGURES_DIR / "34_mejoras_inferencia.png")


def main() -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = CACHE_DIR if any(CACHE_DIR.glob("*.jpg")) else IMAGES_DIR
    loaders, frames = build_dataloaders(SPLITS_CSV, images_dir, 224, 64, 8, 42)
    df_test = frames["test"]

    todo = {}
    for clave, arq in ARQUITECTURA.items():
        ckpt = CHECKPOINT_DIR / f"{clave}.pt"
        if not ckpt.exists():
            print(f"[aviso] falta {ckpt.name}, lo salto")
            continue
        model = build_model(arq, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                         weights_only=False)["model"])
        model.to(device)

        filas = []
        for tta in (False, True):
            probs = probabilidades(model, loaders["test"], device, tta)
            sufijo = " + TTA" if tta else ""
            filas.append(fila(f"por foto{sufijo}", metricas_por_foto(df_test, probs),
                              len(df_test)))
            m_fd, n_fd = metricas_por_fruta_dia(df_test, probs)
            filas.append(fila(f"por fruta-día{sufijo}", m_fd, n_fd))

        todo[clave] = filas
        print(f"\n=== {ETIQUETA[clave]} ===")
        print(pd.DataFrame(filas).to_string(index=False))
        del model
        torch.cuda.empty_cache()

    destino = METRICS_DIR / "mejoras_inferencia.json"
    destino.write_text(json.dumps(todo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {destino}")

    # resumen: cuanto aporta cada mejora sobre la linea base de cada modelo
    print("\n=== Ganancia respecto de 'por foto' (línea base actual) ===")
    resumen = []
    for clave, filas in todo.items():
        base = filas[0]
        for f in filas[1:]:
            resumen.append({
                "modelo": ETIQUETA[clave], "variante": f["variante"],
                "Δ accuracy": round(f["accuracy"] - base["accuracy"], 4),
                "Δ QWK": round(f["qwk"] - base["qwk"], 4),
                "Δ MAE": round(f["mae"] - base["mae"], 4),
                "Δ ±1": round(f["off_by_one"] - base["off_by_one"], 4),
            })
    print(pd.DataFrame(resumen).to_string(index=False))

    fig_mejoras(todo)


if __name__ == "__main__":
    main()
