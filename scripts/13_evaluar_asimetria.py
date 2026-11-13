"""Evaluacion de asimetria ordinal sobre predicciones guardadas (*_test_preds.csv).

Calcula `asymmetric_mae` (penalizacion 2x al error hacia inmaduro por riesgo de
etileno), `underestimation_rate` (% de falsos inmaduros) y `overestimation_rate`
(% de falsos maduros) sin necesidad de reentrenar, actualiza los *_test.json y
genera reports/figures/14_asimetria_errores.png.

Uso:
    python scripts/13_evaluar_asimetria.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.console import use_utf8  # noqa: E402
from paltas.metrics import compute_metrics  # noqa: E402
from paltas.paths import FIGURES_DIR, METRICS_DIR, ensure_dirs  # noqa: E402
from paltas.viz import save, set_style  # noqa: E402

use_utf8()

ORDEN_CANONICO = [
    "referencia_metadatos",
    "resnet50",
    "resnet50_asimetrico",
    "vit_small",
    "vit_small_asimetrico",
    "resnet50_meta",
    "vit_small_meta",
    "resnet50_scratch",
    "vit_small_scratch",
    "hybrid_fusion",
    "hybrid_vit_r26",
]

ORDEN_FIGURA = [
    "referencia_metadatos",
    "resnet50",
    "resnet50_asimetrico",
    "vit_small",
    "vit_small_asimetrico",
    "resnet50_meta",
    "vit_small_meta",
    "hybrid_fusion",
    "hybrid_vit_r26",
]

ETIQUETAS = {
    "referencia_metadatos": "Calendario\n(sin foto)",
    "resnet50": "ResNet-50",
    "vit_small": "ViT-S/16",
    "resnet50_asimetrico": "ResNet-50\n(asim.)",
    "vit_small_asimetrico": "ViT-S/16\n(asim.)",
    "resnet50_meta": "ResNet-50\n+ meta",
    "vit_small_meta": "ViT-S/16\n+ meta",
    "resnet50_scratch": "ResNet-50\n(scratch)",
    "vit_small_scratch": "ViT-S/16\n(scratch)",
    "hybrid_fusion": "Fusión\ntardía",
    "hybrid_vit_r26": "ViT-Hybrid\nR26",
}


def resumir_predicciones(
    preds_por_modelo: dict[str, pd.DataFrame],
    underripe_penalty: float = 2.0,
    overripe_penalty: float = 1.0,
) -> dict[str, dict]:
    """Calcula metricas simetricas y asimetricas para cada modelo."""
    salida: dict[str, dict] = {}
    for nombre, df in preds_por_modelo.items():
        m = compute_metrics(
            df["y_true"].to_numpy(),
            df["y_pred"].to_numpy(),
            underripe_penalty=underripe_penalty,
            overripe_penalty=overripe_penalty,
        )
        salida[nombre] = {
            "n": int(len(df)),
            "accuracy": round(m["accuracy"], 4),
            "qwk": round(m["qwk"], 4),
            "mae": round(m["mae"], 4),
            "asymmetric_mae": round(m["asymmetric_mae"], 4),
            "underestimation_rate": round(m["underestimation_rate"], 4),
            "overestimation_rate": round(m["overestimation_rate"], 4),
            "off_by_one": round(m["off_by_one"], 4),
        }
    return salida


def enriquecer_test_jsons(resumen: dict[str, dict]) -> list[Path]:
    """Agrega campos asimetricos y val a los *_test.json existentes si faltan."""
    actualizados = []
    for nombre, vals in resumen.items():
        fj = METRICS_DIR / f"{nombre}_test.json"
        if not fj.exists():
            continue
        data = json.loads(fj.read_text(encoding="utf-8"))
        test_block = data.setdefault("test", {})
        for k in ("asymmetric_mae", "underestimation_rate", "overestimation_rate"):
            test_block[k] = vals[k]

        # Enriquecer val desde *_val_preds.csv o desde *_history.json en best_epoch
        val_csv = METRICS_DIR / f"{nombre}_val_preds.csv"
        hist_json = METRICS_DIR / f"{nombre}_history.json"
        if val_csv.exists():
            df_val = pd.read_csv(val_csv)
            m_val = compute_metrics(df_val["y_true"].to_numpy(), df_val["y_pred"].to_numpy())
            val_block = data.setdefault("val", {})
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "qwk", "mae",
                      "asymmetric_mae", "underestimation_rate", "overestimation_rate", "off_by_one"):
                if k in m_val:
                    val_block[k] = round(float(m_val[k]), 4)
        elif "val" not in data and hist_json.exists():
            hist = json.loads(hist_json.read_text(encoding="utf-8"))
            best_ep = data.get("best_epoch")
            ep_row = next((r for r in hist if r.get("epoch") == best_ep), hist[-1] if hist else {})
            if ep_row:
                data["val"] = {
                    k: ep_row[k]
                    for k in ("accuracy", "balanced_accuracy", "macro_f1", "qwk", "mae",
                              "asymmetric_mae", "underestimation_rate", "overestimation_rate", "off_by_one")
                    if k in ep_row
                }

        fj.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        actualizados.append(fj)
    return actualizados


def fig_asimetria(resumen: dict[str, dict], out_path: Path = FIGURES_DIR / "14_asimetria_errores.png") -> None:
    """Compara direccion de error (falso inmaduro vs maduro) y MAE asimetrico."""
    set_style()
    modelos = [m for m in ORDEN_FIGURA if m in resumen]
    if not modelos:
        modelos = list(resumen)
    etiquetas = [ETIQUETAS.get(m, m) for m in modelos]
    x = np.arange(len(modelos))
    ancho = 0.36

    under_pct = [100.0 * resumen[m]["underestimation_rate"] for m in modelos]
    over_pct = [100.0 * resumen[m]["overestimation_rate"] for m in modelos]
    mae_sim = [resumen[m]["mae"] for m in modelos]
    mae_asim = [resumen[m]["asymmetric_mae"] for m in modelos]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))
    fig.subplots_adjust(wspace=0.26)

    # Panel izquierdo: direccion de errores (% falso inmaduro vs % falso maduro)
    b1 = axes[0].bar(x - ancho / 2, under_pct, ancho, label="Falso inmaduro (ŷ < y, riesgo etileno)",
                     color="#d1495b", edgecolor="white")
    b2 = axes[0].bar(x + ancho / 2, over_pct, ancho, label="Falso maduro (ŷ > y, menor costo)",
                     color="#1f6fb2", edgecolor="white")
    for b in list(b1) + list(b2):
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 0.35, f"{h:.1f}",
                     ha="center", va="bottom", fontsize=7.0)
    axes[0].set_xticks(x, etiquetas, fontsize=7.8)
    axes[0].set_ylabel("% de fotos en test")
    axes[0].set_ylim(0, max(under_pct + over_pct + [10]) * 1.24)
    axes[0].set_title("Dirección del error (%)", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].grid(axis="x", visible=False)

    # Panel derecho: MAE simetrico vs MAE Asimetrico (2x a falso inmaduro)
    b3 = axes[1].bar(x - ancho / 2, mae_sim, ancho, label="MAE simétrico (1× / 1×)",
                     color="#9AA5AE", edgecolor="white")
    b4 = axes[1].bar(x + ancho / 2, mae_asim, ancho, label="MAE Asimétrico (2×↓ / 1×↑)",
                     color="#6a4c93", edgecolor="white")
    for b in list(b3) + list(b4):
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=7.0)
    axes[1].set_xticks(x, etiquetas, fontsize=7.8)
    axes[1].set_ylabel("Error promedio (escalones ponderados)")
    axes[1].set_ylim(0, max(mae_sim + mae_asim + [0.3]) * 1.24)
    axes[1].set_title("MAE simétrico vs. asimétrico (2×↓)", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].grid(axis="x", visible=False)

    fig.suptitle("Evaluación asimétrica: equivocarse hacia inmaduro cuesta el doble",
                 fontsize=12, y=1.02)
    save(fig, out_path)


def cargar_preds_disponibles() -> dict[str, pd.DataFrame]:
    encontrados = {p.name.replace("_test_preds.csv", ""): p
                   for p in sorted(METRICS_DIR.glob("*_test_preds.csv"))}
    orden = [m for m in ORDEN_CANONICO if m in encontrados]
    orden += [m for m in encontrados if m not in orden]
    return {m: pd.read_csv(encontrados[m]) for m in orden}


def main() -> None:
    ensure_dirs()
    preds = cargar_preds_disponibles()
    if not preds:
        raise SystemExit(f"No hay archivos *_test_preds.csv en {METRICS_DIR}")

    resumen = resumir_predicciones(preds)
    enriquecer_test_jsons(resumen)

    out_json = METRICS_DIR / "asimetria_resumen.json"
    out_json.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    filas = [{"modelo": k, **resumen[k]} for k in ORDEN_CANONICO if k in resumen]
    print("=== Resumen de asimetria ordinal (2x falso inmaduro, 1x falso maduro) ===")
    print(pd.DataFrame(filas).to_string(index=False))
    fig_asimetria(resumen)
    print(f"Guardado: {out_json}")


if __name__ == "__main__":
    main()
