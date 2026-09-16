"""Compara los modelos entrenados: tabla, figuras y tests estadisticos.

Lee reports/metrics/<name>_test.json y <name>_test_preds.csv de cada modelo que
se le indique y produce la evidencia de la comparacion.

Uso:
    python scripts/compare_models.py --models resnet50 vit_small
    python scripts/compare_models.py --models resnet50 vit_small hybrid_fusion --tag final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.compare import cluster_bootstrap_ci, disagreement_table, paired_bootstrap_diff  # noqa: E402
from paltas.metrics import METRIC_NAMES  # noqa: E402
from paltas.paths import CLASS_NAMES, FIGURES_DIR, METRICS_DIR, ensure_dirs  # noqa: E402
from paltas.viz import MODEL_COLORS, RIPENESS_COLORS, save, set_style  # noqa: E402
from paltas.console import use_utf8  # noqa: E402

use_utf8()

ETIQUETAS = {
    "resnet50": "ResNet-50",
    "vit_small": "ViT-S/16",
    "resnet50_scratch": "ResNet-50 (scratch)",
    "vit_small_scratch": "ViT-S/16 (scratch)",
    "hybrid_fusion": "Fusion tardia",
    "hybrid_vit_r26": "ViT-Hybrid R26",
}


def etiqueta(m: str) -> str:
    return ETIQUETAS.get(m, m)


def color(m: str) -> str:
    base = m.replace("_scratch", "")
    c = MODEL_COLORS.get(base, "#888")
    return c


def cargar(models: list[str]) -> tuple[dict, dict, dict]:
    res, preds, hist = {}, {}, {}
    for m in models:
        fj = METRICS_DIR / f"{m}_test.json"
        fp = METRICS_DIR / f"{m}_test_preds.csv"
        if not fj.exists():
            raise FileNotFoundError(f"Falta {fj}. Entrena primero: "
                                    f"python scripts/train.py --config configs/{m}.yaml")
        res[m] = json.loads(fj.read_text(encoding="utf-8"))
        preds[m] = pd.read_csv(fp)
        fh = METRICS_DIR / f"{m}_history.json"
        if fh.exists():
            hist[m] = pd.DataFrame(json.loads(fh.read_text(encoding="utf-8")))
    return res, preds, hist


def tabla_principal(models: list[str], res: dict, preds: dict, n_boot: int) -> pd.DataFrame:
    filas = []
    for m in models:
        r = res[m]
        punto, lo, hi = cluster_bootstrap_ci(preds[m], "qwk", n_boot=n_boot)
        acc, acc_lo, acc_hi = cluster_bootstrap_ci(preds[m], "accuracy", n_boot=n_boot)
        filas.append({
            "modelo": etiqueta(m),
            "params (M)": r["params_M"],
            "GFLOPs": r["gflops"],
            "min. entren.": r["train_minutes"],
            "epoca best": r["best_epoch"],
            "Accuracy": round(acc, 4),
            "Acc. IC95%": f"[{acc_lo:.3f}, {acc_hi:.3f}]",
            "Macro-F1": round(r["test"]["macro_f1"], 4),
            "QWK": round(punto, 4),
            "QWK IC95%": f"[{lo:.3f}, {hi:.3f}]",
            "MAE": round(r["test"]["mae"], 4),
            "Acc. ±1": round(r["test"]["off_by_one"], 4),
        })
    return pd.DataFrame(filas)


def fig_confusiones(models: list[str], res: dict, tag: str) -> None:
    # Con mas de tres modelos una sola fila queda ilegible: envolvemos en dos.
    n = len(models)
    ncols = n if n <= 3 else (n + 1) // 2
    nrows = 1 if n <= 3 else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, m in zip(axes, models):
        cm = np.array(res[m]["test"]["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                if cmn[i, j] > 0.005:
                    ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                            fontsize=8.5, color="white" if cmn[i, j] > 0.55 else "#222")
        ax.set_xticks(range(5), range(1, 6))
        ax.set_yticks(range(5), range(1, 6))
        ax.set_xlabel("Prediccion")
        if list(axes).index(ax) % ncols == 0:
            ax.set_ylabel("Etiqueta real")
        ax.set_title(f"{etiqueta(m)}\nQWK {res[m]['test']['qwk']:.3f} · "
                     f"acc {res[m]['test']['accuracy']:.3f}", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label="% de la fila")
    fig.suptitle("Los errores se concentran en la diagonal vecina: el modelo confunde "
                 "estados adyacentes, no extremos",
                 fontsize=11, y=1.06 if nrows == 1 else 0.99)
    save(fig, FIGURES_DIR / f"10_matrices_confusion_{tag}.png")


def fig_curvas(models: list[str], hist: dict, tag: str) -> None:
    if not hist:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for m, h in hist.items():
        c = color(m)
        ls = "--" if "scratch" in m else "-"
        axes[0].plot(h["epoch"], h["train_loss"], ls, color=c, label=etiqueta(m), lw=2)
        axes[1].plot(h["epoch"], h["qwk"], ls, color=c, marker="o", ms=3, lw=2)
        axes[2].plot(h["epoch"], h["accuracy"], ls, color=c, marker="o", ms=3, lw=2)
    axes[0].set_title("Perdida de entrenamiento")
    axes[1].set_title("QWK en validacion")
    axes[2].set_title("Accuracy en validacion")
    for ax in axes:
        ax.set_xlabel("Epoca")
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, FIGURES_DIR / f"11_curvas_entrenamiento_{tag}.png")


def fig_f1_por_clase(models: list[str], res: dict, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ancho = 0.8 / len(models)
    x = np.arange(5)
    for k, m in enumerate(models):
        f1 = res[m]["test"]["per_class_f1"]
        ax.bar(x + (k - (len(models) - 1) / 2) * ancho, f1, ancho,
               label=etiqueta(m), color=color(m),
               alpha=0.55 if "scratch" in m else 1.0,
               hatch="//" if "scratch" in m else None, edgecolor="white")
    ax.set_xticks(x, [f"{i+1}\n{CLASS_NAMES[i+1].split(' (')[0]}" for i in range(5)], fontsize=8)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("F1 por clase: el cuello de botella son los estados intermedios")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    save(fig, FIGURES_DIR / f"12_f1_por_clase_{tag}.png")


def fig_costo_beneficio(models: list[str], res: dict, tag: str) -> None:
    """Accuracy contra costo: la figura que respalda la decision velocidad/precision."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # ResNet-50 y ViT-S/16 estan casi superpuestos en GFLOPs (8,17 vs 8,48): sin
    # alternar la posicion de la etiqueta, una tapa el marcador de la otra.
    for k, m in enumerate(models):
        r = res[m]
        c = color(m)
        mk = "s" if "hybrid" in m else ("^" if "scratch" in m else "o")
        dy = 13 if k % 2 == 0 else -20
        for ax, x in ((axes[0], r["params_M"]), (axes[1], r["gflops"])):
            ax.scatter(x, r["test"]["qwk"], s=150, color=c, marker=mk,
                       edgecolor="white", zorder=3)
            ax.annotate(etiqueta(m), (x, r["test"]["qwk"]), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8, color=c, zorder=4)
    axes[0].set_xlabel("Parametros (M)")
    axes[1].set_xlabel("GFLOPs por imagen")
    for ax in axes:
        ax.set_ylabel("QWK en test")
        ax.margins(x=0.22, y=0.22)
    fig.suptitle("Costo computacional frente a desempeno", fontsize=12, y=1.02)
    save(fig, FIGURES_DIR / f"13_costo_beneficio_{tag}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tag", default="comparacion")
    args = ap.parse_args()

    ensure_dirs()
    set_style()
    res, preds, hist = cargar(args.models)

    print("=" * 100)
    print("TABLA COMPARATIVA (test, IC95% por bootstrap agrupado por fruta)")
    print("=" * 100)
    tabla = tabla_principal(args.models, res, preds, args.n_boot)
    print(tabla.to_string(index=False))
    tabla.to_csv(METRICS_DIR / f"{args.tag}_tabla.csv", index=False)

    # Registramos n_boot: el numero de replicas determina la resolucion del
    # p-valor, y sin dejarlo escrito es facil comparar salidas de corridas con
    # distinto presupuesto de bootstrap y creer que el resultado cambio.
    salida = {"n_boot": args.n_boot, "modelos": args.models,
              "tabla": tabla.to_dict("records"), "comparaciones": [], "desacuerdo": {}}

    # --- Comparaciones pareadas contra el primer modelo de la lista ---
    base = args.models[0]
    for m in args.models[1:]:
        print("\n" + "-" * 100)
        print(f"{etiqueta(m)}  vs  {etiqueta(base)}   (bootstrap pareado, {args.n_boot} replicas)")
        comp = {"modelo_a": m, "modelo_b": base, "metricas": {}}
        for met in ("qwk", "accuracy", "macro_f1"):
            d = paired_bootstrap_diff(preds[m], preds[base], met, n_boot=args.n_boot)
            comp["metricas"][met] = d
            veredicto = "SIGNIFICATIVA" if d["significativo"] else "no significativa"
            print(f"  Δ {METRIC_NAMES.get(met, met):<15} = {d['diff']:+.4f}  "
                  f"IC95% [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
                  f"p={d['p_value']:.3f}  -> {veredicto}")
        salida["comparaciones"].append(comp)

        dis = disagreement_table(preds[m], preds[base], etiqueta(m), etiqueta(base))
        print(f"\n  Desacuerdo entre {etiqueta(m)} y {etiqueta(base)}:")
        print("  " + dis.to_string(index=False).replace("\n", "\n  "))
        print(f"  Techo de un oraculo que eligiera el modelo correcto: "
              f"{dis.attrs['oraculo']:.4f}")
        salida["desacuerdo"][f"{m}_vs_{base}"] = {
            "tabla": dis.to_dict("records"),
            "oraculo_accuracy": dis.attrs["oraculo"],
            "ambos_fallan": dis.attrs["ambos_fallan"],
        }

    print("\nGenerando figuras...")
    fig_confusiones(args.models, res, args.tag)
    fig_curvas(args.models, hist, args.tag)
    fig_f1_por_clase(args.models, res, args.tag)
    fig_costo_beneficio(args.models, res, args.tag)

    (METRICS_DIR / f"{args.tag}.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultados en {METRICS_DIR / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
