"""Resultados con tres semillas, por temperatura y frente a la referencia sin imagen.

Responde tres observaciones del feedback del avance:

1. Una sola semilla no permite afirmar que un modelo le gana a otro. Cada modelo
   se entreno con las semillas 42, 1 y 2; se reporta media y desviacion estandar
   en test, y la diferencia ResNet - ViT semilla por semilla (bootstrap pareado
   por fruta en cada una).
2. Saber solo la temperatura y el dia ya predice bien la madurez. Se compara cada
   modelo con la referencia sin imagen (scripts/10_referencia_metadatos.py) y se
   separa el test en paltas "en calendario" (la referencia acierta) y "fuera de
   calendario" (la referencia falla por construccion): ahi se ve que aporta la foto.
3. La afirmacion "anda igual en las tres temperaturas" no se sostenia. Se
   reporta la accuracy por grupo de almacenamiento, con su variacion entre semillas.

Uso:
    python scripts/11_resumen_semillas.py
    python scripts/11_resumen_semillas.py --solo-figuras   # sin repetir el bootstrap
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.compare import paired_bootstrap_diff  # noqa: E402
from paltas.console import use_utf8  # noqa: E402
from paltas.metrics import compute_metrics  # noqa: E402
from paltas.paths import FIGURES_DIR, METRICS_DIR, ensure_dirs  # noqa: E402
from paltas.viz import save, set_style  # noqa: E402

use_utf8()

MODELOS = {
    "resnet50": "ResNet-50",
    "vit_small": "ViT-S/16",
    "resnet50_meta": "ResNet-50 + temp. y día",
    "vit_small_meta": "ViT-S/16 + temp. y día",
    "resnet50_asimetrico": "ResNet-50 (asim.)",
    "vit_small_asimetrico": "ViT-S/16 (asim.)",
}
# Mismos colores que el resto del proyecto (paltas.viz.MODEL_COLORS): ResNet
# azul y ViT rojo; sus versiones con metadatos, el mismo tono mas claro.
COLORES = {
    "resnet50": "#1f6fb2",
    "vit_small": "#d1495b",
    "resnet50_meta": "#8cbbe3",
    "vit_small_meta": "#eb9aa5",
    "resnet50_asimetrico": "#164f80",
    "vit_small_asimetrico": "#983442",
}
SEMILLAS = {42: "", 1: "_s1", 2: "_s2"}
METRICAS = ("accuracy", "qwk", "macro_f1", "mae", "asymmetric_mae",
            "underestimation_rate", "overestimation_rate")
REFERENCIA = "referencia_metadatos"
N_BOOT = 2000


def corridas(modelo: str) -> dict[int, str]:
    """Semilla -> nombre del experimento, solo para las corridas que existen."""
    out = {}
    for s, suf in SEMILLAS.items():
        nombre = f"{modelo}{suf}"
        if (METRICS_DIR / f"{nombre}_test_preds.csv").exists():
            out[s] = nombre
    return out


def preds(nombre: str, split: str = "test") -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / f"{nombre}_{split}_preds.csv")


def metricas(df: pd.DataFrame) -> dict:
    m = compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
    return {k: float(m[k]) for k in METRICAS}


def media_sd(valores: list[float]) -> dict:
    v = np.asarray(valores, dtype=float)
    return {"media": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def calendario_test() -> pd.Series | None:
    """file_name -> True si la referencia sin imagen acierta (palta 'en calendario')."""
    f = METRICS_DIR / f"{REFERENCIA}_test_preds.csv"
    if not f.exists():
        print(f"[aviso] falta {f.name}: corre scripts/10_referencia_metadatos.py")
        return None
    ref = pd.read_csv(f)
    return pd.Series((ref["y_true"] == ref["y_pred"]).to_numpy(), index=ref["file_name"])


def main() -> None:
    ensure_dirs()
    if "--solo-figuras" in sys.argv:
        # Regenera las figuras desde semillas.json sin repetir el bootstrap.
        s = json.loads((METRICS_DIR / "semillas.json").read_text(encoding="utf-8"))
        figuras(pd.DataFrame(s["por_semilla"]), s.get("referencia"),
                pd.DataFrame(s["por_temperatura"]), s.get("calendario"))
        return
    runs = {m: corridas(m) for m in MODELOS}
    runs = {m: r for m, r in runs.items() if r}
    for m, r in runs.items():
        print(f"{MODELOS[m]:26s} semillas: {sorted(r)}")

    en_cal = calendario_test()
    salida: dict = {"semillas": {m: sorted(r) for m, r in runs.items()}}

    # --- 1. media y desviacion por modelo (val y test) ---
    tabla, por_semilla = [], []
    for m, r in runs.items():
        fila = {"modelo": MODELOS[m], "semillas": len(r)}
        for split in ("val", "test"):
            ms = {s: metricas(preds(n, split)) for s, n in r.items()}
            for k in METRICAS:
                ag = media_sd([ms[s][k] for s in ms])
                fila[f"{split}_{k}"] = ag["media"]
                fila[f"{split}_{k}_sd"] = ag["sd"]
            if split == "test":
                for s, v in ms.items():
                    por_semilla.append({"modelo": m, "semilla": s, **v})
        tabla.append(fila)
    tabla = pd.DataFrame(tabla)
    salida["resumen"] = tabla.to_dict("records")
    salida["por_semilla"] = por_semilla

    ref = None
    if (METRICS_DIR / f"{REFERENCIA}_test_preds.csv").exists():
        ref = {"val": metricas(preds(REFERENCIA, "val")), "test": metricas(preds(REFERENCIA))}
        salida["referencia"] = ref

    print("\n=== Test: media ± desviación estándar entre semillas ===")
    for _, f in tabla.iterrows():
        print(f"{f['modelo']:26s} acc {100*f['test_accuracy']:.1f} ± {100*f['test_accuracy_sd']:.1f}"
              f"   QWK {f['test_qwk']:.4f} ± {f['test_qwk_sd']:.4f}"
              f"   F1 {f['test_macro_f1']:.3f} ± {f['test_macro_f1_sd']:.3f}"
              f"   MAE {f['test_mae']:.3f} ± {f['test_mae_sd']:.3f}")
    if ref:
        t = ref["test"]
        print(f"{'Referencia sin imagen':26s} acc {100*t['accuracy']:.1f}         "
              f"   QWK {t['qwk']:.4f}           F1 {t['macro_f1']:.3f}           "
              f"MAE {t['mae']:.3f}")

    # --- 2. comparaciones pareadas, semilla por semilla ---
    pares = [("resnet50", "vit_small"), ("resnet50_meta", "resnet50"),
             ("vit_small_meta", "vit_small"), ("resnet50_meta", "vit_small_meta"),
             ("resnet50_asimetrico", "resnet50"), ("vit_small_asimetrico", "vit_small")]
    comparaciones = []
    for a, b in pares:
        if a not in runs or b not in runs:
            continue
        for s in sorted(set(runs[a]) & set(runs[b])):
            pa, pb = preds(runs[a][s]), preds(runs[b][s])
            for k in ("qwk", "accuracy", "asymmetric_mae",
                      "underestimation_rate", "overestimation_rate"):
                d = paired_bootstrap_diff(pa, pb, k, n_boot=N_BOOT)
                comparaciones.append({"a": a, "b": b, "semilla": s, "metrica": k, **d})
    if ref:
        pref = preds(REFERENCIA)
        for a in runs:
            for s, n in runs[a].items():
                for k in ("qwk", "accuracy"):
                    d = paired_bootstrap_diff(preds(n), pref, k, n_boot=N_BOOT)
                    comparaciones.append({"a": a, "b": REFERENCIA, "semilla": s,
                                          "metrica": k, **d})
    salida["comparaciones"] = comparaciones
    print("\n=== Diferencias pareadas A - B en test (bootstrap por fruta) ===")
    print(pd.DataFrame(comparaciones)[["a", "b", "semilla", "metrica", "diff", "ci_low",
                                       "ci_high", "p_value"]].round(4).to_string(index=False))

    # --- 3. por temperatura y por calendario ---
    filas_grupo, filas_cal = [], []
    for m, r in runs.items():
        for s, n in r.items():
            p = preds(n)
            for g, sub in p.groupby("storage_group"):
                filas_grupo.append({"modelo": m, "semilla": s, "grupo": g,
                                    "accuracy": float((sub.y_true == sub.y_pred).mean())})
            if en_cal is not None:
                cal = en_cal.reindex(p["file_name"]).to_numpy()
                for etiqueta, mask in (("en calendario", cal), ("fuera de calendario", ~cal)):
                    sub = p[mask]
                    filas_cal.append({"modelo": m, "semilla": s, "subconjunto": etiqueta,
                                      "n": int(len(sub)),
                                      "accuracy": float((sub.y_true == sub.y_pred).mean())})
    if ref:
        pr = preds(REFERENCIA)
        for g, sub in pr.groupby("storage_group"):
            filas_grupo.append({"modelo": REFERENCIA, "semilla": 0, "grupo": g,
                                "accuracy": float((sub.y_true == sub.y_pred).mean())})

    grupo = (pd.DataFrame(filas_grupo).groupby(["modelo", "grupo"])["accuracy"]
             .agg(["mean", "std", "count"]).reset_index())
    salida["por_temperatura"] = grupo.to_dict("records")
    print("\n=== Accuracy en test por temperatura de almacenamiento ===")
    print(grupo.pivot(index="modelo", columns="grupo", values="mean").round(3).to_string())

    if filas_cal:
        cal = (pd.DataFrame(filas_cal).groupby(["modelo", "subconjunto"])
               .agg(n=("n", "first"), accuracy=("accuracy", "mean"),
                    sd=("accuracy", "std")).reset_index())
        salida["calendario"] = cal.to_dict("records")
        print("\n=== Accuracy en test: paltas en / fuera de calendario ===")
        print(cal.round(3).to_string(index=False))

    destino = METRICS_DIR / "semillas.json"
    destino.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    tabla.round(4).to_csv(METRICS_DIR / "semillas_tabla.csv", index=False)
    print(f"\nGuardado en {destino}")

    figuras(pd.DataFrame(por_semilla), ref, grupo, salida.get("calendario"))


def figuras(por_semilla: pd.DataFrame, ref: dict | None, grupo: pd.DataFrame,
            cal: list | None) -> None:
    set_style()
    modelos = [m for m in MODELOS if m in set(por_semilla["modelo"])]

    # 35: accuracy y QWK por semilla, con la referencia como linea horizontal
    # Ancha y baja a proposito: en la slide el alto es lo que escasea.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.3))
    for ax, k, titulo in ((axes[0], "accuracy", "Acierta la clase exacta"),
                          (axes[1], "qwk", "QWK (penaliza más los errores grandes)")):
        for i, m in enumerate(modelos):
            v = por_semilla.loc[por_semilla.modelo == m, k].to_numpy()
            ax.scatter(np.full(len(v), i), v, color=COLORES[m], s=70, zorder=3,
                       edgecolor="white")
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color=COLORES[m], lw=3)
        if ref:
            ax.axhline(ref["test"][k], color="#555", ls="--", lw=1.3)
            ax.text(-0.45, ref["test"][k], "solo temp. y día",
                    ha="left", va="bottom", fontsize=10, color="#555")
        ax.set_xticks(range(len(modelos)),
                      [MODELOS[m].replace(" + ", "\n+ ") for m in modelos], fontsize=10)
        ax.set_xlim(-0.5, len(modelos) - 0.5)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(titulo, fontsize=12)
        ax.grid(axis="x", visible=False)
        if k == "accuracy":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f} %"))
    # la explicacion "cada punto es una semilla" va en el subtitulo de la slide
    save(fig, FIGURES_DIR / "35_semillas.png")

    # 36: por temperatura y por calendario. Se guarda la figura combinada (para el
    # informe) y cada panel por separado (para que en las slides se vean grandes).
    paneles = 2 if cal else 1
    fig, axes = plt.subplots(1, paneles, figsize=(5.6 * paneles, 4.2), squeeze=False)
    panel_temperatura(axes[0, 0], modelos, grupo)
    if cal:
        panel_calendario(axes[0, 1], modelos, cal)
    save(fig, FIGURES_DIR / "36_temperatura_y_calendario.png")

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    panel_temperatura(ax, modelos, grupo, leyenda_fuera=True)
    save(fig, FIGURES_DIR / "36a_por_temperatura.png")
    if cal:
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        panel_calendario(ax, modelos, cal, leyenda=True)
        save(fig, FIGURES_DIR / "36b_calendario.png")


def panel_temperatura(ax, modelos: list[str], grupo: pd.DataFrame,
                      leyenda_fuera: bool = False) -> None:
    grupos = sorted(grupo["grupo"].unique())
    nombres_g = {"T10": "10 °C", "T20": "20 °C", "Tam": "Ambiente"}
    series = modelos + ([REFERENCIA] if REFERENCIA in set(grupo["modelo"]) else [])
    ancho = 0.8 / len(series)
    for j, m in enumerate(series):
        sub = grupo[grupo.modelo == m].set_index("grupo").reindex(grupos)
        ax.bar(np.arange(len(grupos)) + (j - (len(series) - 1) / 2) * ancho, sub["mean"],
               ancho, yerr=sub["std"].fillna(0), capsize=2,
               color=COLORES.get(m, "#999"), label=MODELOS.get(m, "Solo temp. y día"))
    ax.set_xticks(range(len(grupos)), [nombres_g.get(g, g) for g in grupos])
    ax.set_ylim(0.5, 0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f} %"))
    ax.set_title("Por temperatura de almacenamiento", fontsize=11)
    if leyenda_fuera:
        ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(axis="x", visible=False)


def panel_calendario(ax, modelos: list[str], cal: list, leyenda: bool = False) -> None:
    c = pd.DataFrame(cal)
    subs = ["en calendario", "fuera de calendario"]
    ancho = 0.8 / (len(modelos) + 1)
    ref_vals = [1.0, 0.0]
    for j, m in enumerate(modelos + [REFERENCIA]):
        if m == REFERENCIA:
            vals, err = ref_vals, [0, 0]
        else:
            sub = c[c.modelo == m].set_index("subconjunto").reindex(subs)
            vals, err = sub["accuracy"].to_numpy(), sub["sd"].fillna(0).to_numpy()
        xs = np.arange(2) + (j - len(modelos) / 2) * ancho
        ax.bar(xs, vals, ancho, yerr=err, capsize=2, color=COLORES.get(m, "#999"),
               label=MODELOS.get(m, "Solo temp. y día"))
        for x, val in zip(xs, vals):
            if val > 0:
                ax.text(x, val + 0.025, f"{100 * val:.0f}", ha="center", fontsize=7.5)
        if m == REFERENCIA:
            # la barra de 0 % no se ve: se rotula para que no parezca omitida
            ax.text(xs[1], 0.015, "0 %", ha="center", va="bottom", fontsize=8.5,
                    color="#555", fontweight="bold")
    n = c.groupby("subconjunto")["n"].first().reindex(subs)
    ax.set_xticks(range(2), [f"{s}\n({k} fotos)" for s, k in zip(subs, n)])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f} %"))
    ax.set_title("Paltas que maduran a ritmo normal vs. fuera de ritmo", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="x", visible=False)
    if leyenda:
        ax.legend(frameon=False, fontsize=8.5, loc="center left",
                  bbox_to_anchor=(1.01, 0.5))


if __name__ == "__main__":
    main()
