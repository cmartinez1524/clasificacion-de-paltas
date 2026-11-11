"""Referencia sin imagen: predecir la madurez solo con temperatura y dia.

Observacion del feedback del avance: la madurez sigue de cerca el calendario de
almacenamiento, asi que una regla que no mira la foto puede ser una referencia
exigente. Si un modelo de imagen no la supera, la foto no esta aportando nada
que no diga ya la etiqueta del lote.

Regla: para cada par (grupo de almacenamiento, dia) se predice la moda del indice
de madurez en train. Si el par no aparece en train se usa el dia mas cercano del
mismo grupo. Es el clasificador optimo (en accuracy) que solo ve esas dos
variables, estimado sin tocar val ni test.

Salidas:
    reports/metrics/referencia_metadatos.json
    reports/metrics/referencia_metadatos_{val,test}_preds.csv  (mismo formato que
        los *_preds.csv de los modelos, para compararlos con compare_models.py)

Uso:
    python scripts/10_referencia_metadatos.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.compare import cluster_bootstrap_ci  # noqa: E402
from paltas.console import use_utf8  # noqa: E402
from paltas.metrics import compute_metrics  # noqa: E402
from paltas.paths import METRICS_DIR, SPLITS_CSV, ensure_dirs  # noqa: E402

use_utf8()

NOMBRE = "referencia_metadatos"
N_BOOT = 2000
METRICAS = ("accuracy", "qwk", "macro_f1", "mae", "off_by_one")


def ajustar(train: pd.DataFrame) -> dict[tuple[str, int], int]:
    """(grupo, dia) -> moda del indice de madurez en train (empates: el menor)."""
    moda = (train.groupby(["storage_group", "day"])["ripening_index"]
            .agg(lambda s: s.value_counts().sort_index().idxmax()))
    return moda.to_dict()


def predecir(tabla: dict[tuple[str, int], int], df: pd.DataFrame) -> pd.Series:
    dias_por_grupo: dict[str, list[int]] = {}
    for g, d in tabla:
        dias_por_grupo.setdefault(g, []).append(d)

    def regla(g: str, d: int) -> int:
        if (g, d) in tabla:
            return tabla[(g, d)]
        cercano = min(dias_por_grupo[g], key=lambda x: (abs(x - d), x))
        return tabla[(g, cercano)]

    return pd.Series([regla(g, d) for g, d in zip(df["storage_group"], df["day"])],
                     index=df.index)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(SPLITS_CSV)
    tabla = ajustar(df[df["split"] == "train"])

    salida = {"regla": "moda del indice por (storage_group, day) en train",
              "n_celdas": len(tabla)}
    for split in ("val", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)
        pred = predecir(tabla, sub)
        out = pd.DataFrame({
            "file_name": sub["file_name"], "sample_id": sub["sample_id"],
            "storage_group": sub["storage_group"], "day": sub["day"], "side": sub["side"],
            # mismo rango 0-4 que las predicciones de los modelos
            "y_true": sub["ripening_index"] - 1, "y_pred": pred - 1,
        })
        out.to_csv(METRICS_DIR / f"{NOMBRE}_{split}_preds.csv", index=False)

        m = compute_metrics(out["y_true"].to_numpy(), out["y_pred"].to_numpy())
        res = {k: float(m[k]) for k in METRICAS}
        for k in ("accuracy", "qwk"):
            _, lo, hi = cluster_bootstrap_ci(out, k, n_boot=N_BOOT)
            res[f"{k}_ic95"] = [lo, hi]
        res["por_grupo"] = (out.assign(ok=out.y_true == out.y_pred)
                            .groupby("storage_group")["ok"].mean().round(4).to_dict())
        res["n"] = int(len(out))
        salida[split] = res
        print(f"{split:4s}  acc {res['accuracy']:.4f}  QWK {res['qwk']:.4f}  "
              f"F1 {res['macro_f1']:.4f}  MAE {res['mae']:.4f}  ±1 {res['off_by_one']:.4f}")

    destino = METRICS_DIR / f"{NOMBRE}.json"
    destino.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado en {destino}")


if __name__ == "__main__":
    main()
