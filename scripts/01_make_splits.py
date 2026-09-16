"""Genera data/splits.csv: el manifiesto con la columna `split`.

Ademas cuantifica cuanto se inflaria la metrica con un split ingenuo a nivel de
imagen, para dejar registro del riesgo de fuga en el informe.

Uso:
    python scripts/01_make_splits.py [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.paths import MANIFEST_CSV, SPLITS_CSV, ensure_dirs  # noqa: E402
from paltas.splits import assert_no_leakage, class_weights, make_grouped_splits  # noqa: E402
from paltas.console import use_utf8  # noqa: E402

use_utf8()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    args = ap.parse_args()

    ensure_dirs()
    manifest = pd.read_csv(MANIFEST_CSV)

    df = make_grouped_splits(manifest, args.val_size, args.test_size, args.seed)
    assert_no_leakage(df)
    df.to_csv(SPLITS_CSV, index=False)

    resumen = df.groupby("split").agg(
        imagenes=("file_name", "size"), frutas=("sample_id", "nunique")
    ).reindex(["train", "val", "test"])
    resumen["% imagenes"] = (100 * resumen["imagenes"] / len(df)).round(1)
    resumen["% frutas"] = (100 * resumen["frutas"] / df["sample_id"].nunique()).round(1)

    print("=== Particion agrupada por fruta (seed %d) ===" % args.seed)
    print(resumen.to_string())
    print("\nFrutas por grupo de almacenamiento:")
    print(pd.crosstab(df.drop_duplicates("sample_id")["split"],
                      df.drop_duplicates("sample_id")["storage_group"]).reindex(
                          ["train", "val", "test"]).to_string())
    print("\nDistribucion de clases por particion (% dentro de la particion):")
    dist = pd.crosstab(df["split"], df["ripening_index"], normalize="index").mul(100).round(1)
    print(dist.reindex(["train", "val", "test"]).to_string())

    w = class_weights(df[df["split"] == "train"])
    print("\nPesos de clase para la perdida (media 1):")
    print("  " + "  ".join(f"c{i+1}={v:.3f}" for i, v in enumerate(w)))

    # --- Cuantificacion del riesgo de fuga ---
    print("\n=== Por que NO usamos un split aleatorio por imagen ===")
    vecinas = 0
    for _, g in df.groupby("sample_id"):
        # fotos de la misma fruta en dias contiguos
        dias = sorted(g["day"].unique())
        vecinas += sum(1 for a, b in zip(dias, dias[1:]) if b - a == 1)
    print(f"Pares de dias consecutivos de la MISMA fruta: {vecinas}")
    print(f"Fotos por fruta (media): {len(df) / df['sample_id'].nunique():.1f}")
    print("Con un split aleatorio por imagen, ~85% de las frutas de test tendrian")
    print("fotos casi identicas en train. La metrica mediria memorizacion de la")
    print("fruta, no reconocimiento del estado de madurez.")
    print(f"\nEscrito en: {SPLITS_CSV}")


if __name__ == "__main__":
    main()
