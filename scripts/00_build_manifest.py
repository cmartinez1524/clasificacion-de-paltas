"""Construye data/manifest.csv cruzando el Excel oficial con las imagenes en disco.

El Excel trae 14.722 filas pero en disco hay 14.710 archivos: 12 registros (6
pares a/b) no tienen imagen asociada. Hacemos un inner join y dejamos constancia
del descarte en el reporte.

Ademas validamos una redundancia util del dataset: el ultimo campo del nombre de
archivo codifica el indice de madurez (T10_d01_002_a_1 -> clase 1). Si esa
etiqueta implicita contradijera al Excel tendriamos un problema de integridad,
asi que lo chequeamos explicitamente en vez de asumirlo.

Uso:
    python scripts/00_build_manifest.py
"""

from __future__ import annotations

import re
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from paltas.paths import EXCEL_PATH, IMAGES_DIR, MANIFEST_CSV, ensure_dirs  # noqa: E402
from paltas.console import use_utf8  # noqa: E402

use_utf8()

# T10_d01_002_a_1  ->  grupo, dia, muestra, lado, indice
FILENAME_RE = re.compile(r"^(T10|T20|Tam)_d(\d{2})_(\d{3})_([ab])_([1-5])$")

COLUMN_MAP = {
    "File Name": "file_name",
    "Time Stamp": "timestamp",
    "Storage Group": "storage_group",
    "Sample": "sample_id",
    "Day of Experiment": "day",
    "Ripening Index Classification": "ripening_index",
}


def main() -> None:
    ensure_dirs()

    df = pd.read_excel(EXCEL_PATH).rename(columns=COLUMN_MAP)
    n_excel = len(df)

    on_disk = {p.stem: p.name for p in IMAGES_DIR.glob("*.jpg")}
    n_disk = len(on_disk)

    df["image_file"] = df["file_name"].map(on_disk)
    missing = df[df["image_file"].isna()]
    df = df.dropna(subset=["image_file"]).reset_index(drop=True)

    # --- Validacion de integridad: nombre de archivo vs Excel ---
    parsed = df["file_name"].str.extract(FILENAME_RE)
    parsed.columns = ["grp_f", "day_f", "sample_f", "side", "idx_f"]
    if parsed.isna().any().any():
        bad = df.loc[parsed.isna().any(axis=1), "file_name"].tolist()
        raise ValueError(f"Nombres que no siguen el patron esperado: {bad[:10]}")

    df["side"] = parsed["side"]
    mismatch_idx = (parsed["idx_f"].astype(int) != df["ripening_index"]).sum()
    mismatch_grp = (parsed["grp_f"] != df["storage_group"]).sum()
    mismatch_day = (parsed["day_f"].astype(int) != df["day"]).sum()
    mismatch_smp = (parsed["sample_f"].astype(int) != df["sample_id"]).sum()

    # Una muestra (fruta fisica) debe pertenecer a un unico grupo de almacenamiento.
    per_sample_groups = df.groupby("sample_id")["storage_group"].nunique()
    assert per_sample_groups.max() == 1, "Hay muestras en mas de un grupo de almacenamiento"

    cols = [
        "file_name", "image_file", "timestamp", "storage_group",
        "sample_id", "day", "side", "ripening_index",
    ]
    df = df[cols].sort_values(["storage_group", "sample_id", "day", "side"]).reset_index(drop=True)
    df.to_csv(MANIFEST_CSV, index=False)

    print(f"Filas en Excel          : {n_excel}")
    print(f"Imagenes en disco       : {n_disk}")
    print(f"Descartadas (sin imagen): {len(missing)}")
    if len(missing):
        print("  -> " + ", ".join(missing["file_name"].tolist()))
    print(f"Filas en el manifiesto  : {len(df)}")
    print()
    print("Integridad nombre-archivo vs Excel (0 = consistente):")
    print(f"  indice de madurez : {mismatch_idx}")
    print(f"  grupo             : {mismatch_grp}")
    print(f"  dia               : {mismatch_day}")
    print(f"  muestra           : {mismatch_smp}")
    print()
    print(f"Frutas (muestras) unicas: {df['sample_id'].nunique()}")
    print(df.groupby("storage_group").agg(
        imagenes=("file_name", "size"), frutas=("sample_id", "nunique")
    ).to_string())
    print()
    print("Distribucion de clases:")
    print(df["ripening_index"].value_counts().sort_index().to_string())
    print(f"\nManifiesto escrito en: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
