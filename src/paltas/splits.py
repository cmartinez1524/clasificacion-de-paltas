"""Particionado train/val/test con agrupamiento por fruta.

Por que agrupar
---------------
El dataset es longitudinal: cada palta fue fotografiada a diario durante hasta
26 dias, dos veces por dia (lados a y b). Dos fotos consecutivas de la misma
fruta son casi identicas -- misma piel, mismas manchas, misma forma, mismo
pedunculo. Un split aleatorio a nivel de imagen pondria el dia 5 de la fruta
#173 en train y el dia 6 de esa misma fruta en test, y el modelo podria
"reconocer la fruta" en vez de aprender el estado de madurez. La accuracy
resultante seria optimista y no transferiria a una palta nueva.

Por eso particionamos a nivel de `sample_id` (fruta fisica): las 478 frutas se
reparten 70/15/15 y todas las fotos de una fruta caen en la misma particion.

Estratificacion
---------------
Estratificamos por grupo de almacenamiento (T10 / T20 / Tamb) porque determina
la velocidad de maduracion y, por lo tanto, cuantas fotos y de que clases aporta
cada fruta. Sin estratificar, un test dominado por T10 (refrigerado, maduracion
lenta) tendria un perfil de clases distinto al de train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "val", "test")


def make_grouped_splits(
    manifest: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Asigna cada imagen a train/val/test agrupando por fruta.

    Args:
        manifest: DataFrame con al menos `sample_id` y `storage_group`.
        val_size: fraccion de FRUTAS destinada a validacion.
        test_size: fraccion de FRUTAS destinada a test.
        seed: semilla del reparto.

    Returns:
        Copia del manifiesto con una columna `split` adicional.
    """
    # Una fila por fruta: es la unidad que repartimos.
    frutas = (
        manifest.groupby("sample_id", as_index=False)["storage_group"].first()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    trainval, test = train_test_split(
        frutas,
        test_size=test_size,
        stratify=frutas["storage_group"],
        random_state=seed,
    )
    # val_size esta expresado sobre el total, no sobre trainval.
    val_rel = val_size / (1.0 - test_size)
    train, val = train_test_split(
        trainval,
        test_size=val_rel,
        stratify=trainval["storage_group"],
        random_state=seed,
    )

    asignacion = {}
    for nombre, parte in zip(SPLIT_NAMES, (train, val, test)):
        asignacion.update({sid: nombre for sid in parte["sample_id"]})

    out = manifest.copy()
    out["split"] = out["sample_id"].map(asignacion)
    assert out["split"].notna().all(), "Quedaron imagenes sin particion asignada"
    return out


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Falla si alguna fruta aparece en mas de una particion."""
    por_fruta = df.groupby("sample_id")["split"].nunique()
    culpables = por_fruta[por_fruta > 1]
    if len(culpables):
        raise AssertionError(
            f"Fuga de datos: {len(culpables)} frutas en multiples particiones "
            f"(ej. {culpables.index[:5].tolist()})"
        )


def class_weights(df_train: pd.DataFrame, num_classes: int = 5) -> np.ndarray:
    """Pesos inversos a la frecuencia, normalizados a media 1.

    El desbalance es moderado (clase 1: 3.568 imagenes vs clase 2: 2.228, un
    factor 1,6x), asi que no hace falta remuestrear; basta con ponderar la
    perdida.
    """
    counts = df_train["ripening_index"].value_counts().reindex(
        range(1, num_classes + 1), fill_value=0
    ).to_numpy(dtype=np.float64)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return (w / w.mean()).astype(np.float32)
