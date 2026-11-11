"""La particion por fruta es la decision que evita la fuga de datos: se verifica."""

import pandas as pd
import pytest

from paltas.paths import SPLITS_CSV
from paltas.splits import assert_no_leakage

pytestmark = pytest.mark.skipif(not SPLITS_CSV.exists(),
                                reason="falta data/splits.csv (scripts/01_make_splits.py)")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(SPLITS_CSV)


def test_tamano_del_dataset(df):
    assert len(df) == 14710
    assert df["sample_id"].nunique() == 478


def test_ninguna_fruta_en_dos_particiones(df):
    assert_no_leakage(df)
    assert (df.groupby("sample_id")["split"].nunique() == 1).all()


def test_proporciones_por_fruta(df):
    frutas = df.drop_duplicates("sample_id")["split"].value_counts(normalize=True)
    assert frutas["train"] == pytest.approx(0.70, abs=0.02)
    assert frutas["val"] == pytest.approx(0.15, abs=0.02)
    assert frutas["test"] == pytest.approx(0.15, abs=0.02)


def test_detecta_fuga():
    malo = pd.DataFrame({"sample_id": [1, 1], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        assert_no_leakage(malo)
