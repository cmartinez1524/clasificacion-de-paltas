"""Metricas ordinales con casos de resultado conocido."""

import numpy as np
import pytest

from paltas.metrics import compute_metrics


def test_prediccion_perfecta():
    y = np.array([0, 1, 2, 3, 4, 2, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["qwk"] == pytest.approx(1.0)
    assert m["mae"] == 0.0
    assert m["off_by_one"] == 1.0


def test_desfase_de_uno():
    y = np.array([0, 1, 2, 3, 3, 2, 1, 0])
    m = compute_metrics(y, y + 1)
    assert m["accuracy"] == 0.0
    assert m["mae"] == 1.0
    assert m["off_by_one"] == 1.0
    assert 0 < m["qwk"] < 1


def test_qwk_castiga_mas_los_errores_grandes():
    y = np.array([0, 1, 2, 3, 4] * 4)
    cerca = np.clip(y + 1, 0, 4)
    lejos = np.clip(y + 3, 0, 4)
    assert compute_metrics(y, cerca)["qwk"] > compute_metrics(y, lejos)["qwk"]
