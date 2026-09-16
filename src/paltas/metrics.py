"""Metricas de evaluacion para un objetivo ordinal.

El indice de madurez no es una variable categorica: 1 < 2 < 3 < 4 < 5. Confundir
una palta de clase 1 con una de clase 2 es un error casi irrelevante en la
practica (un dia de diferencia); confundir una clase 1 con una clase 5 significa
mandar a la venta fruta podrida. La accuracy plana trata ambos errores igual, asi
que la acompanamos de metricas sensibles a la distancia:

- QWK (kappa de Cohen con pesos cuadraticos): penaliza el error proporcionalmente
  al cuadrado de la distancia entre clases y corrige por acuerdo azaroso. Es la
  metrica principal del proyecto.
- MAE en el indice: error medio en "escalones" de madurez, directamente
  interpretable.
- Accuracy off-by-one: fraccion de predicciones que caen en la clase correcta o
  en una adyacente. Aproxima la utilidad operativa real.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

METRIC_NAMES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced acc.",
    "macro_f1": "Macro-F1",
    "qwk": "QWK",
    "mae": "MAE (escalones)",
    "off_by_one": "Acc. ±1 clase",
}
#: Metrica usada para seleccionar el mejor checkpoint y para ordenar resultados.
PRIMARY_METRIC = "qwk"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> dict:
    """Calcula el paquete completo de metricas. Espera etiquetas en 0..C-1."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = list(range(num_classes))

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "off_by_one": float((np.abs(y_true - y_pred) <= 1).mean()),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels,
                                 zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def format_metrics(m: dict) -> str:
    """Una linea legible para los logs de entrenamiento."""
    return "  ".join(
        f"{k}={m[k]:.4f}" for k in ("accuracy", "macro_f1", "qwk", "mae", "off_by_one")
    )


def scalar_metrics(m: dict) -> dict:
    """Solo las metricas escalares, para tablas comparativas."""
    return {k: m[k] for k in METRIC_NAMES if k in m}
