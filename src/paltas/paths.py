"""Rutas canonicas del proyecto.

Todo el codigo resuelve rutas a partir de la raiz del repositorio, de modo que
los scripts funcionan sin importar desde que directorio se invoquen. El dataset
crudo puede vivir fuera del repo: se sobreescribe con la variable de entorno
PALTAS_RAW_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/paltas/paths.py -> src/paltas -> src -> raiz
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CONFIG_DIR = ROOT / "configs"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
CHECKPOINT_DIR = ROOT / "checkpoints"

MANIFEST_CSV = DATA_DIR / "manifest.csv"
SPLITS_CSV = DATA_DIR / "splits.csv"

_DEFAULT_RAW = ROOT / "Hass Avocado Ripening Photographic Dataset"

#: Carpeta que contiene el .xlsx y la carpeta de imagenes descargados de Mendeley.
RAW_DIR = Path(os.environ.get("PALTAS_RAW_DIR", _DEFAULT_RAW))
IMAGES_DIR = RAW_DIR / "Avocado Ripening Dataset"
EXCEL_PATH = RAW_DIR / "Avocado Ripening Dataset.xlsx"

#: Nombres legibles del indice de madurez (1-5) segun el paper original.
CLASS_NAMES = {
    1: "Verde / no maduro",
    2: "Iniciando maduracion",
    3: "Maduro (etapa 1)",
    4: "Maduro (etapa 2)",
    5: "Sobremaduro",
}
NUM_CLASSES = 5


def ensure_dirs() -> None:
    """Crea los directorios de salida si no existen."""
    for d in (DATA_DIR, CACHE_DIR, FIGURES_DIR, METRICS_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)
