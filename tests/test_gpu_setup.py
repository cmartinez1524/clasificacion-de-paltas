"""Comprueba el lanzador local y las metricas necesarias para la tarea E."""
import importlib.util
from pathlib import Path
import subprocess
import os

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != "nt", reason="Lanzadores PowerShell para Windows")
def test_windows_launchers_parse():
    for relative in ("scripts/setup_gpu.ps1", "scripts/run_asimetricas.ps1",
                     "app/instalar.ps1", "app/iniciar.ps1"):
        path = str(ROOT / relative).replace("'", "''")
        subprocess.run([
            "powershell", "-NoProfile", "-Command",
            f"$tokens=$null; $errors=$null; [void][System.Management.Automation.Language.Parser]::ParseFile('{path}',[ref]$tokens,[ref]$errors); if ($errors) {{ $errors; exit 1 }}",
        ], check=True, capture_output=True)


def test_seed_summary_includes_asymmetric_metrics():
    path = ROOT / "scripts" / "11_resumen_semillas.py"
    spec = importlib.util.spec_from_file_location("resumen_gpu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("resnet50_asimetrico", "vit_small_asimetrico"):
        assert name in mod.MODELOS and name in mod.COLORES
    result = mod.metricas(pd.DataFrame({"y_true": [0, 2], "y_pred": [1, 1]}))
    assert np.isclose(result["asymmetric_mae"], 1.5)
    assert result["underestimation_rate"] == result["overestimation_rate"] == 0.5
