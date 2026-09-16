"""Comparacion estadistica entre modelos.

Por que bootstrap agrupado por fruta
------------------------------------
Las 2.262 imagenes de test NO son observaciones independientes: provienen de
solo 72 frutas, unas 31 fotos por fruta. Si una palta tiene una forma o una
mancha que confunde al modelo, se equivocara en las 31 fotos a la vez. Un
intervalo de confianza que asuma independencia entre imagenes seria
artificialmente angosto y declararia significativas diferencias que no lo son.

Remuestreamos FRUTAS con reemplazo (cluster bootstrap): cada replica toma 72
frutas al azar y evalua sobre todas sus fotos. Asi el intervalo refleja la
incertidumbre real, que esta dominada por el tamano de la muestra de frutas, no
por el de imagenes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    metric: str = "qwk",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    group_col: str = "sample_id",
) -> tuple[float, float, float]:
    """IC percentil para una metrica, remuestreando frutas con reemplazo.

    `df` necesita las columnas `y_true`, `y_pred` y `group_col`.
    Devuelve (valor_puntual, limite_inferior, limite_superior).
    """
    rng = np.random.default_rng(seed)
    grupos = df[group_col].to_numpy()
    unicos = np.unique(grupos)
    # Indices precomputados por fruta: el remuestreo se vuelve una concatenacion.
    idx_por_grupo = {g: np.flatnonzero(grupos == g) for g in unicos}

    punto = compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())[metric]

    muestras = np.empty(n_boot)
    for b in range(n_boot):
        elegidos = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([idx_por_grupo[g] for g in elegidos])
        sub = df.iloc[idx]
        muestras[b] = compute_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())[metric]

    lo, hi = np.percentile(muestras, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(punto), float(lo), float(hi)


def paired_bootstrap_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str = "qwk",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    group_col: str = "sample_id",
) -> dict:
    """Bootstrap PAREADO de la diferencia A - B sobre las mismas frutas.

    Pareado es importante: ambos modelos se evaluan exactamente sobre las mismas
    imagenes, asi que la varianza compartida (frutas faciles y dificiles) se
    cancela y el test gana potencia frente a comparar dos IC independientes.

    Devuelve el punto, el IC de la diferencia y un p-valor bilateral aproximado
    por la proporcion de replicas que cruzan el cero.
    """
    a = df_a.sort_values("file_name").reset_index(drop=True)
    b = df_b.sort_values("file_name").reset_index(drop=True)
    if not a["file_name"].equals(b["file_name"]):
        raise ValueError("Los dos conjuntos de predicciones no cubren las mismas imagenes")
    if not (a["y_true"] == b["y_true"]).all():
        raise ValueError("Las etiquetas verdaderas no coinciden entre modelos")

    rng = np.random.default_rng(seed)
    grupos = a[group_col].to_numpy()
    unicos = np.unique(grupos)
    idx_por_grupo = {g: np.flatnonzero(grupos == g) for g in unicos}

    y = a["y_true"].to_numpy()
    pa, pb = a["y_pred"].to_numpy(), b["y_pred"].to_numpy()
    punto = compute_metrics(y, pa)[metric] - compute_metrics(y, pb)[metric]

    difs = np.empty(n_boot)
    for i in range(n_boot):
        elegidos = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([idx_por_grupo[g] for g in elegidos])
        difs[i] = compute_metrics(y[idx], pa[idx])[metric] - compute_metrics(y[idx], pb[idx])[metric]

    lo, hi = np.percentile(difs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # p-valor bilateral: 2x la cola mas pequena respecto de cero.
    p = 2 * min((difs <= 0).mean(), (difs >= 0).mean())
    return {
        "diff": float(punto),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "significativo": bool(lo > 0 or hi < 0),
    }


def disagreement_table(df_a: pd.DataFrame, df_b: pd.DataFrame,
                       name_a: str, name_b: str) -> pd.DataFrame:
    """Reparte el test en las cuatro combinaciones de acierto/error de ambos modelos.

    Es la tabla que hace falta para decir algo mas que "A supera a B por X puntos":
    muestra si los modelos fallan en las mismas imagenes (errores correlacionados,
    el ensamble no va a ayudar) o en imagenes distintas (son complementarios, y
    ahi la fusion tiene sentido).
    """
    a = df_a.sort_values("file_name").reset_index(drop=True)
    b = df_b.sort_values("file_name").reset_index(drop=True)
    ok_a = a["y_true"] == a["y_pred"]
    ok_b = b["y_true"] == b["y_pred"]

    filas = [
        (f"Ambos aciertan", int((ok_a & ok_b).sum())),
        (f"Solo {name_a}", int((ok_a & ~ok_b).sum())),
        (f"Solo {name_b}", int((~ok_a & ok_b).sum())),
        (f"Ambos fallan", int((~ok_a & ~ok_b).sum())),
    ]
    out = pd.DataFrame(filas, columns=["caso", "n"])
    out["%"] = (100 * out["n"] / len(a)).round(2)

    # Techo teorico de un oraculo que eligiera siempre el modelo correcto.
    out.attrs["oraculo"] = float((ok_a | ok_b).mean())
    out.attrs["ambos_fallan"] = float((~ok_a & ~ok_b).mean())
    return out
