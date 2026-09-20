"""Genera las dos presentaciones en Beamer (LaTeX) desde las metricas reales.

Produce carpetas autocontenidas: el .tex y una subcarpeta figuras/ con solo las
imagenes que ese deck usa. Asi se puede arrastrar la carpeta completa a Overleaf
sin tocar nada.

Uso:
    python scripts/07_deck_latex.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.beamer import (  # noqa: E402
    columnas,
    documento,
    figura,
    frame,
    items,
    kpis,
    nombre_seguro,
    num,
    pct,
    portada,
    seccion,
    signo,
    tabla,
)
from paltas.console import use_utf8  # noqa: E402
from paltas.paths import FIGURES_DIR, METRICS_DIR, ROOT  # noqa: E402

use_utf8()

AUTOR = "Cristóbal Martínez"
FECHA_AVANCE = "Avance, 28 de septiembre de 2026"
FECHA_FINAL = "Entrega final, 23 de noviembre de 2026"
REPO = r"\texttt{github.com/cmartinez1524/clasificacion-de-paltas}"

ETIQUETA = {
    "resnet50": "ResNet-50",
    "vit_small": "ViT-S/16",
    "resnet50_scratch": "ResNet-50 (scratch)",
    "vit_small_scratch": "ViT-S/16 (scratch)",
    "hybrid_fusion": "Híbrido: fusión tardía",
    "hybrid_vit_r26": "ViT-Hybrid R26+S/32",
}
GRUPO = {"T10": r"T10 (10\,\textdegree C)", "T20": r"T20 (20\,\textdegree C)",
         "Tam": "Tamb"}


def cargar(nombre: str) -> dict | None:
    p = METRICS_DIR / f"{nombre}_test.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cargar_comp(tag: str) -> dict | None:
    p = METRICS_DIR / f"{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def copiar_figuras(cuerpo: str, destino: Path) -> int:
    """Copia a `destino` exactamente las figuras que el .tex referencia.

    La lista se extrae del propio LaTeX generado en vez de mantenerse a mano:
    asi no se copian figuras de mas ni se olvida ninguna cuando cambian los
    slides. El guion bajo rompe \\includegraphics (ver
    paltas.beamer.nombre_seguro), por eso la copia lleva guion medio.
    """
    referidas = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", cuerpo))
    destino.mkdir(parents=True, exist_ok=True)
    for f in destino.glob("*.png"):
        f.unlink()
    for seguro in sorted(referidas):
        # deshacemos el renombrado para encontrar el original
        candidatos = [p for p in FIGURES_DIR.glob("*.png")
                      if nombre_seguro(p.stem) == seguro]
        if candidatos:
            shutil.copy2(candidatos[0], destino / f"{seguro}.png")
        else:
            print(f"  [aviso] no encuentro la figura para {seguro}")
    return len(referidas)


# --------------------------------------------------------------------------- #
# Bloques de contenido compartidos entre los dos decks
# --------------------------------------------------------------------------- #
TABLA_INDICE = tabla(
    ["Índ.", "Estado", "Decisión"],
    [["1", "Verde", "Dejar madurar"],
     ["2", "Iniciando", "Lista en 2--3 días"],
     ["3", "Maduro I", "Venta inmediata"],
     ["4", "Maduro II", "Último día"],
     ["5", "Sobremaduro", "Fuera de punto"]],
    spec="cll", tam=r"\tiny")


def frame_problema(fecha_nota: str) -> str:
    izq = items([
        ("Chile es uno de los mayores exportadores de palta Hass del mundo.", True),
        "La clasificación por madurez se hace hoy a ojo y al tacto, operario "
        "por operario, sin criterio homogéneo ni trazabilidad.",
        "Fruta verde en góndola: el consumidor la descarta. Fruta pasada en "
        "tránsito: pérdida total. Los dos errores cuestan.",
        ("La tarea es \\emph{ordinal}, no categórica: $1<2<3<4<5$.", True),
        "Confundir un 1 con un 2 es un error de un día. Confundir un 1 con un 5 "
        "es mandar fruta podrida a la venta.",
    ], tam=r"\scriptsize", sep="0.3em")
    return frame(
        "El problema", "Por qué importa acá y no solo en el paper",
        columnas(izq, TABLA_INDICE, "0.58", "0.38")
        + "\n" + r"\cierre{La métrica tiene que castigar más los errores lejanos. "
                 r"La accuracy plana no lo hace: por eso usamos QWK.}",
        nota=fecha_nota)


def frame_redes(incluir_hibridos: bool) -> str:
    """El frame que explica QUE redes se usaron y como funciona cada una.

    Con los hibridos son cuatro arquitecturas en un slide, asi que el texto va
    un punto mas chico y sin las lineas de \\texttt{timm}: si no, desborda.
    """
    tam = r"\tiny" if incluir_hibridos else r"\scriptsize"
    enc = r"\footnotesize" if incluir_hibridos else r"\small"

    izq = (rf"{{{enc}\textbf{{\color{{azul}}ResNet-50}} \;\textit{{(convolucional, 2015)}}}}"
           + "\n" + items([
               "Filtros pequeños que se deslizan por toda la imagen, apilados en "
               "50 capas con conexiones residuales.",
               ("Trae un supuesto incorporado: localidad y equivarianza a la "
                "traslación.", True),
               "Una mancha café significa lo mismo esté donde esté sobre la fruta. "
               "Eso \\emph{no hay que aprenderlo}: está en la arquitectura.",
           ], tam=tam, sep="0.18em"))
    der = (rf"{{{enc}\textbf{{\color{{rojo}}ViT-S/16}} \;\textit{{(transformer, 2020)}}}}"
           + "\n" + items([
               "Corta la imagen de $224\\times224$ en 196 parches de $16\\times16$ "
               "y los procesa con auto-atención.",
               "Cada parche mira a todos los demás desde la primera capa; un token "
               "\\texttt{CLS} resume la imagen.",
               ("No asume nada sobre estructura espacial.", True),
               "Tiene que aprender de los datos lo que la convolución trae gratis.",
           ], tam=tam, sep="0.18em"))

    if not incluir_hibridos:
        izq += ("\n\\vspace{0.3em}\n"
                + r"{\scriptsize\color{grisSuave}\texttt{timm: resnet50.a1\_in1k}}")
        der += ("\n\\vspace{0.3em}\n"
                + r"{\scriptsize\color{grisSuave}"
                  r"\texttt{timm: deit\_small\_patch16\_224.fb\_in1k}}")

    cuerpo = columnas(izq, der, "0.47", "0.47")
    if incluir_hibridos:
        cuerpo += "\n\\vspace{0.35em}\n" + r"{\color{verdecla}\rule{\textwidth}{0.8pt}}" + "\n"
        cuerpo += "\\vspace{0.2em}\n" + columnas(
            rf"{{{enc}\textbf{{\color{{morado}}Híbrido 1: fusión tardía}}}}" + "\n"
            + items([
                "Los dos \\emph{backbones} corren en paralelo sobre la misma imagen; "
                "sus salidas se proyectan a 512 y se concatenan.",
                "Es el único modelo escrito a mano (\\texttt{src/paltas/models.py}).",
            ], tam=tam, sep="0.12em"),
            rf"{{{enc}\textbf{{\color{{morado}}Híbrido 2: ViT-Hybrid R26+S/32}}}}" + "\n"
            + items([
                "El híbrido del paper original de ViT: un \\emph{stem} convolucional "
                "alimenta al transformer con su mapa de features.",
                "Híbrido \\emph{interno}, no una unión de dos redes.",
            ], tam=tam, sep="0.12em"),
            "0.47", "0.47")
        cierre = (r"\cierre{Los dos híbridos NO son equiparables a los baselines "
                  r"(2$\times$ cómputo uno, ImageNet-21k el otro). Van como referencia.}")
    else:
        cierre = (r"\cierre{Ninguna se escribió desde cero: usamos \texttt{timm}, "
                  r"la biblioteca estándar de modelos de visión pre-entrenados.}")
    return frame("Las redes que comparamos", "Qué es cada una y qué asume",
                 cuerpo + "\n" + cierre)


def frame_comparabilidad(r: dict, v: dict) -> str:
    dp = abs(r["params_M"] - v["params_M"]) / max(r["params_M"], v["params_M"]) * 100
    dg = abs(r["gflops"] - v["gflops"]) / max(r["gflops"], v["gflops"]) * 100
    t = tabla(
        ["Eje igualado", "ResNet-50", "ViT-S/16 (DeiT-S)", "Diferencia"],
        [["Parámetros", f"{num(r['params_M'], 2)}\\,M", f"{num(v['params_M'], 2)}\\,M",
          f"{num(dp, 1)}\\,\\%"],
         ["GFLOPs @ 224\\,px", num(r["gflops"], 2), num(v["gflops"], 2),
          f"{num(dg, 1)}\\,\\%"],
         ["Pre-entrenamiento", "ImageNet-1k", "ImageNet-1k", "ninguna"]],
        spec="lccc", tam=r"\scriptsize")
    cuerpo = t + "\n\\vspace{0.5em}\n" + items([
        ("El tercer eje es el que casi nadie iguala, y el que más distorsiona.", True),
        "Los pesos ViT-S más usados de \\texttt{timm} "
        "(\\texttt{vit\\_small\\_patch16\\_224.augreg\\_in21k\\_ft\\_in1k}) vienen de "
        "\\textbf{ImageNet-21k}: catorce veces más datos que los de la ResNet. "
        "Con esos pesos la comparación mediría el tamaño del corpus, no la arquitectura.",
        "Por eso usamos \\textbf{DeiT-S}: exactamente la arquitectura ViT-S/16, pero "
        "entrenada solo en ImageNet-1k, frente a \\texttt{resnet50.a1\\_in1k}.",
    ], tam=r"\scriptsize", sep="0.25em")
    return frame(
        "Diseño de la comparación", "Tres ejes igualados a la vez, no solo los parámetros",
        cuerpo + "\n" + r"\cierre{Comparar \guillemotleft{}una ResNet\guillemotright{} "
                        r"contra \guillemotleft{}un ViT\guillemotright{} sin igualar el "
                        r"presupuesto no dice nada: cualquier diferencia se explicaría "
                        r"por el tamaño.}")


FRAME_RECETA = frame(
    "La receta de entrenamiento", "Idéntica para ambas redes, salvo lo que tiene que cambiar",
    columnas(
        r"{\small\textbf{Igual para las dos}}" + "\n" + items([
            "AdamW, schedule coseno con warmup",
            "AMP fp16 (casi duplica la velocidad en la 4070)",
            "Label smoothing 0{,}05",
            "Pesos de clase inversos a la frecuencia",
            "Recorte de gradiente a 1{,}0",
            "Mismas aumentaciones y mismos datos",
            "Checkpoint elegido por \\textbf{QWK de validación}, no por accuracy",
            "Early stopping a las 5 épocas sin mejora",
        ], tam=r"\tiny", sep="0.15em"),
        r"{\small\textbf{Distinto en el ViT, y por qué}}" + "\n" + items([
            ("Learning rate $3\\times$ menor ($1$e-4 vs $3$e-4).", True),
            "Los transformers divergen con los LR que una ResNet con BatchNorm "
            "tolera sin problema.",
            ("Warmup más largo (2 épocas vs 1).", True),
            "Sin warmup, la atención colapsa en las primeras épocas.",
            ("\\texttt{drop\\_path} $=0{,}1$.", True),
            "\\emph{Stochastic depth}, el regularizador estándar de los ViT.",
        ], tam=r"\tiny", sep="0.15em"),
        "0.47", "0.47")
    + "\n\\vspace{0.3em}\n"
    + r"{\scriptsize\textbf{Aumentaciones de color deliberadamente suaves} "
      r"(\texttt{hue=0.02} en vez de \texttt{0.1}): acá el color \emph{es} la etiqueta, "
      r"y un \texttt{ColorJitter} estándar convertiría una clase 2 en una clase 4 sin "
      r"cambiarle la etiqueta. Las geométricas sí son agresivas: recorte 0{,}65--1{,}0, "
      r"volteo, rotación $\pm 20^{\circ}$.}"
    + "\n" + r"\cierre{Igualar el learning rate \guillemotleft{}por justicia"
             r"\guillemotright{} no sería más justo: "
             r"sería solo peor para el ViT.}")


def frame_baseline(nombre: str, d: dict, color: str) -> str:
    t = d["test"]
    k = kpis([("QWK (principal)", num(t["qwk"], 3), color),
              ("Accuracy", pct(t["accuracy"]), color),
              ("Acc. $\\pm$1 clase", pct(t["off_by_one"]), color),
              ("MAE", num(t["mae"], 3), color),
              ("Macro-F1", num(t["macro_f1"], 3), color)])
    filas = [[GRUPO.get(g, g), str(int(x["n"])), pct(x["accuracy"]), num(x["mae"], 3)]
             for g, x in d.get("test_por_grupo", {}).items()]
    tg = tabla(["Grupo", "n", "Accuracy", "MAE"], filas,
               spec="lccc", tam=r"\tiny")
    der = items([
        f"Los errores se concentran en clases adyacentes: {pct(t['off_by_one'])} "
        "cae en la clase correcta o en una vecina.",
        f"MAE de {num(t['mae'], 2)} escalones sobre una escala de 5.",
        "Desempeño homogéneo entre grupos: el modelo \\textbf{no} está explotando "
        "el sesgo \\guillemotleft{}verde $\\Rightarrow$ refrigerada\\guillemotright{}.",
    ], tam=r"\tiny", sep="0.25em")
    sub = (f"{num(d['params_M'], 2)}\\,M params, {num(d['gflops'], 2)} GFLOPs, "
           f"{d['train_minutes']:.0f} min, mejor época {d['best_epoch']}")
    return frame(
        f"Baseline {nombre}", sub,
        k + "\n\\vspace{0.4em}\n" + columnas(tg, der, "0.46", "0.50")
        + "\n" + r"\cierre{Test: 2.262 imágenes de 72 frutas que ningún modelo vio.}")


def lineas_significancia(comp: dict, solo=None) -> list:
    out = []
    for c in comp.get("comparaciones", []):
        if solo and c["modelo_a"] not in solo:
            continue
        d = c["metricas"]["qwk"]
        ver = r"\sig" if d["significativo"] else r"\nosig"
        out.append((f"{ETIQUETA.get(c['modelo_a'], c['modelo_a'])}: "
                    f"$\\Delta$QWK $= {signo(d['diff'])}$, "
                    f"IC 95\\,\\% $[{signo(d['ci_low'])}, {signo(d['ci_high'])}]$, "
                    f"$p={num(d['p_value'], 3)}$ $\\rightarrow$ {ver}", True))
    return out


FRAME_RIESGOS = frame(
    "Riesgos identificados y mitigación", "Concretos de este dataset, no genéricos",
    tabla(["Riesgo", "Por qué es real acá", "Mitigación"],
          [["Fuga longitudinal", "6.871 pares de días consecutivos de la misma fruta",
            "Split por fruta + \\texttt{assert} automático"],
           ["Domain gap lab.\\ $\\rightarrow$ terreno",
            "Fondo blanco, fruta centrada, luz difusa controlada",
            "Declarado en la data card; recolectar fotos de celular"],
           ["Color $=$ señal, no ruido",
            "Un \\texttt{ColorJitter} estándar vuelve una clase 2 en una clase 4",
            "Jitter suave: \\texttt{hue} 0{,}02 en vez de 0{,}1"],
           ["Sesgo T10 $\\Rightarrow$ clase 1",
            "T10 aporta 60\\,\\% de las imágenes con 40\\,\\% de las frutas",
            "Pesos de clase + métricas desglosadas por grupo"],
           ["Etiquetas subjetivas", "Sin acuerdo inter-evaluador: techo humano desconocido",
            "QWK y MAE en vez de accuracy; no perseguir 100\\,\\%"],
           ["Ventaja oculta de pre-entren.",
            "Los pesos ViT habituales vienen de ImageNet-21k",
            "DeiT-S: ambos modelos solo ImageNet-1k"]],
          spec=r">{\raggedright\arraybackslash}p{0.21\textwidth}"
               r">{\raggedright\arraybackslash}p{0.37\textwidth}"
               r">{\raggedright\arraybackslash}p{0.34\textwidth}",
          tam=r"\scriptsize")
    + "\n" + r"\cierre{El domain gap es el riesgo que decide si esto sirve en un packing "
             r"o solo en un paper.}")


# --------------------------------------------------------------------------- #
# Deck de avance
# --------------------------------------------------------------------------- #
def construir_avance(r: dict, v: dict, comp: dict | None) -> str:
    cuerpo = portada(
        "Clasificación del estado de\\\\madurez de paltas Hass",
        "ResNet-50 frente a ViT-S/16 con presupuesto equiparado",
        AUTOR, FECHA_AVANCE,
        "14.710 fotografías, 478 frutas, índice de madurez de 5 niveles")

    cuerpo += frame_problema("")

    cuerpo += frame(
        "Los datos", "Hass Avocado Ripening Photographic Dataset, Mendeley, CC BY 4.0",
        kpis([("Imágenes", "14.710", "verdeosc"), ("Frutas", "478", "azul"),
              ("Clases", "5", "morado"), ("Grupos", "3", "rojo")])
        + "\n\\vspace{0.4em}\n" + figura("01_distribucion_clases", r"0.84\textwidth", "3.2cm")
        + "\n" + r"\cierre{Xavier, Rodrigues \& Silva (2024), DOI 10.17632/3xd9n945v8.1. "
                 r"Etiquetado visual experto, sin acuerdo inter-evaluador reportado.}")

    cuerpo += frame(
        "El riesgo que define el proyecto", "Fuga de datos longitudinal",
        columnas(
            items([
                ("La misma palta fue fotografiada a diario durante hasta 26 días.", True),
                "Dos fotos consecutivas de la fruta \\#173 son casi idénticas: "
                "misma piel, mismas manchas, misma forma.",
                ("Un split aleatorio por imagen pondría el día 5 en train y el "
                 "día 6 en test.", True),
                "El modelo reconocería la fruta, no el estado de madurez. "
                "La métrica sería alta y completamente falsa.",
                "Hay 6.871 pares de días consecutivos de la misma fruta.",
                ("Partimos las 478 frutas, no las 14.710 imágenes.", True),
            ], tam=r"\scriptsize", sep="0.25em"),
            figura("04_particiones", r"\textwidth", "3.1cm"),
            "0.48", "0.48")
        + "\n" + r"\cierre{Estratificado por grupo de almacenamiento, con un "
                 r"\texttt{assert} en el código que falla si una fruta aparece "
                 r"en dos particiones.}")

    cuerpo += frame_redes(incluir_hibridos=False)
    cuerpo += frame_comparabilidad(r, v)
    cuerpo += FRAME_RECETA

    cuerpo += seccion("01", "Resultados preliminares")
    cuerpo += frame_baseline("ResNet-50", r, "azul")
    cuerpo += frame_baseline("ViT-S/16", v, "rojo")

    filas = []
    for nom, d in (("ResNet-50", r), ("ViT-S/16", v)):
        t = d["test"]
        filas.append([nom, f"{num(d['params_M'], 1)}\\,M", num(d["gflops"], 1),
                      num(t["qwk"], 3), pct(t["accuracy"]), num(t["macro_f1"], 3),
                      num(t["mae"], 3), pct(t["off_by_one"])])
    mejor = 0 if r["test"]["qwk"] >= v["test"]["qwk"] else 1
    tcomp = tabla(["Modelo", "Params", "GFLOPs", "QWK", "Accuracy", "Macro-F1",
                   "MAE", "Acc. $\\pm$1"], filas, spec="lccccccc", tam=r"\scriptsize",
                  destacar=mejor)

    puntos = lineas_significancia(comp) if comp else []
    if comp:
        for _, dd in comp.get("desacuerdo", {}).items():
            puntos.append(
                f"Un oráculo que eligiera siempre el modelo correcto llegaría a "
                f"{pct(dd['oraculo_accuracy'])} de accuracy, frente a "
                f"{pct(max(r['test']['accuracy'], v['test']['accuracy']))} del mejor "
                f"modelo individual: hay complementariedad que explotar.")
            puntos.append(f"Ambos modelos fallan a la vez en solo "
                          f"{pct(dd['ambos_fallan'])} de las imágenes.")
            break
    puntos.append("Las 2.262 imágenes de test vienen de solo 72 frutas: asumir "
                  "independencia entre imágenes daría intervalos artificialmente angostos, "
                  "por eso el bootstrap remuestrea \\emph{frutas}.")

    cuerpo += frame(
        "ResNet-50 frente a ViT-S/16",
        "Intervalos por bootstrap agrupado por fruta, no por imagen",
        tcomp + "\n\\vspace{0.4em}\n"
        + columnas(items(puntos, tam=r"\tiny", sep="0.25em"),
                   figura("10_matrices_confusion_avance", r"\textwidth", "3.0cm"),
                   "0.50", "0.46")
        + "\n" + r"\cierre{La ResNet acierta la clase exacta más seguido; el ViT se "
                 r"equivoca por menos distancia. Sobre QWK, que pondera el error por "
                 r"distancia, empatan.}")

    cuerpo += frame(
        "Curvas de entrenamiento", "Misma receta, convergencia distinta",
        figura("11_curvas_entrenamiento_avance", r"\textwidth", "4.3cm")
        + "\n\\vspace{0.3em}\n" + items([
            f"ResNet-50 alcanza su mejor QWK de validación en la época "
            f"{r['best_epoch']}; ViT-S/16 en la {v['best_epoch']}.",
            f"Tiempo de entrenamiento: {r['train_minutes']:.0f} min frente a "
            f"{v['train_minutes']:.0f} min en una RTX 4070 Laptop de 8\\,GB. "
            "El ViT converge al doble de velocidad.",
        ], tam=r"\scriptsize", sep="0.2em"))

    cuerpo += FRAME_RIESGOS

    cuerpo += frame(
        "Qué viene para el entregable final",
        "Lo que falta para noviembre",
        items([
            ("Ablación sin pre-entrenamiento.", True),
            "Los mismos dos modelos entrenados desde cero. Es la evidencia empírica "
            "propia de por qué ResNet y ViT difieren: el transformer no tiene sesgo "
            "inductivo de localidad y tiene que aprenderlo de los datos.",
            ("Dos modelos híbridos.", True),
            "Fusión tardía de los dos backbones ya afinados, y el ViT-Hybrid R26+S/32 "
            "del paper original de ViT.",
            ("Interpretabilidad comparada.", True),
            "Grad-CAM sobre la ResNet y attention rollout sobre el ViT, en los casos "
            "donde los dos modelos discrepan.",
            ("Demo funcional en Gradio y decisión explícita de velocidad frente a "
             "precisión.", True),
        ], tam=r"\small")
        + "\n" + rf"\cierre{{Repositorio: {REPO}}}")

    return cuerpo


# --------------------------------------------------------------------------- #
# Deck final
# --------------------------------------------------------------------------- #
def construir_final(res: dict, comp: dict | None,
                    abl_r: dict | None, abl_v: dict | None) -> str:
    r, v = res["resnet50"], res["vit_small"]
    disponibles = [k for k in ETIQUETA if res.get(k)]

    cuerpo = portada(
        "Clasificación del estado de\\\\madurez de paltas Hass",
        "ResNet-50, ViT-S/16 e híbridos con presupuesto equiparado",
        AUTOR, FECHA_FINAL,
        "14.710 fotografías, 478 frutas, índice ordinal de 5 niveles")

    cuerpo += frame_problema("")

    cuerpo += frame(
        "Datos y limitaciones",
        "Hass Avocado Ripening Photographic Dataset, Mendeley, CC BY 4.0",
        kpis([("Imágenes", "14.710", "verdeosc"), ("Frutas", "478", "azul"),
              ("Train/val/test", "334/72/72", "morado"), ("Licencia", "CC BY 4.0", "rojo")])
        + "\n\\vspace{0.6em}\n" + items([
            ("Fuga longitudinal: resuelta.", True),
            "6.871 pares de días consecutivos de la misma fruta. Partimos frutas, no "
            "imágenes, con un \\texttt{assert} que falla si una fruta aparece en dos "
            "particiones.",
            ("Domain gap laboratorio $\\rightarrow$ terreno: no resuelta, declarada.", True),
            "Fondo blanco uniforme, una sola fruta centrada, iluminación difusa "
            "controlada, distancia fija. Una foto de celular en una feria está fuera "
            "de distribución. Es la limitación que decide si esto sirve en un packing "
            "o solo en un paper.",
            ("Un lote, una cosecha, un cultivar, etiquetas subjetivas.", True),
            "478 frutas portuguesas de 2022, solo Hass, etiquetado visual sin acuerdo "
            "inter-evaluador reportado: no conocemos el techo humano de esta tarea.",
        ], tam=r"\scriptsize", sep="0.2em")
        + "\n" + r"\cierre{Detalle completo en la data card "
                 r"(\texttt{entregables/final/data\_card.md}).}")

    cuerpo += frame_redes(incluir_hibridos=True)
    cuerpo += frame_comparabilidad(r, v)
    cuerpo += FRAME_RECETA

    cuerpo += seccion("01", "Resultados")

    filas = []
    for k in disponibles:
        d = res[k]
        t = d["test"]
        filas.append([ETIQUETA[k], f"{num(d['params_M'], 1)}\\,M", num(d["gflops"], 1),
                      num(t["qwk"], 3), pct(t["accuracy"]), num(t["macro_f1"], 3),
                      num(t["mae"], 3), pct(t["off_by_one"]), f"{d['train_minutes']:.0f}"])
    mejor = max(range(len(disponibles)), key=lambda i: res[disponibles[i]]["test"]["qwk"])
    cuerpo += frame(
        "Tabla comparativa completa",
        "Test: 2.262 imágenes de 72 frutas nunca vistas",
        tabla(["Modelo", "Params", "GFLOPs", "QWK", "Accuracy", "Macro-F1", "MAE",
               "Acc. $\\pm$1", "min"], filas, spec="lcccccccc", tam=r"\scriptsize",
              destacar=mejor)
        + "\n" + r"\cierre{Fila destacada: mejor QWK. El ViT-Hybrid R26 usa pesos de "
                 r"ImageNet-21k y más parámetros: es cota superior de referencia, "
                 r"no competidor equiparable.}")

    puntos = lineas_significancia(comp) if comp else []
    puntos += [
        "Todas las matrices son bandeadas: los errores de distancia $\\geq 2$ son casi "
        "inexistentes. Ningún modelo confunde una palta verde con una sobremadura.",
        "El cuello de botella son las clases intermedias (2, 3 y 4), donde la frontera "
        "es un corte continuo y las etiquetas son más subjetivas.",
    ]
    cuerpo += frame(
        "ResNet frente a ViT: la comparación, no solo el número final",
        "Todas las diferencias contra ResNet-50, bootstrap pareado agrupado por fruta",
        columnas(figura("10_matrices_confusion_final", r"\textwidth", "5.0cm"),
                 items(puntos, tam=r"\tiny", sep="0.25em"),
                 "0.50", "0.46"))

    # --- ablación ---
    rs, vs = res.get("resnet50_scratch"), res.get("vit_small_scratch")
    if rs and vs:
        d_r = r["test"]["qwk"] - rs["test"]["qwk"]
        d_v = v["test"]["qwk"] - vs["test"]["qwk"]
        a_r = r["test"]["accuracy"] - rs["test"]["accuracy"]
        a_v = v["test"]["accuracy"] - vs["test"]["accuracy"]
        tab = tabla(
            ["", "ResNet-50", "ViT-S/16", "Lectura"],
            [["QWK con ImageNet-1k", num(r["test"]["qwk"], 4), num(v["test"]["qwk"], 4), ""],
             ["QWK desde cero", num(rs["test"]["qwk"], 4), num(vs["test"]["qwk"], 4), ""],
             ["$\\Delta$ QWK sin pre-entren.", signo(-d_r), signo(-d_v),
              f"el ViT pierde {num(d_v / d_r, 1)}$\\times$ más"],
             ["$\\Delta$ accuracy sin pre-entren.",
              signo(-100 * a_r, 1) + "\\,pts", signo(-100 * a_v, 1) + "\\,pts",
              f"el ViT pierde {num(a_v / a_r, 1)}$\\times$ más"]],
            spec="lccl", tam=r"\scriptsize")
        cuerpo += frame(
            "Por qué difieren (o no) ResNet y ViT",
            "La ablación sin pre-entrenamiento es la evidencia",
            tab + "\n\\vspace{0.5em}\n" + items([
                ("La ResNet trae el sesgo inductivo incorporado en la arquitectura.", True),
                "La convolución asume localidad y equivarianza a la traslación. "
                "Eso no hay que aprenderlo.",
                ("El ViT tiene que aprender ese sesgo de los datos.", True),
                "La auto-atención es global desde la primera capa y no asume nada sobre "
                "estructura espacial. Con 10.208 imágenes de 334 frutas no alcanza para "
                "descubrirlo desde cero; con ImageNet-1k detrás, sí.",
                ("Es un problema de textura local, no de forma global.", True),
                "El índice se lee en el color y las manchas de la cáscara, justo donde la "
                "convolución es fuerte. El campo receptivo global del ViT aporta poco.",
            ], tam=r"\scriptsize", sep="0.2em")
            + "\n" + r"\cierre{Cada modelo comparado contra \emph{su propia} versión "
                     r"pre-entrenada. Las cuatro caídas son significativas ($p<0{,}0001$). "
                     r"El sesgo inductivo es un sustituto de datos.}")

    # --- híbridos ---
    hib = [k for k in ("hybrid_fusion", "hybrid_vit_r26") if res.get(k)]
    if hib:
        tarjetas = [(ETIQUETA[k], f"QWK {num(res[k]['test']['qwk'], 3)}", "morado")
                    for k in hib]
        det = items([
            f"{ETIQUETA[k]}: {num(res[k]['params_M'], 1)}\\,M params, "
            f"{num(res[k]['gflops'], 1)} GFLOPs, acc {pct(res[k]['test']['accuracy'])}"
            for k in hib], tam=r"\scriptsize", sep="0.15em")
        pts = []
        if comp:
            for kk, dd in comp.get("desacuerdo", {}).items():
                if kk.startswith("hybrid_fusion"):
                    pts.append(f"En su tabla de desacuerdo, fusión y ResNet fallan "
                               f"\\emph{{a la vez}} en {pct(dd['ambos_fallan'])} de las "
                               f"imágenes: más que el 18{{,}}0\\,\\% del par ResNet/ViT. "
                               f"La fusión no elimina los casos difíciles, los absorbe.")
        pts += [
            ("La fusión tardía cuesta el doble de cómputo por imagen.", True),
            "Dos backbones completos en el forward. La pregunta no es si mejora, sino si "
            "mejora lo suficiente para justificar $2\\times$ de latencia.",
            "Alcanza su mejor checkpoint en la \\textbf{época 2} (3{,}5 min) con los "
            "backbones congelados: las representaciones ya contenían la información, "
            "lo que faltaba era combinarlas.",
        ]
        cuerpo += frame(
            "Los híbridos", "\\textquestiondown{}Suma combinar ambos enfoques?",
            kpis(tarjetas, ancho=r"0.47\textwidth") + "\n\\vspace{0.3em}\n" + det
            + "\n\\vspace{0.3em}\n"
            + columnas(items(pts, tam=r"\tiny", sep="0.22em"),
                       figura("13_costo_beneficio_final", r"\textwidth", "2.9cm"),
                       "0.52", "0.44"))

    cuerpo += frame(
        "Dónde mira cada arquitectura",
        "Grad-CAM en la ResNet, attention rollout en el ViT",
        columnas(figura("20_interpretabilidad_desacuerdo", r"\textwidth", "5.4cm"),
                 items([
                     "No usamos la misma técnica en ambos a propósito: Grad-CAM necesita "
                     "un mapa de activaciones con estructura espacial, y en un ViT ese "
                     "mapa es una secuencia de tokens que produce resultados ruidosos.",
                     "Attention rollout acumula las matrices de atención de todas las "
                     "capas para estimar cuánto aporta cada parche al token \\texttt{CLS}.",
                     ("La ResNet se activa sobre texturas locales: manchas, arrugas, "
                      "zonas oscuras.", True),
                     ("El ViT reparte la atención sobre regiones más extensas.", True),
                     "En varios casos de desacuerdo, el Grad-CAM de la ResNet se activa "
                     "sobre el \\emph{fondo}, fuera de la fruta, y son justamente esos "
                     "los casos en que se equivoca.",
                 ], tam=r"\tiny", sep="0.25em"),
                 "0.30", "0.66"))

    # --- velocidad vs precisión ---
    candidatos = [k for k in disponibles if "scratch" not in k]
    base_gf, base_q = r["gflops"], r["test"]["qwk"]
    ver = {
        "resnet50": "Referencia. La recomendación operativa.",
        "vit_small": "Empata en QWK al mismo costo y converge $2\\times$ más rápido.",
        "hybrid_fusion": "Mejora real pero marginal por $2\\times$ de cómputo.",
        "hybrid_vit_r26": "Mejor QWK y más barato, pero con ventaja de ImageNet-21k.",
    }
    filas = []
    for k in candidatos:
        d = res[k]
        filas.append([ETIQUETA[k], num(d["test"]["qwk"], 4),
                      "---" if k == "resnet50" else signo(d["test"]["qwk"] - base_q),
                      num(d["gflops"], 2), num(d["gflops"] / base_gf, 2) + "$\\times$",
                      ver.get(k, "")])
    cuerpo += frame(
        "Decisión: velocidad frente a precisión",
        "Qué modelo llevaríamos a una línea de empaque",
        tabla(["Modelo", "QWK", "$\\Delta$ QWK", "GFLOPs", "Costo rel.", "Veredicto"],
              filas,
              spec=r"l cccc >{\raggedright\arraybackslash}p{0.34\textwidth}",
              tam=r"\tiny", destacar=candidatos.index("resnet50"))
        + "\n\\vspace{0.4em}\n" + items([
            ("El criterio no es el QWK máximo, es el QWK por unidad de cómputo.", True),
            "En un packing la inferencia corre sobre miles de frutas por hora, "
            "probablemente en un equipo sin GPU dedicada.",
            "Una diferencia de QWK que no es estadísticamente significativa no justifica "
            "duplicar el cómputo: si los intervalos se solapan, gana el modelo más barato.",
            "Con accuracy $\\pm$1 clase por sobre el 99\\,\\%, el error residual está "
            "dentro del ruido de etiquetado humano. Optimizar más el modelo no es donde "
            "está el retorno; cerrar el domain gap sí.",
        ], tam=r"\tiny", sep="0.25em"))

    cuerpo += frame(
        "Demo", "Gradio, casos nuevos del conjunto de test",
        items([
            ("Selección de modelo, distribución de probabilidad sobre las 5 clases y "
             "recomendación operativa.", True),
            "Muestra además el mapa de atención del modelo: la demo no es una caja negra.",
            "Reporta el índice esperado (promedio ponderado), más informativo que el "
            "\\texttt{argmax} cuando el modelo duda entre estados adyacentes.",
            "Avisa explícitamente cuando la confianza es baja y conviene revisión humana.",
            ("Los ejemplos precargados salen solo del conjunto de test: son frutas que "
             "ningún modelo vio durante el entrenamiento.", True),
            "Advierte en pantalla sobre el domain gap.",
        ], tam=r"\small")
        + "\n" + r"\cierre{\texttt{python app/gradio\_app.py} $\rightarrow$ "
                 r"\texttt{http://localhost:7860}}")

    cuerpo += frame(
        "Conclusiones y trabajo futuro",
        "Qué aprendimos y qué sigue",
        items([
            ("Con presupuesto y pre-entrenamiento equiparados, las dos arquitecturas "
             "resuelven esta tarea a un nivel comparable.", True),
            "La diferencia relevante no está entre CNN y transformer, sino entre tener o "
            "no tener pre-entrenamiento: ahí sí se separan, y de forma asimétrica.",
            ("El techo de esta tarea lo pone el etiquetado, no la arquitectura.", True),
            "Sin acuerdo inter-evaluador reportado no sabemos cuánto del error residual "
            "es del modelo y cuánto es ruido de la etiqueta.",
            ("Trabajo futuro, en orden de retorno esperado:", True),
            "1) Recolectar fotos de celular en condiciones reales chilenas y medir la "
            "caída. 2) Pérdida ordinal explícita (CORAL) en vez de entropía cruzada plana. "
            "3) Predicción de vida útil restante en días, que es la pregunta que de verdad "
            "importa en la cadena de frío.",
        ], tam=r"\small")
        + "\n" + rf"\cierre{{{REPO}}}")

    return cuerpo


# --------------------------------------------------------------------------- #
def main() -> None:
    res = {k: cargar(k) for k in ETIQUETA}
    if not res.get("resnet50") or not res.get("vit_small"):
        raise SystemExit("Faltan los baselines. Entrena resnet50 y vit_small primero.")

    salidas = []

    trabajos = [
        ("avance", "presentacion_avance.tex", "Avance",
         construir_avance(res["resnet50"], res["vit_small"], cargar_comp("avance"))),
        ("final", "presentacion_final.tex", "Final",
         construir_final(res, cargar_comp("final"), cargar_comp("ablacion_resnet"),
                         cargar_comp("ablacion_vit"))),
    ]
    for carpeta, archivo, etiqueta, cuerpo in trabajos:
        destino = ROOT / "entregables" / carpeta
        n_figs = copiar_figuras(cuerpo, destino / "figuras")
        tex = documento(
            "", f"Clasificación de madurez de paltas Hass — {etiqueta}", AUTOR, cuerpo)
        (destino / archivo).write_text(tex, encoding="utf-8")
        salidas.append((destino / archivo, cuerpo, n_figs))

    for path, c, n_figs in salidas:
        print(f"{path.relative_to(ROOT)}  "
              f"({c.count(chr(92) + 'begin{frame}')} frames, {n_figs} figuras)")
    print("\nPara compilar: subir la carpeta completa (.tex + figuras/) a Overleaf,")
    print("o localmente:  pdflatex presentacion_avance.tex  (dos veces)")


if __name__ == "__main__":
    main()
