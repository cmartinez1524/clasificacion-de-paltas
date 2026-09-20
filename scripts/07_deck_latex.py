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
    cajas,
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


def frame_redes() -> str:
    """Las dos arquitecturas explicadas con una imagen y una analogia.

    Deliberadamente sin jerga: la figura hace el trabajo y el texto solo la
    subtitula. Los nombres tecnicos (convolucion, auto-atencion) aparecen una
    vez entre parentesis, para que queden asociados a algo concreto.
    """
    return frame(
        "Las dos redes, en una imagen", "La diferencia es CÓMO miran la foto",
        figura("30_como_mira_cada_red", r"0.95\textwidth", "4.25cm")
        + "\n\\vspace{0.25em}\n"
        + columnas(
            items([
                "Va de a pedacitos y reutiliza el mismo detector en toda la foto "
                "\\textit{(convolución)}.",
                "Buena para detalles: manchas, arrugas, tono.",
            ], tam=r"\tiny", sep="0.12em"),
            items([
                "Parte la foto en 196 cuadraditos y los compara todos contra todos "
                "\\textit{(auto-atención)}.",
                "Buena para relacionar zonas distantes.",
            ], tam=r"\tiny", sep="0.12em"),
            "0.47", "0.47")
        + "\n" + r"\cierre{Ninguna se programó desde cero: las dos vienen de "
                 r"\texttt{timm}, la biblioteca estándar de modelos de visión.}")


def frame_por_que_entrenan_distinto() -> str:
    """Por que uno depende tanto mas del pre-entrenamiento que el otro."""
    return frame(
        "Por qué entrenan tan distinto",
        "Todo se explica por lo que cada red ya sabe antes de empezar",
        cajas([
            ("azul", "ResNet-50 trae dos reglas de fábrica",
             items([
                 "Lo que importa está \\textbf{cerca}",
                 "\\textbf{No importa} en qué parte de la foto esté",
             ], tam=r"\scriptsize", sep="0.1em")
             + "\n\\vspace{0.25em}\n"
             + r"{\tiny\color{grisSuave}\itshape Una mancha café es una mancha café, "
               r"arriba o abajo de la fruta. Eso no lo aprende: ya lo trae.}"),
            ("rojo", "ViT-S/16 no trae ninguna",
             items([
                 "Todos los cuadraditos le parecen \\textbf{iguales}",
                 "Ni siquiera sabe cuáles son \\textbf{vecinos}",
             ], tam=r"\scriptsize", sep="0.1em")
             + "\n\\vspace{0.25em}\n"
             + r"{\tiny\color{grisSuave}\itshape Tiene que deducir de los ejemplos hasta "
               r"la noción de \guillemotleft{}al lado\guillemotright{}.}"),
        ])
        + "\n\\vspace{0.55em}\n"
        + items([
            ("El ViT necesita muchísimos más ejemplos.", True),
            "La ResNet arranca sabiendo mirar imágenes; el ViT tiene que descubrir "
            "primero \\emph{cómo} se mira una imagen, y recién después aprender de paltas.",
            ("Por eso los dos parten desde ImageNet.", True),
            "Antes de ver una sola palta ya vieron 1,3 millones de fotos de cosas "
            "cotidianas. Ahí el ViT compensa: llega con el oficio aprendido.",
            ("Y por eso el ViT es más delicado de entrenar.", True),
            "Entrenar es bajar un cerro a ciegas y el \\emph{learning rate} es el tamaño "
            "del paso. La ResNet aguanta pasos grandes; el ViT se cae.",
        ], tam=r"\tiny", sep="0.2em"))


def frame_hibridos_arquitectura() -> str:
    """Que es cada hibrido, en criollo. Solo va en el deck final."""
    return frame(
        "Los dos híbridos", "Dos formas distintas de juntar ambas ideas",
        columnas(
            r"{\small\textbf{\color{morado}1. Fusión tardía}}" + "\n"
            + items([
                "Las dos redes miran la misma foto por separado y al final se juntan "
                "sus dos opiniones para decidir.",
                "Como pedir una segunda opinión médica.",
                "Es el único modelo que escribimos nosotros "
                "(\\texttt{src/paltas/models.py}).",
                ("Cuesta el doble: corre las dos redes enteras.", True),
            ], tam=r"\scriptsize", sep="0.15em"),
            r"{\small\textbf{\color{morado}2. ViT-Hybrid R26+S/32}}" + "\n"
            + items([
                "Una sola red: la parte convolucional procesa la foto primero y le "
                "entrega el resultado al transformer.",
                "En vez de dos opiniones, es una cadena de montaje.",
                "Es la arquitectura del paper original de ViT.",
                ("Tiene ventaja: más parámetros y más pre-entrenamiento.", True),
            ], tam=r"\scriptsize", sep="0.15em"),
            "0.47", "0.47")
        + "\n" + r"\cierre{Ninguno de los dos es equiparable a los baselines, y hay que "
                 r"decirlo: van como referencia, no como competidores.}")


def frame_comparabilidad(r: dict, v: dict) -> str:
    dp = abs(r["params_M"] - v["params_M"]) / max(r["params_M"], v["params_M"]) * 100
    dg = abs(r["gflops"] - v["gflops"]) / max(r["gflops"], v["gflops"]) * 100
    t = tabla(
        ["Qué igualamos", "ResNet-50", "ViT-S/16", "Diferencia"],
        [["Tamaño del \\guillemotleft{}cerebro\\guillemotright{} (conexiones)",
          f"{num(r['params_M'], 1)} millones", f"{num(v['params_M'], 1)} millones",
          f"{num(dp, 1)}\\,\\%"],
         ["Esfuerzo que hace por foto",
          f"{num(r['gflops'], 1)} mil millones", f"{num(v['gflops'], 1)} mil millones",
          f"{num(dg, 1)}\\,\\%"],
         ["Fotos que vio antes de empezar",
          "1,3 millones", "1,3 millones", "ninguna"]],
        spec="lccc", tam=r"\scriptsize")
    cuerpo = t + "\n\\vspace{0.5em}\n" + items([
        ("Si una red es más grande y gana, no aprendimos nada: ganó por grande.", True),
        "Por eso igualamos el tamaño (parámetros), el trabajo que hace por foto "
        "(GFLOPs) y --- lo que casi nadie iguala --- \\textbf{cuántas fotos vio antes}.",
        ("El tercer punto es el que más distorsiona.", True),
        "El ViT que usa todo el mundo viene entrenado con 14 millones de fotos; la "
        "ResNet, con 1,3 millones. Comparándolos así estaríamos midiendo quién estudió "
        "más, no qué arquitectura es mejor.",
        "Usamos una versión del ViT llamada \\textbf{DeiT-S}: es exactamente la misma "
        "red, pero entrenada con las mismas 1,3 millones de fotos que la ResNet. "
        "Ahí sí la comparación es limpia.",
    ], tam=r"\scriptsize", sep="0.25em")
    return frame(
        "Cómo hacemos que la comparación sea justa",
        "Igualar tres cosas, no solo el tamaño",
        cuerpo + "\n" + r"\cierre{Como comparar dos atletas: sirve si entrenaron lo mismo "
                        r"y compiten en la misma categoría de peso.}")


FRAME_RECETA = frame(
    "Cómo las entrenamos", "Todo igual para las dos, salvo lo que tiene que cambiar",
    columnas(
        r"{\small\textbf{Idéntico para ambas}}" + "\n" + items([
            "Las mismas fotos, en el mismo orden",
            "Las mismas deformaciones de las fotos",
            "La misma cantidad de vueltas al dataset",
            "El mismo criterio para elegir el mejor modelo",
            "El mismo criterio para cortar el entrenamiento",
        ], tam=r"\scriptsize", sep="0.16em"),
        r"{\small\textbf{Distinto en el ViT}}" + "\n"
        + r"{\scriptsize\itshape Siguiendo con la imagen del cerro:}" + "\n"
        + items([
            ("Le pedimos pasos 3 veces más cortos.", True),
            "Es el \\emph{learning rate}. Con pasos largos el ViT tropieza y "
            "deja de aprender.",
            ("Y que empiece a caminar más lento.", True),
            "Es el \\emph{warmup}: las primeras vueltas van a media máquina "
            "hasta que agarra el ritmo.",
        ], tam=r"\scriptsize", sep="0.16em"),
        "0.47", "0.47")
    + "\n\\vspace{0.4em}\n"
    + r"{\scriptsize\textbf{Un detalle que suele hacerse mal:} en casi cualquier otra "
      r"tarea conviene alterar mucho los colores de las fotos de entrenamiento, para que "
      r"el modelo no dependa del color. Acá el color \emph{es} la respuesta: si alteramos "
      r"mucho el tono, una palta clase 2 se vuelve idéntica a una clase 4 pero con la "
      r"etiqueta vieja. Le estaríamos enseñando mal. Por eso el color casi no se toca; "
      r"girar y recortar la foto, en cambio, sí.}"
    + "\n" + r"\cierre{Igualar el learning rate \guillemotleft{}por justicia"
             r"\guillemotright{} no sería más justo: "
             r"sería solo peor para el ViT.}")


FRAME_TRAMPA = frame(
    "El riesgo de que la IA haga trampa",
    "El error que habría inflado todos los resultados",
    columnas(
        items([
            ("A cada palta le sacaron una foto por día, hasta 26 días seguidos.", True),
            "La foto del día 5 y la del día 6 de la misma palta son casi idénticas: "
            "misma piel, mismas manchas, misma forma.",
            ("Si repartimos las fotos al azar, la trampa es inevitable.", True),
            "El día 5 queda en el material de estudio y el día 6 en la prueba. "
            "El modelo \\emph{reconoce esa palta} en vez de juzgar su madurez, "
            "como un alumno que vio las respuestas antes del examen.",
            "El resultado saldría altísimo y sería completamente falso.",
            ("Por eso repartimos las 478 paltas, no las 14.710 fotos.", True),
            "Todas las fotos de una misma palta van juntas: o están en el estudio, "
            "o están en la prueba. Nunca en las dos.",
        ], tam=r"\tiny", sep="0.2em"),
        figura("04_particiones", r"\textwidth", "2.9cm"),
        "0.52", "0.44")
    + "\n" + r"\cierre{El código tiene un chequeo automático que falla si alguna "
             r"palta llegara a aparecer en los dos lados.}")


def frame_baseline(nombre: str, d: dict, color: str) -> str:
    """Resultados de un baseline, con dos cifras grandes y el resto en letra chica.

    El publico no es tecnico: cinco metricas del mismo tamano abruman y no
    dicen nada. Van grandes las dos que se entienden sin explicacion previa
    --cuanto se equivoca y por cuanto-- y el resto queda en una linea menor,
    con el detalle completo en el anexo.
    """
    t = d["test"]
    grande = kpis([
        ("Acierta, o se pasa por un solo escalón", pct(t["off_by_one"]), color),
        ("Cuando se equivoca, se equivoca por", f"{num(t['mae'], 2)} escalones", color),
    ], ancho=r"0.47\textwidth")
    menores = (r"{\scriptsize\color{grisSuave}Acierta la clase exacta en el "
               + pct(t["accuracy"])
               + r" de las fotos \; \textbullet\; puntaje con castigo (QWK) "
               + num(t["qwk"], 3)
               + r" \; \textbullet\; Macro-F1 " + num(t["macro_f1"], 3) + r"}")
    lectura = items([
        ("Casi nunca se equivoca feo.", True),
        f"En {pct(t['off_by_one'])} de las fotos dice la clase correcta o una "
        "vecina. Confundir una palta verde con una podrida no le pasó nunca.",
        ("Se equivoca por menos de medio escalón.", True),
        f"{num(t['mae'], 2)} en una escala de 5 niveles: del orden de medio día "
        "de maduración.",
        ("Anda igual de bien en las tres temperaturas.", True),
        "No está haciendo trampa con el atajo "
        "\\guillemotleft{}verde $\\Rightarrow$ seguro venía del refrigerador"
        "\\guillemotright{}.",
    ], tam=r"\scriptsize", sep="0.22em")
    sub = (f"Un cerebro de {num(d['params_M'], 1)} millones de conexiones, "
           f"{d['train_minutes']:.0f} minutos de entrenamiento")
    return frame(
        f"Resultado: {nombre}", sub,
        grande + "\n\\vspace{0.45em}\n" + menores + "\n\\vspace{0.6em}\n" + lectura
        + "\n" + r"\cierre{Medido sobre 2.262 fotos de 72 paltas que el modelo "
                 r"nunca había visto.}")


def frame_empate(r: dict, v: dict, comp: dict | None) -> str:
    """Las dos redes comparadas, sin jerga estadistica.

    Los intervalos de confianza y los p-valores estan en el anexo y en el
    informe. Aca solo la conclusion, que es la que le sirve a la audiencia.
    """
    puntos = [
        ("Las dos rinden prácticamente igual.", True),
        "La ResNet acierta la clase exacta un poco más seguido. Pero cuando el ViT "
        "se equivoca, se equivoca por menos. Se compensa.",
        "Hicimos la prueba estadística formal y la diferencia entre ambas no es "
        "concluyente: para efectos prácticos, empatan.",
    ]
    if comp:
        for _, dd in comp.get("desacuerdo", {}).items():
            puntos += [
                ("Pero no se equivocan en las mismas fotos.", True),
                f"Fallan las dos a la vez en solo {pct(dd['ambos_fallan'])} de los "
                f"casos. Si alguien pudiera elegir siempre cuál de las dos tiene "
                f"razón, acertaría {pct(dd['oraculo_accuracy'])} en vez de "
                f"{pct(max(r['test']['accuracy'], v['test']['accuracy']))}.",
                "Eso es lo que van a intentar aprovechar los modelos híbridos.",
            ]
            break
    return frame(
        "ResNet-50 frente a ViT-S/16", "El resultado central del avance",
        figura("33_matrices_didactica", r"0.86\textwidth", "3.6cm")
        + "\n\\vspace{0.3em}\n" + items(puntos, tam=r"\tiny", sep="0.2em")
        + "\n" + r"\cierre{Los números completos y las pruebas estadísticas están "
                 r"en el anexo.}")


def frame_mejoras_gratis() -> str:
    """Dos mejoras de inferencia que no requieren reentrenar.

    Se calculan en scripts/09_mejoras_inferencia.py; si ese script no corrio,
    el frame simplemente no se agrega.
    """
    datos = cargar_comp("mejoras_inferencia")
    if not datos:
        return ""

    def val(modelo: str, variante: str) -> float:
        return next(f["accuracy"] for f in datos[modelo] if f["variante"] == variante)

    r_base, r_caras = val("resnet50", "por foto"), val("resnet50", "por fruta-día")
    return frame(
        "Dos mejoras que salieron gratis",
        "Sin reentrenar nada, solo cambiando cómo se usa el modelo ya entrenado",
        figura("34_mejoras_inferencia", r"0.88\textwidth", "3.9cm")
        + "\n\\vspace{0.3em}\n"
        + items([
            ("1. Mirar las dos caras de la palta antes de decidir.", True),
            "El dataset fotografía cada fruta por ambos lados el mismo día. Hasta ahora "
            "clasificábamos cada foto por separado; juntar las dos opiniones sube a "
            f"la ResNet de {pct(r_base)} a {pct(r_caras)}, y mejora a los cuatro "
            "modelos. En una planta igual se fotografían las dos caras: es gratis.",
            ("2. Mostrarle también la foto espejada y promediar.", True),
            "Ayuda a las convolucionales, pero al ViT lo empeora un poco. Lo reportamos "
            "así: no todas las recetas estándar sirven en todos los modelos.",
        ], tam=r"\tiny", sep="0.22em")
        + "\n" + r"\cierre{Antes de comprar más cómputo conviene revisar si se está "
                 r"usando bien el modelo que ya se tiene.}")


def frames_anexo(modelos: list[str], res: dict, comp: dict | None) -> str:
    """Anexo con el detalle que se saco del cuerpo de la presentacion.

    No se borra: se mueve. La audiencia general no lo necesita, pero si alguien
    del jurado pregunta por los intervalos de confianza o por el desglose por
    temperatura, esta a una slide de distancia.
    """
    salida = seccion("A", "Anexo")

    filas = []
    for k in modelos:
        d = res[k]
        t = d["test"]
        filas.append([ETIQUETA[k], f"{num(d['params_M'], 1)}\\,M", num(d["gflops"], 1),
                      num(t["qwk"], 4), pct(t["accuracy"]), num(t["macro_f1"], 3),
                      num(t["mae"], 3), pct(t["off_by_one"]),
                      f"{d['train_minutes']:.0f}"])
    salida += frame(
        "Anexo: todas las métricas", "Test: 2.262 imágenes de 72 frutas nunca vistas",
        tabla(["Modelo", "Params", "GFLOPs", "QWK", "Accuracy", "Macro-F1", "MAE",
               "Acc. $\\pm$1", "min"], filas, spec="lcccccccc", tam=r"\scriptsize")
        + "\n\\vspace{0.5em}\n"
        + items([
            "\\textbf{QWK}: kappa de Cohen con pesos cuadráticos. Penaliza el error "
            "según el cuadrado de la distancia entre clases y corrige por acuerdo "
            "azaroso. Es la métrica con la que elegimos el mejor modelo.",
            "\\textbf{MAE}: error medio en escalones del índice de madurez.",
            "\\textbf{Acc. $\\pm$1}: fracción de predicciones en la clase correcta "
            "o una adyacente.",
            "\\textbf{Macro-F1}: F1 promediado por clase; controla que las clases "
            "minoritarias no queden abandonadas.",
        ], tam=r"\tiny", sep="0.18em"))

    if comp:
        puntos = lineas_significancia(comp)
        puntos.append(
            "Los intervalos y los $p$-valores salen de un \\emph{bootstrap pareado "
            "agrupado por fruta}: las 2.262 imágenes vienen de solo 72 paltas, así "
            "que remuestrear imágenes daría intervalos artificialmente angostos. "
            "Se remuestrean frutas, 2.000 réplicas.")
        salida += frame(
            "Anexo: pruebas estadísticas",
            "Diferencias contra ResNet-50, con intervalo de confianza del 95\\,\\%",
            items(puntos, tam=r"\scriptsize", sep="0.25em"))

    # solo los modelos pre-entrenados: con los seis la tabla no entra en el slide
    filas_g = []
    for k in [m for m in modelos if "scratch" not in m]:
        for g, x in res[k].get("test_por_grupo", {}).items():
            filas_g.append([ETIQUETA[k], GRUPO.get(g, g), str(int(x["n"])),
                            pct(x["accuracy"]), num(x["mae"], 3)])
    if filas_g:
        salida += frame(
            "Anexo: desglose por temperatura de almacenamiento",
            "Sirve para detectar si el modelo se apoya en un atajo",
            tabla(["Modelo", "Grupo", "n", "Accuracy", "MAE"], filas_g,
                  spec="llccc", tam=r"\tiny")
            + "\n" + r"\cierre{Desempeño parejo entre grupos: el modelo no está "
                     r"usando el atajo \guillemotleft{}verde $\Rightarrow$ "
                     r"refrigerada\guillemotright{}.}")
    return salida


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
    tabla(["Riesgo", "Por qué es real acá", "Qué hicimos"],
          [["Que la IA haga trampa",
            "Hay 6.871 pares de fotos casi idénticas de la misma palta",
            "Repartir paltas, no fotos, con chequeo automático"],
           ["El choque con la vida real",
            "Las fotos son de laboratorio: fondo blanco, luz perfecta, una fruta "
            "centrada. Un packing no se parece a eso",
            "Lo declaramos como la limitación principal. Hay que salir a sacar "
            "fotos con celular"],
           ["Alterar el color arruinaría el aprendizaje",
            "El color \\emph{es} la respuesta: si lo cambiamos, la etiqueta queda mal",
            "Alteramos el color lo mínimo; giramos y recortamos sin problema"],
           ["Una temperatura domina los datos",
            "El grupo refrigerado aporta el 60\\,\\% de las fotos y casi todas las "
            "paltas verdes",
            "Compensamos en el entrenamiento y revisamos el resultado temperatura "
            "por temperatura"],
           ["Nadie sabe cuál es el techo",
            "Las etiquetas las puso una persona mirando; no hay una segunda opinión "
            "con qué contrastar",
            "Usamos métricas que perdonan el error de un escalón y no perseguimos "
            "el 100\\,\\%"],
           ["Darle ventaja a una de las dos redes",
            "El ViT que se usa por defecto vino entrenado con 14 millones de fotos, "
            "no 1,3",
            "Elegimos la versión del ViT que vio exactamente las mismas fotos que "
            "la ResNet"]],
          spec=r">{\raggedright\arraybackslash}p{0.20\textwidth}"
               r">{\raggedright\arraybackslash}p{0.38\textwidth}"
               r">{\raggedright\arraybackslash}p{0.34\textwidth}",
          tam=r"\tiny")
    + "\n" + r"\cierre{El choque con la vida real es el riesgo que decide si esto "
             r"sirve en una planta empacadora o solo en un informe.}")


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

    cuerpo += FRAME_TRAMPA

    cuerpo += frame_redes()
    cuerpo += frame_por_que_entrenan_distinto()
    cuerpo += frame_comparabilidad(r, v)
    cuerpo += FRAME_RECETA

    cuerpo += seccion("01", "Resultados preliminares")
    cuerpo += frame_baseline("ResNet-50", r, "azul")
    cuerpo += frame_baseline("ViT-S/16", v, "rojo")

    cuerpo += frame_empate(r, v, comp)

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
            ("Desarmar el modelo para ver qué pieza importa.", True),
            "Reentrenar las dos redes sin ImageNet, solo con las fotos de paltas. "
            "Debería mostrar que el ViT depende mucho más de haber estudiado antes.",
            ("Dos formas de juntar ambas redes en una.", True),
            "Una que las hace opinar por separado y combina las dos respuestas; otra "
            "que las encadena, con la parte convolucional alimentando a la otra.",
            ("Ver dónde mira cada red.", True),
            "Pintar sobre la foto las zonas que cada modelo usó para decidir, "
            "sobre todo en los casos donde las dos discrepan.",
            ("Una demo que funcione: subir una foto y ver la respuesta.", True),
            ("Y una recomendación explícita: cuál llevaríamos a una planta real, "
             "pesando precisión contra costo.", True),
        ], tam=r"\scriptsize", sep="0.25em")
        + "\n" + rf"\cierre{{Repositorio: {REPO}}}")

    cuerpo += frames_anexo(["resnet50", "vit_small"],
                           {"resnet50": r, "vit_small": v}, comp)

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
            ("Que la IA hiciera trampa: resuelto.", True),
            "Hay 6.871 pares de fotos casi idénticas de la misma palta en días "
            "seguidos. Repartimos las 478 paltas, no las 14.710 fotos, con un "
            "chequeo automático que falla si alguna aparece en los dos lados.",
            ("El choque con la vida real: no resuelto, declarado.", True),
            "Son fotos de laboratorio: fondo blanco, una sola fruta centrada, luz "
            "pareja, siempre a la misma distancia. Una foto de celular en una feria "
            "no se parece en nada. Es la limitación que decide si esto sirve en una "
            "planta empacadora o solo en un informe.",
            ("Un solo lote, una sola cosecha, una sola variedad.", True),
            "478 paltas portuguesas de 2022, todas Hass. Y las etiquetas las puso "
            "una persona mirando, sin una segunda opinión con qué contrastar: no "
            "sabemos cuánto acertaría un experto humano en esta misma tarea.",
        ], tam=r"\scriptsize", sep="0.2em")
        + "\n" + r"\cierre{Detalle completo en la data card "
                 r"(\texttt{entregables/final/data\_card.md}).}")

    cuerpo += frame_redes()
    cuerpo += frame_por_que_entrenan_distinto()
    cuerpo += frame_hibridos_arquitectura()
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
    # Version simplificada: tres columnas que se entienden sin explicacion.
    # La tabla completa con las seis metricas esta en el anexo.
    filas_simple = []
    for k in disponibles:
        t = res[k]["test"]
        filas_simple.append([
            ETIQUETA[k],
            pct(t["off_by_one"]),
            f"{num(t['mae'], 2)} escalones",
            pct(t["accuracy"]),
        ])
    cuerpo += frame(
        "Todos los modelos, lado a lado",
        "Medido sobre 2.262 fotos de 72 paltas que ningún modelo había visto",
        # el encabezado largo va en una columna de ancho fijo para que envuelva
        # solo: un \\ dentro de la celda cerraria la fila
        tabla(["Modelo", "Acierta o se pasa por un escalón",
               "Error promedio", "Clase exacta"],
              filas_simple,
              spec=r"l >{\centering\arraybackslash}p{0.20\textwidth} "
                   r">{\centering\arraybackslash}p{0.17\textwidth} c",
              tam=r"\scriptsize", destacar=mejor)
        + "\n\\vspace{0.6em}\n"
        + items([
            ("Todos los modelos con pre-entrenamiento están sobre el 99\\,\\%.", True),
            "La diferencia entre ellos es de décimas. Los dos que aparecen abajo son "
            "los entrenados desde cero, y se nota.",
            ("El renglón que importa es el del medio.", True),
            "Menos de medio escalón de error promedio significa, en la práctica, "
            "medio día de maduración.",
        ], tam=r"\scriptsize", sep="0.22em")
        + "\n" + r"\cierre{La tabla completa, con las seis métricas técnicas, está "
                 r"en el anexo.}")

    cuerpo += frame_empate(r, v, comp)

    # --- ablación ---
    rs, vs = res.get("resnet50_scratch"), res.get("vit_small_scratch")
    if rs and vs:
        d_r = r["test"]["qwk"] - rs["test"]["qwk"]
        d_v = v["test"]["qwk"] - vs["test"]["qwk"]
        a_r = r["test"]["accuracy"] - rs["test"]["accuracy"]
        a_v = v["test"]["accuracy"] - vs["test"]["accuracy"]
        cuerpo += frame(
            "El experimento que lo demuestra",
            "Reentrenamos las dos redes SIN ImageNet, solo con las fotos de paltas",
            figura("32_dependencia_pretraining", r"0.84\textwidth", "3.6cm")
            + "\n\\vspace{0.3em}\n"
            + items([
                (f"El ViT pierde {num(d_v / d_r, 1)}$\\times$ más QWK y "
                 f"{num(a_v / a_r, 1)}$\\times$ más accuracy que la ResNet.", True),
                "Es lo que predice la idea de las \\guillemotleft{}reglas de "
                "fábrica\\guillemotright{}: la ResNet ya sabía mirar imágenes, el ViT "
                "tenía que aprenderlo, y 10.208 fotos de paltas no alcanzan para eso.",
                "Las cuatro caídas son estadísticamente sólidas ($p<0{,}0001$), y cada "
                "red se compara contra \\emph{su propia} versión con ImageNet.",
            ], tam=r"\tiny", sep="0.22em")
            + "\n" + r"\cierre{Con ImageNet las dos empatan. Sin ImageNet, no. "
                     r"La diferencia real entre ellas no es el techo, es cuántos datos "
                     r"necesitan para llegar.}")

        cuerpo += frame(
            "Y hay una segunda razón",
            "Esta tarea juega en la cancha de la convolución",
            items([
                ("El estado de madurez se lee en detalles pequeños y locales.", True),
                "El color de la cáscara, las manchas, el arrugamiento. Todo eso está "
                "repartido por la superficie de la fruta y se ve de cerca.",
                ("Justo lo que la ResNet hace bien con su lupa.", True),
                "La gran ventaja del ViT es relacionar zonas lejanas de una imagen. Acá "
                "no hay nada lejano que relacionar: todas las paltas tienen la misma "
                "forma, están centradas y sobre el mismo fondo.",
                ("Por eso el empate no es casualidad.", True),
                "En un problema donde importara la composición global de la escena, "
                "probablemente el ViT sacaría ventaja. En este, su superpoder no suma.",
            ], tam=r"\small", sep="0.3em")
            + "\n" + r"\cierre{Conclusión honesta: no es que el ViT sea peor, es que "
                     r"este problema no le pide lo que él hace mejor.}")

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
        "\\textquestiondown{}En qué se fijó el modelo para decidir?",
        "Pintamos sobre la foto las zonas que más pesaron en la respuesta",
        columnas(figura("20_interpretabilidad_desacuerdo", r"\textwidth", "5.4cm"),
                 items([
                     "Rojo = la zona que más influyó en la decisión; azul = la que "
                     "casi no se usó.",
                     "Cada red se abre con una técnica distinta, porque por dentro "
                     "funcionan distinto. Las dos responden la misma pregunta.",
                     ("La ResNet se concentra en puntos chicos: una mancha, una "
                      "arruga, una zona oscura.", True),
                     ("El ViT reparte la mirada por áreas más grandes de la "
                      "cáscara.", True),
                     ("Un hallazgo incómodo y honesto:", True),
                     "En varios de los casos en que las dos redes discrepan, la "
                     "ResNet está mirando el \\emph{fondo}, fuera de la fruta. Y son "
                     "justamente esos los casos en que se equivoca. Sin esta figura "
                     "no lo habríamos notado.",
                 ], tam=r"\tiny", sep="0.25em"),
                 "0.30", "0.66"))

    cuerpo += frame_mejoras_gratis()

    # --- velocidad vs precisión ---
    candidatos = [k for k in disponibles if "scratch" not in k]
    base_gf, base_q = r["gflops"], r["test"]["qwk"]
    ver = {
        "resnet50": "La que recomendamos. Es la referencia.",
        "vit_small": "Empata, cuesta lo mismo y se entrena en la mitad del tiempo.",
        "hybrid_fusion": "Mejora poquito y cuesta el doble. No lo vale.",
        "hybrid_vit_r26": "Mejor y más barato, pero estudió con más fotos: no es "
                          "comparación limpia.",
    }
    filas = []
    for k in candidatos:
        d = res[k]
        filas.append([ETIQUETA[k], pct(d["test"]["off_by_one"]),
                      num(d["gflops"] / base_gf, 2) + r"$\times$",
                      ver.get(k, "")])
    cuerpo += frame(
        "\\textquestiondown{}Cuál llevaríamos a una planta empacadora?",
        "No gana el más preciso: gana el que rinde mejor por lo que cuesta",
        tabla(["Modelo", "Acierta o se pasa por uno", "Cuesta", "Veredicto"],
              filas,
              spec=r"l >{\centering\arraybackslash}p{0.15\textwidth} c "
                   r">{\raggedright\arraybackslash}p{0.36\textwidth}",
              tam=r"\tiny", destacar=candidatos.index("resnet50"))
        + "\n\\vspace{0.5em}\n" + items([
            ("En una planta hay que clasificar miles de paltas por hora.", True),
            "Probablemente en un computador común, sin tarjeta gráfica potente. "
            "Ahí el costo de cada foto importa tanto como el acierto.",
            ("Si dos modelos empatan, gana el más barato.", True),
            "Duplicar el costo para ganar unas décimas no se justifica.",
            ("Y ya estamos en el techo de lo exigible.", True),
            "Con más del 99\\,\\% de aciertos dentro de un escalón, lo que queda de "
            "error probablemente también lo cometería una persona.",
        ], tam=r"\scriptsize", sep="0.22em"))

    cuerpo += frame(
        "Demo", "Subir una foto y ver qué responde el modelo",
        items([
            ("Se elige el modelo, se sube la foto y aparece la respuesta.", True),
            "Muestra qué tan seguro está de cada uno de los 5 estados, y qué hacer "
            "con esa palta: dejarla madurar, venderla hoy o descartarla.",
            ("Muestra también dónde miró para decidir.", True),
            "No es una caja negra: se ve pintada sobre la foto la zona que usó.",
            ("Avisa cuando no está seguro.", True),
            "Si duda entre dos estados vecinos lo dice, en vez de responder con "
            "falsa confianza. Ahí conviene que mire una persona.",
            ("Los ejemplos que trae cargados son paltas que ningún modelo vio nunca.", True),
            "Y advierte en pantalla que fue entrenado con fotos de laboratorio.",
        ], tam=r"\small")
        + "\n" + r"\cierre{\texttt{python app/gradio\_app.py} $\rightarrow$ "
                 r"\texttt{http://localhost:7860}}")

    cuerpo += frame(
        "Conclusiones y trabajo futuro",
        "Qué aprendimos y qué sigue",
        items([
            ("Con el mismo tamaño y el mismo estudio previo, las dos redes resuelven "
             "esto igual de bien.", True),
            "La pregunta interesante no era cuál gana, sino en qué se diferencian. "
            "Y la diferencia no está en el techo que alcanzan: está en cuántos datos "
            "necesitan para llegar.",
            ("El límite de esta tarea lo pone quien etiquetó las fotos, no la red.", True),
            "Como nadie revisó esas etiquetas dos veces, no sabemos cuánto del error "
            "que queda es del modelo y cuánto es desacuerdo entre humanos.",
            ("Qué haríamos ahora, en orden de importancia:", True),
            "1) Salir a sacar fotos con celular en un packing chileno y medir cuánto "
            "cae. 2) Enseñarle explícitamente que las clases están ordenadas, cosa que "
            "hoy solo se le pide al evaluar. 3) Predecir directamente cuántos días de "
            "vida le quedan a la palta, que es lo que de verdad necesita la cadena "
            "de frío.",
        ], tam=r"\scriptsize", sep="0.28em")
        + "\n" + rf"\cierre{{{REPO}}}")

    cuerpo += frames_anexo(disponibles, res, comp)

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
            "", f"Clasificación de madurez de paltas Hass --- {etiqueta}", AUTOR, cuerpo)
        (destino / archivo).write_text(tex, encoding="utf-8")
        salidas.append((destino / archivo, cuerpo, n_figs))

    for path, c, n_figs in salidas:
        print(f"{path.relative_to(ROOT)}  "
              f"({c.count(chr(92) + 'begin{frame}')} frames, {n_figs} figuras)")
    print("\nPara compilar: subir la carpeta completa (.tex + figuras/) a Overleaf,")
    print("o localmente:  pdflatex presentacion_avance.tex  (dos veces)")


if __name__ == "__main__":
    main()
