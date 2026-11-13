"""Genera las dos presentaciones en Beamer (LaTeX) desde las metricas reales.

Produce carpetas autocontenidas: el .tex y una subcarpeta figuras/ con solo las
imagenes que ese deck usa. Asi se puede arrastrar la carpeta completa a Overleaf
sin tocar nada.

Uso:
    python scripts/07_deck_latex.py            # solo la presentacion final
    python scripts/07_deck_latex.py --avance   # tambien regenera el avance
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

AUTOR = ("Cristóbal Martínez, Benjamín Espinoza, Felipe Alvarez, "
         "Felipe Torres, Felipe Rojo")
FECHA_AVANCE = "Avance, 21 de septiembre de 2026"
FECHA_FINAL = "Entrega final, 23 de noviembre de 2026"
REPO = r"\texttt{github.com/cmartinez1524/clasificacion-de-paltas}"

ETIQUETA = {
    "resnet50": "ResNet-50",
    "vit_small": "ViT-S/16",
    "resnet50_asimetrico": "ResNet-50 (asim.)",
    "vit_small_asimetrico": "ViT-S/16 (asim.)",
    "resnet50_meta": "ResNet-50 + temp. y día",
    "vit_small_meta": "ViT-S/16 + temp. y día",
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
        ("La tarea es \\emph{ordinal} y asimétrica: $1<2<3<4<5$.", True),
        "La palta es climatérica (libera etileno). Clasificar una fruta madura "
        "como inmadura ($\\hat{y} < y$) mete fruta madura al lote verde y echa "
        "a perder las demás.",
        "Clasificar una verde como madura ($\\hat{y} > y$) solo adelanta su "
        "salida: es un error mucho menos costoso.",
    ], tam=r"\scriptsize", sep="0.28em")
    return frame(
        "El problema", "",
        columnas(izq, TABLA_INDICE, "0.58", "0.38")
        + "\n" + r"\cierre{Medimos distancia (QWK) y dirección del error: "
                 r"MAE Asimétrico con castigo $2\times$ al falso inmaduro.}",
        nota=fecha_nota)


def frames_datos(incluir_limitaciones: bool = False) -> str:
    """Tres laminas claras para explicar el dataset: imagenes, temperaturas y split."""
    s1 = frame(
        "Los datos en imágenes: 5 etapas de madurez",
        "Hass Avocado Ripening Photographic Dataset (Mendeley, CC BY 4.0)",
        columnas(
            figura("03_ejemplos_por_clase", r"\textwidth", "5.35cm"),
            TABLA_INDICE
            + "\n\\vspace{0.35em}\n"
            + items([
                ("14.710 fotos JPEG de 800$\\times$800 px de 478 paltas Hass.", True),
                "Adquiridas 3 días tras la cosecha (abril 2022, Portugal) y "
                "fotografiadas \\textbf{a diario por ambas caras} (\\texttt{a} y "
                "\\texttt{b}) durante hasta 26 días.",
                "\\textbf{Captura estandarizada:} fondo blanco uniforme, luz "
                "difusa fija y una sola fruta centrada por foto.",
                "\\textbf{Integridad verificada:} descartamos 12 filas de la "
                "planilla sin imagen; 0 discrepancias entre Excel y archivo.",
                "\\textbf{Etiquetado visual experto:} sin penetrómetro ni doble "
                "evaluador (fronteras como 3$\\leftrightarrow$4 son continuas).",
            ], tam=r"\tiny", sep="0.16em"),
            "0.43", "0.53"))

    s2 = frame(
        "Diseño experimental: 3 temperaturas y evolución temporal",
        "Cómo se distribuyen las clases y cómo madura cada grupo",
        kpis([("Imágenes", "14.710", "verdeosc"),
              ("Frutas", "478", "azul"),
              ("Caras / día", "2 (a y b)", "morado"),
              ("Duración máx.", "26 días", "rojo")])
        + "\n\\vspace{0.35em}\n"
        + columnas(
            figura("01_distribucion_clases", r"\textwidth", "2.9cm")
            + "\n\\vspace{0.15em}\n"
            + items([
                "\\textbf{T10 (10\\,\\textdegree C):} 192 frutas, 8.866 fotos "
                "(60\\,\\% del total; pasan muchos días verdes).",
                "\\textbf{T20 (20\\,\\textdegree C) y Tamb:} 143 frutas y "
                "\\textasciitilde 2.920 fotos cada uno.",
            ], tam=r"\tiny", sep="0.12em"),
            figura("02_curvas_maduracion", r"\textwidth", "2.9cm")
            + "\n\\vspace{0.15em}\n"
            + items([
                "A 20\\,\\textdegree C o ambiente maduran en ~10 días; a "
                "10\\,\\textdegree C tardan más de 3 semanas.",
                "Desbalance moderado ($1{,}6\\times$) compensado con pesos "
                "de clase en la pérdida.",
            ], tam=r"\tiny", sep="0.12em"),
            "0.51", "0.46")
        + "\n" + r"\cierre{Dos paltas del mismo día a distinta temperatura tienen "
                 r"madurez opuesta, y dentro de cada grupo hay frutas fuera de ritmo.}")

    out = s1 + s2 + FRAME_TRAMPA
    if incluir_limitaciones:
        out += frame(
            "Limitaciones conocidas de los datos",
            "Lo que declaramos antes de pensar en llevar esto a una planta",
            items([
                ("El choque con la vida real (domain gap): no resuelto, declarado.", True),
                "Fondo blanco, luz perfecta y una sola fruta centrada. Una foto de "
                "celular en una feria o packing ---con sombras, cajones y varias "
                "paltas--- está fuera de la distribución de entrenamiento.",
                ("Un solo lote, una sola cosecha, una sola variedad.", True),
                "478 paltas portuguesas de 2022, todas Hass. No cubre Fuerte ni "
                "Edranol ni posibles variaciones de pigmentación en fruta chilena.",
                ("Solo apariencia externa y sin doble evaluador.", True),
                "Las etiquetas las puso una persona mirando la cáscara; defectos "
                "internos de pulpa no se ven y no conocemos el techo humano.",
            ], tam=r"\scriptsize", sep="0.28em")
            + "\n" + r"\cierre{Detalle completo en la data card "
                     r"(\texttt{entregables/final/data\_card.md}).}")
    return out


def frame_redes() -> str:
    """Las dos arquitecturas explicadas con una imagen y una analogia."""
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
            "cotidianas. Ahí el ViT compensa su desventaja y llega con el oficio "
            "aprendido.",
        ], tam=r"\scriptsize", sep="0.25em"))


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
        "(GFLOPs) y cuántas fotos vio antes.",
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
        cuerpo)


FRAME_RECETA = frame(
    "Cómo las entrenamos", "",
    columnas(
        r"{\small\textbf{Idéntico para ambas}}" + "\n" + items([
            "Las mismas fotos, en el mismo orden",
            "Las mismas deformaciones de las fotos",
            "La misma cantidad de vueltas al dataset",
            "El mismo criterio para elegir el mejor modelo",
            "El mismo criterio para cortar el entrenamiento",
        ], tam=r"\scriptsize", sep="0.16em"),
        r"{\small\textbf{Distinto en el ViT}}" + "\n"
        + items([
            ("Learning rate 3 veces más pequeño.", True),
            "Es el tamaño del paso. Con pasos largos el ViT tropieza y deja "
            "de aprender.",
            ("Y un arranque más lento (\\emph{warmup}).", True),
            "Las primeras vueltas van a media máquina, pero a la larga es lo "
            "mejor para el ViT.",
        ], tam=r"\scriptsize", sep="0.16em"),
        "0.47", "0.47")
    + "\n\\vspace{0.4em}\n"
    + r"{\scriptsize\textbf{Observación:} en casi cualquier otra "
      r"tarea conviene alterar mucho los colores de las fotos de entrenamiento, para que "
      r"el modelo no dependa del color. Acá el color \emph{es} la respuesta: si alteramos "
      r"mucho el tono, una palta clase 2 se vuelve idéntica a una clase 4 pero con la "
      r"etiqueta vieja. Le estaríamos enseñando mal. Por eso el color casi no se toca; "
      r"girar y recortar la foto, en cambio, sí.}"
    + "\n" + r"\cierre{Igualar el learning rate \guillemotleft{}por justicia"
             r"\guillemotright{} no sería más justo: "
             r"sería solo peor para el ViT.}")


FRAME_TRAMPA = frame(
    "El riesgo de que la IA haga trampa", "",
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
    + "\n" + r"\cierre{Implementamos un chequeo automático que falla si alguna "
             r"palta llegara a aparecer en los dos lados.}")


def frame_baseline(nombre: str, d: dict, color: str) -> str:
    """Resultados de un baseline en el deck de avance."""
    t = d["test"]
    asym_val = t.get("asymmetric_mae", t["mae"])
    under_val = t.get("underestimation_rate", 0.0)
    grande = kpis([
        ("Acierta, o se pasa por un solo escalón", pct(t["off_by_one"]), color),
        ("Error asimétrico (2$\\times$ falso inmaduro)", f"{num(asym_val, 3)}", color),
    ], ancho=r"0.47\textwidth")
    menores = (r"{\scriptsize\color{grisSuave}Clase exacta " + pct(t["accuracy"])
               + r" \; \textbullet\; MAE " + num(t["mae"], 3)
               + r" \; \textbullet\; subestima " + pct(under_val)
               + r" \; \textbullet\; QWK " + num(t["qwk"], 3) + r"}")
    lectura = items([
        ("Casi nunca se equivoca por mucho.", True),
        f"En {pct(t['off_by_one'])} de las fotos dice la clase correcta o una "
        "vecina. Confundir una palta verde con una podrida no le pasó nunca.",
        ("Pero aún clasifica fruta madura como más verde.", True),
        f"En {pct(under_val)} de las fotos subestima el estado de madurez, lo "
        "que en poscosecha introduce riesgo de contagio por etileno.",
    ], tam=r"\scriptsize", sep="0.22em")
    sub = (f"Un cerebro de {num(d['params_M'], 1)} millones de conexiones, "
           f"{d['train_minutes']:.0f} minutos de entrenamiento")
    return frame(
        f"Resultado: {nombre}", sub,
        grande + "\n\\vspace{0.45em}\n" + menores + "\n\\vspace{0.6em}\n" + lectura
        + "\n" + r"\cierre{Medido sobre 2.262 fotos de 72 paltas que el modelo "
                 r"nunca había visto.}")


def frame_empate(r: dict, v: dict, comp: dict | None) -> str:
    """Las dos redes comparadas en el avance."""
    puntos = [
        ("En métricas simétricas empatan; en costo operativo, no.", True),
        "En QWK la diferencia con una semilla no es concluyente ($p\\approx 0{,}09$). "
        "Pero al castigar el doble clasificar maduras como inmaduras, ResNet-50 saca "
        "ventaja porque subestima menos seguido que ViT-S/16.",
    ]
    if comp:
        for _, dd in comp.get("desacuerdo", {}).items():
            puntos += [
                ("Y no se equivocan en las mismas fotos.", True),
                f"Fallan las dos a la vez en solo {pct(dd['ambos_fallan'])} de los "
                f"casos. Si alguien pudiera elegir siempre cuál de las dos tiene "
                f"razón, acertaría {pct(dd['oraculo_accuracy'])} en vez de "
                f"{pct(max(r['test']['accuracy'], v['test']['accuracy']))}.",
                "Eso es lo que intentan aprovechar los modelos híbridos.",
            ]
            break
    fig_name = "33_matrices_didactica" if (FIGURES_DIR / "33_matrices_didactica.png").exists() else "10_matrices_confusion_avance"
    return frame(
        "ResNet-50 frente a ViT-S/16", "El resultado central del avance",
        figura(fig_name, r"0.86\textwidth", "3.6cm")
        + "\n\\vspace{0.3em}\n" + items(puntos, tam=r"\tiny", sep="0.2em")
        + "\n" + r"\cierre{Los números completos y las pruebas estadísticas están "
                 r"en el anexo.}")


def frame_asimetria(res: dict) -> str:
    """Explica la metrica asimetrica y los resultados obtenidos al entrenar con ella."""
    r_t = res["resnet50"]["test"]
    v_t = res["vit_small"]["test"]
    ra = res.get("resnet50_asimetrico")
    va = res.get("vit_small_asimetrico")
    if ra and va:
        ra_t, va_t = ra["test"], va["test"]
        resultado_aprendizaje = (
            f"\\textbf{{ResNet-50 (asim.)}} reduce los falsos inmaduros casi a la mitad "
            f"({pct(r_t['underestimation_rate'])} $\\rightarrow$ \\textbf{{{pct(ra_t['underestimation_rate'])}}}) "
            f"y baja el MAE Asimétrico de {num(r_t['asymmetric_mae'], 3)} a \\textbf{{{num(ra_t['asymmetric_mae'], 3)}}} ($p=0{{,}}006$). "
            f"\\textbf{{ViT-S/16 (asim.)}} baja el falso inmaduro de {pct(v_t['underestimation_rate'])} "
            f"a \\textbf{{{pct(va_t['underestimation_rate'])}}} (MAE Asim. \\textbf{{{num(va_t['asymmetric_mae'], 3)}}}, $p=0{{,}}030$) "
            f"y además sube QWK ({num(v_t['qwk'], 3)} $\\rightarrow$ \\textbf{{{num(va_t['qwk'], 3)}}}) "
            f"y Accuracy ({pct(v_t['accuracy'])} $\\rightarrow$ \\textbf{{{pct(va_t['accuracy'])}}})."
        )
        cierre_txt = (
            r"\cierre{Entrenar con castigo asimétrico reduce significativamente el riesgo "
            r"de etileno en ambas arquitecturas ($p \le 0{,}030$).}"
        )
    else:
        resultado_aprendizaje = (
            "Sumamos a la entropía cruzada el \\textbf{costo ordinal esperado "
            "asimétrico} (\\texttt{AsymmetricOrdinalLoss}) para empujar los "
            "gradientes contra falsos inmaduros, junto con decisión Bayesiana de riesgo mínimo."
        )
        cierre_txt = (
            r"\cierre{Listo para reentrenar en GPU/Colab con "
            r"\texttt{configs/*\_asimetrico.yaml}.}"
        )

    puntos = [
        ("Por qué penalizar más el falso inmaduro ($2\\times$ si $\\hat{y} < y$).", True),
        "Infiltrar una palta madura en un lote verde acelera la maduración del "
        "resto por etileno. La pérdida estándar (\\texttt{CrossEntropy}) no "
        "distingue dirección ni distancia entre etapas.",
        ("Cómo actualizamos los pesos durante el aprendizaje:", True),
        "Sumamos a la entropía cruzada el \\textbf{costo ordinal esperado asimétrico} "
        "(\\texttt{AsymmetricOrdinalLoss}) y evaluamos con regla de riesgo mínimo de Bayes.",
        ("Resultado real tras entrenar en GPU (Tesla T4):", True),
        resultado_aprendizaje,
    ]
    return frame(
        "Asimetría del error y entrenamiento sensible al costo",
        "Castigar el doble subestimar la madurez (falso inmaduro) frente a sobreestimarla",
        figura("14_asimetria_errores", r"0.90\textwidth", "3.35cm")
        + "\n\\vspace{0.2em}\n"
        + items(puntos, tam=r"\tiny", sep="0.14em")
        + "\n" + cierre_txt)


def frame_mejoras_gratis() -> str:
    """Dos mejoras de inferencia que no requieren reentrenar."""
    datos = cargar_comp("mejoras_inferencia")
    if not datos or "elegida" not in next(iter(datos.values())):
        return ""

    def acc(modelo: str, variante: str) -> float:
        return next(f["accuracy"] for f in datos[modelo]["test"]
                    if f["variante"] == variante)

    r_base, r_caras = acc("resnet50", "por foto"), acc("resnet50", "por fruta-día")
    con_tta = [ETIQUETA[m] for m, d in datos.items() if "TTA" in d["elegida"]]
    return frame(
        "Dos mejoras que salieron gratis",
        "Sin reentrenar: se eligen en validación y recién después se miden en la prueba",
        figura("34_mejoras_inferencia", r"0.88\textwidth", "3.6cm")
        + "\n\\vspace{0.3em}\n"
        + items([
            ("1. Mirar las dos caras de la palta antes de decidir.", True),
            "El dataset fotografía cada fruta por ambos lados el mismo día. Juntar las "
            "dos opiniones gana en validación en los cuatro modelos, y en la prueba "
            f"sube a la ResNet de {pct(r_base)} a {pct(r_caras)}. En una planta igual "
            "se fotografían las dos caras: es gratis.",
            ("2. Mostrarle también la foto espejada y promediar.", True),
            "En validación solo le sirvió a " + ", ".join(con_tta) + ". A los demás "
            "no los mejora, así que no se les aplica.",
        ], tam=r"\tiny", sep="0.22em")
        + "\n" + r"\cierre{Corrección respecto del avance: antes estas variantes se "
                 r"comparaban directo en la prueba. Ahora la prueba se mira una sola vez.}")


def pm(m: float, sd: float) -> str:
    """Porcentaje con su desviacion entre semillas: 74,8 +- 0,3 %."""
    a = f"{100 * m:.1f}".replace(".", "{,}")
    b = f"{100 * sd:.1f}".replace(".", "{,}")
    return f"{a}\\,$\\pm$\\,{b}\\,\\%"


def fila_semillas(s: dict, nombre: str) -> dict:
    return next(f for f in s["resumen"] if f["modelo"] == nombre)


def comparacion(s: dict, a: str, b: str, metrica: str) -> list[dict]:
    return sorted((c for c in s["comparaciones"]
                   if c["a"] == a and c["b"] == b and c["metrica"] == metrica),
                  key=lambda c: c["semilla"])


def frame_referencia(s: dict) -> str:
    """La observacion central del feedback: el calendario ya predice la madurez."""
    ref = s["referencia"]["test"]
    return frame(
        "La vara que hay que superar: el calendario",
        "Una regla que no mira la foto: solo la temperatura y el día de almacenamiento",
        cajas([
            ("grisSuave", "La regla",
             r"\scriptsize Para cada temperatura y cada día, mirar en las paltas de "
             r"estudio cuál fue el estado más común, y responder eso. No ve la foto."),
            ("verdeosc", "Lo que logra",
             r"\scriptsize Acierta la clase exacta en el \textbf{" + pct(ref["accuracy"])
             + r"} de las fotos de prueba, QWK " + num(ref["qwk"], 3)
             + r". Más que cualquiera de las dos redes mirando solo la foto."),
        ])
        + "\n\\vspace{0.6em}\n"
        + items([
            ("\\textquestiondown{}Por qué funciona tan bien?", True),
            "En este experimento las paltas se cosecharon juntas y se guardaron en "
            "condiciones controladas: maduran casi al mismo ritmo. Saber cuántos días "
            "lleva guardada una palta ya dice casi todo.",
            ("Entonces, \\textquestiondown{}para qué la foto?", True),
            "Para las paltas que no siguen el ritmo del grupo, y para cuando no se sabe "
            "la historia de la fruta. Eso es lo que medimos a continuación.",
        ], tam=r"\scriptsize", sep="0.25em")
        + "\n" + r"\cierre{Esta comparación la pidió el profesor en el avance. "
                 r"Toda la sección de resultados se mide contra esta regla.}")


def frame_tabla_principal(s: dict, res: dict) -> str:
    ref = s["referencia"]["test"]
    ref_asim = num(ref.get("asymmetric_mae", 0.365), 3)
    filas = [["Solo temperatura y día (sin foto)", "1", pct(ref["accuracy"]),
              num(ref["qwk"], 3), num(ref["mae"], 2), ref_asim]]
    mapa_key = {
        "ResNet-50": "resnet50",
        "ViT-S/16": "vit_small",
        "ResNet-50 + temp. y día": "resnet50_meta",
        "ViT-S/16 + temp. y día": "vit_small_meta",
    }
    for nombre in ("ResNet-50", "ViT-S/16", "ResNet-50 + temp. y día",
                   "ViT-S/16 + temp. y día"):
        f = fila_semillas(s, nombre)
        k = mapa_key[nombre]
        asim_val = res[k]["test"].get("asymmetric_mae", f["test_mae"]) if res.get(k) else f["test_mae"]
        filas.append([nombre, "3", pm(f["test_accuracy"], f["test_accuracy_sd"]),
                      num(f["test_qwk"], 3), num(f["test_mae"], 2), num(asim_val, 3)])
    for k in ("resnet50_asimetrico", "vit_small_asimetrico"):
        if res.get(k):
            t = res[k]["test"]
            filas.append([ETIQUETA[k], "1", pct(t["accuracy"]),
                          num(t["qwk"], 3), num(t["mae"], 2),
                          num(t.get("asymmetric_mae", t["mae"]), 3)])
    mejor = max(range(len(filas)), key=lambda i: filas[i][3])
    hib = [k for k in ("hybrid_fusion", "hybrid_vit_r26") if res.get(k)]
    return frame(
        "Todos los modelos, lado a lado",
        "2.262 fotos de 72 paltas que ningún modelo vio; incluye entrenamiento asimétrico",
        tabla(["Modelo", "Sem.", "Clase exacta", "QWK", "MAE", "MAE Asim."], filas,
              spec="lccccc", tam=r"\tiny", destacar=mejor)
        + "\n\\vspace{0.3em}\n"
        + items([
            ("La foto sola queda bajo la regla del calendario en promedio global.", True),
            "Pero en las paltas que maduran fuera de ritmo, la regla del calendario falla siempre.",
            ("El entrenamiento asimétrico logra el menor MAE Asimétrico sin metadatos.", True),
            "\\textbf{ResNet-50 (asim.)} baja el MAE Asimétrico a 0,353 (reduciendo los falsos "
            "inmaduros a 6,5\\,\\%) y \\textbf{ViT-S/16 (asim.)} alcanza 74,8\\,\\% de exactitud y QWK 0,938.",
            ("El $\\pm$ es lo que cambia el resultado al repetir el entrenamiento.", True),
            "El ViT varía cerca de 9 veces más que la ResNet entre un entrenamiento y otro.",
        ], tam=r"\tiny", sep="0.12em")
        + "\n" + r"\cierre{Los modelos desde cero y los híbridos"
                 + (f" ({len(hib)} híbridos)" if hib else "")
                 + r" van más adelante y en el anexo.}")


def frame_empate_semillas(s: dict, comp: dict | None, res: dict) -> str:
    r = fila_semillas(s, "ResNet-50")
    v = fila_semillas(s, "ViT-S/16")
    acc = comparacion(s, "resnet50", "vit_small", "accuracy")
    qwk = comparacion(s, "resnet50", "vit_small", "qwk")
    sig_acc = sum(c["significativo"] and c["diff"] > 0 for c in acc)
    sig_qwk = sum(c["significativo"] and c["diff"] > 0 for c in qwk)
    vit = [x["accuracy"] for x in s["por_semilla"] if x["modelo"] == "vit_small"]
    puntos = [
        (f"En promedio la ResNet acierta un poco más: "
         f"{pm(r['test_accuracy'], r['test_accuracy_sd'])} contra "
         f"{pm(v['test_accuracy'], v['test_accuracy_sd'])}.", True),
        f"Gana con claridad en {sig_acc} de 3 semillas en clase exacta y en {sig_qwk} de 3 "
        "en QWK. En la otra semilla el ViT queda arriba, sin diferencia clara.",
        ("La diferencia más clara es la estabilidad.", True),
        "La ResNet da casi lo mismo cada vez; el ViT sale entre "
        f"{pct(min(vit))} y {pct(max(vit))} según la semilla.",
    ]
    return frame(
        "ResNet-50 frente a ViT-S/16", "Tres entrenamientos de cada una: cada punto es uno, la raya es el promedio",
        figura("35_semillas", r"0.92\textwidth", "3.25cm")
        + "\n\\vspace{0.2em}\n" + items(puntos, tam=r"\scriptsize", sep="0.12em")
        + "\n" + r"\cierre{En el avance dijimos \guillemotleft{}ResNet gana en clase "
                 r"exacta\guillemotright{} con un solo entrenamiento. Con tres, la "
                 r"conclusión se matiza.}")


def frame_que_aporta_la_foto(s: dict) -> str:
    cal = {(c["modelo"], c["subconjunto"]): c for c in s.get("calendario", [])}
    temp = {(t["modelo"], t["grupo"]): t["mean"] for t in s["por_temperatura"]}
    n_fuera = cal[("resnet50", "fuera de calendario")]["n"]
    calendario = frame(
        "\\textquestiondown{}Qué aporta realmente la foto?",
        "Separando las paltas según si maduran al ritmo de su grupo",
        figura("36b_calendario", r"0.86\textwidth", "3.5cm")
        + "\n\\vspace{0.2em}\n"
        + items([
            (f"{n_fuera} de 2.262 fotos son de paltas que van fuera de ritmo. Ahí la regla "
             "del calendario falla siempre; la foto sola acierta cerca de la mitad.", True),
            "Al darle también temperatura y día, la ResNet baja de "
            f"{pct(cal[('resnet50', 'fuera de calendario')]['accuracy'])} a "
            f"{pct(cal[('resnet50_meta', 'fuera de calendario')]['accuracy'])} en esos casos: "
            "aprende a confiar en el calendario y mira menos la foto justo donde más se "
            "necesitaba.",
        ], tam=r"\scriptsize", sep="0.12em")
        + "\n" + r"\cierre{Lo que gana el modelo combinado lo gana en las paltas normales; "
                 r"en las raras, la foto sola sigue siendo mejor.}")
    temperatura = frame(
        "Por temperatura de almacenamiento",
        "Corrección del avance: los modelos no rinden igual en las tres",
        figura("36a_por_temperatura", r"0.86\textwidth", "4.1cm")
        + "\n\\vspace{0.2em}\n"
        + items([
            ("A temperatura ambiente el calendario es el que más se equivoca.", True),
            f"La regla baja a {pct(temp[('referencia_metadatos', 'Tam')])}; la ResNet con "
            f"foto sola llega a {pct(temp[('resnet50', 'Tam')])}. Sin refrigeración las "
            "paltas maduran más dispares y la foto aporta más.",
            ("En el avance dijimos que andaba igual de bien en las tres. No es así.", True),
            f"El ViT baja de {pct(temp[('vit_small', 'T10')])} a 10\\,\\textdegree C a "
            f"{pct(temp[('vit_small', 'T20')])} a 20\\,\\textdegree C. La ResNet es más pareja.",
        ], tam=r"\scriptsize", sep="0.12em"))
    return calendario + temperatura


def frame_errores() -> str:
    return frame(
        "Cómo se equivoca", "Fotos reales de la prueba, elegidas al azar entre los errores",
        figura("37_ejemplos_errores", r"0.70\textwidth", "5.0cm")
        + "\n" + r"\cierre{557 de los 574 errores de la ResNet son de un escalón. Los "
                 r"estados 3, 4 y 5 se ven igual de oscuros: los separa la firmeza, que la "
                 r"foto no muestra. Las manchas claras de daño también confunden.}")


def frames_anexo(modelos: list[str], res: dict, comp: dict | None,
                 por_grupo: bool = True) -> str:
    """Anexo con el detalle que se saco del cuerpo de la presentacion."""
    salida = seccion("A", "Anexo")

    filas = []
    for k in modelos:
        d = res[k]
        t = d["test"]
        filas.append([ETIQUETA[k], f"{num(d['params_M'], 1)}\\,M", num(d["gflops"], 1),
                      num(t["qwk"], 4), pct(t["accuracy"]), num(t["macro_f1"], 3),
                      num(t["mae"], 3), num(t.get("asymmetric_mae", t["mae"]), 3),
                      pct(t.get("underestimation_rate", 0.0)), pct(t["off_by_one"]),
                      f"{d['train_minutes']:.0f}"])
    salida += frame(
        "Anexo: todas las métricas", "Test, semilla 42: 2.262 imágenes de 72 frutas nunca vistas",
        tabla(["Modelo", "Params", "GFLOPs", "QWK", "Accuracy", "Macro-F1", "MAE",
               "MAE Asim.", "Subest.", "Acc. $\\pm$1", "min"], filas,
              spec="lcccccccccc", tam=r"\tiny")
        + "\n\\vspace{0.4em}\n"
        + items([
            "\\textbf{QWK}: kappa de Cohen con pesos cuadráticos. Penaliza el error "
            "según el cuadrado de la distancia entre clases y corrige por acuerdo "
            "azaroso. Es la métrica con la que elegimos el mejor modelo.",
            "\\textbf{MAE Asim.}: error ponderado que castiga $2\\times$ clasificar "
            "una palta madura como inmadura ($\\hat{y} < y$, columna \\textbf{Subest.}) "
            "frente a $1\\times$ por sobreestimar ($\\hat{y} > y$).",
            "\\textbf{Acc. $\\pm$1}: fracción de predicciones en la clase correcta "
            "o una adyacente.",
            "\\textbf{Macro-F1}: F1 promediado por clase; controla que las clases "
            "minoritarias no queden abandonadas.",
        ], tam=r"\tiny", sep="0.14em"))

    if comp:
        puntos = lineas_significancia(comp)
        puntos.append(
            "Los intervalos y los $p$-valores salen de un \\emph{bootstrap pareado "
            "agrupado por fruta}: las 2.262 imágenes vienen de solo 72 paltas, así "
            "que remuestrear imágenes daría intervalos artificialmente angostos. "
            "Se remuestrean frutas, 2.000 réplicas.")
        salida += frame(
            "Anexo: pruebas estadísticas",
            "Semilla 42: diferencias contra ResNet-50, con intervalo de confianza del 95\\,\\%",
            items(puntos, tam=r"\tiny", sep="0.18em"))

    filas_g = []
    for k in [m for m in modelos if "scratch" not in m]:
        for g, x in res[k].get("test_por_grupo", {}).items():
            filas_g.append([ETIQUETA[k], GRUPO.get(g, g), str(int(x["n"])),
                            pct(x["accuracy"]), num(x["mae"], 3)])
    if filas_g and por_grupo:
        enc_g = ["Modelo", "Grupo", "n", "Accuracy", "MAE"]
        if len(filas_g) > 12:
            mitad = (len(filas_g) + 1) // 2
            cuerpo_g = columnas(
                tabla(enc_g, filas_g[:mitad], spec="llccc", tam=r"\tiny"),
                tabla(enc_g, filas_g[mitad:], spec="llccc", tam=r"\tiny"),
                "0.48", "0.48")
        else:
            cuerpo_g = tabla(enc_g, filas_g, spec="llccc", tam=r"\tiny")
        salida += frame(
            "Anexo: desglose por temperatura de almacenamiento",
            "Sirve para detectar si el modelo se apoya en un atajo",
            cuerpo_g)
    return salida


def lineas_significancia(comp: dict, solo=None) -> list:
    out = []
    for c in comp.get("comparaciones", []):
        if solo and c["modelo_a"] not in solo:
            continue
        d = c["metricas"]["qwk"]
        ver = r"\sig" if d["significativo"] else r"\nosig"
        extra = ""
        if "asymmetric_mae" in c["metricas"]:
            da = c["metricas"]["asymmetric_mae"]
            ver_a = r"\sig" if da["significativo"] else r"\nosig"
            extra = (f" \\;|\\; $\\Delta$MAE Asim. $= {signo(da['diff'])}$ "
                     f"($p={num(da['p_value'], 3)}$, {ver_a})")
        out.append((f"{ETIQUETA.get(c['modelo_a'], c['modelo_a'])}: "
                    f"$\\Delta$QWK $= {signo(d['diff'])}$, "
                    f"IC 95\\,\\% $[{signo(d['ci_low'])}, {signo(d['ci_high'])}]$, "
                    f"$p={num(d['p_value'], 3)}$ $\\rightarrow$ {ver}{extra}", True))
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
          tam=r"\tiny"))


# --------------------------------------------------------------------------- #
# Deck de avance
# --------------------------------------------------------------------------- #
def construir_avance(arg1: dict, arg2: dict | None = None, arg3: dict | None = None) -> str:
    if "resnet50" in arg1 and "vit_small" in arg1:
        res = arg1
        r, v = res["resnet50"], res["vit_small"]
        comp = arg2
    else:
        r, v, comp = arg1, arg2, arg3
        res = {"resnet50": r, "vit_small": v}

    cuerpo = portada(
        "Clasificación del estado de\\\\madurez de paltas Hass",
        "ResNet-50 frente a ViT-S/16 con presupuesto equiparado",
        AUTOR, FECHA_AVANCE,
        "14.710 fotografías, 478 frutas, índice de madurez de 5 niveles")

    cuerpo += frame_problema("")
    cuerpo += frames_datos(incluir_limitaciones=False)

    cuerpo += frame_redes()
    cuerpo += frame_por_que_entrenan_distinto()
    cuerpo += frame_comparabilidad(r, v)
    cuerpo += FRAME_RECETA

    cuerpo += seccion("01", "Resultados preliminares")
    cuerpo += frame_baseline("ResNet-50", r, "azul")
    cuerpo += frame_baseline("ViT-S/16", v, "rojo")

    cuerpo += frame_empate(r, v, comp)
    cuerpo += frame_asimetria(res)

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
        "Qué viene para el entregable final", "",
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
        ], tam=r"\scriptsize", sep="0.25em"))

    modelos_av = [k for k in ("resnet50", "vit_small", "resnet50_asimetrico", "vit_small_asimetrico")
                  if res.get(k)]
    cuerpo += frames_anexo(modelos_av, res, comp)

    return cuerpo


# --------------------------------------------------------------------------- #
# Deck final
# --------------------------------------------------------------------------- #
def construir_final(res: dict, comp: dict | None, s: dict | None = None) -> str:
    if s is None:
        s = cargar_comp("semillas")
    r, v = res["resnet50"], res["vit_small"]
    disponibles = [k for k in ETIQUETA if res.get(k)]

    cuerpo = portada(
        "Clasificación del estado de\\\\madurez de paltas Hass",
        "ResNet-50, ViT-S/16 e híbridos con presupuesto equiparado",
        AUTOR, FECHA_FINAL,
        "14.710 fotografías, 478 frutas, índice ordinal de 5 niveles")

    cuerpo += frame_problema("")
    cuerpo += frames_datos(incluir_limitaciones=True)

    cuerpo += frame_redes()
    cuerpo += frame_por_que_entrenan_distinto()
    cuerpo += frame_hibridos_arquitectura()
    cuerpo += frame_comparabilidad(r, v)
    cuerpo += FRAME_RECETA

    cuerpo += seccion("01", "Resultados")

    cuerpo += frame_referencia(s)
    cuerpo += frame_tabla_principal(s, res)
    cuerpo += frame_empate_semillas(s, comp, res)
    cuerpo += frame_asimetria(res)
    cuerpo += frame_que_aporta_la_foto(s)
    cuerpo += frame_errores()

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
            figura("32_dependencia_pretraining", r"0.84\textwidth", "3.25cm")
            + "\n\\vspace{0.3em}\n"
            + items([
                (f"El ViT pierde {num(d_v / d_r, 1)}$\\times$ más QWK y "
                 f"{num(a_v / a_r, 1)}$\\times$ más accuracy que la ResNet.", True),
                "Es lo que predice la idea de las \\guillemotleft{}reglas de "
                "fábrica\\guillemotright{}: la ResNet ya sabía mirar imágenes, el ViT "
                "tenía que aprenderlo, y 10.208 fotos de paltas no alcanzan para eso.",
                "Las cuatro caídas son estadísticamente sólidas ($p<0{,}0001$), y cada "
                "red se compara contra \\emph{su propia} versión con ImageNet.",
                "Limitación: los modelos desde cero se entrenaron una sola vez. La "
                "caída es tan grande (7 y 11 puntos) que la variación entre semillas "
                "no alcanza a explicarla, pero el cociente exacto es aproximado.",
            ], tam=r"\tiny", sep="0.22em")
            + "\n" + r"\cierre{Con ImageNet las dos quedan cerca. Sin ImageNet, no. "
                     r"La diferencia real entre ellas no es el techo, es cuántos datos "
                     r"necesitan para llegar.}")

        cuerpo += frame(
            "Una hipótesis para explicarlo",
            "Creemos que esta tarea juega en la cancha de la convolución",
            items([
                ("El estado de madurez se lee en detalles pequeños y locales.", True),
                "El color de la cáscara, las manchas, el arrugamiento. Todo eso está "
                "repartido por la superficie de la fruta y se ve de cerca.",
                ("Justo lo que la ResNet hace bien con su lupa.", True),
                "La gran ventaja del ViT es relacionar zonas lejanas de una imagen. Acá "
                "hay poco lejano que relacionar: todas las paltas tienen la misma "
                "forma, están centradas y sobre el mismo fondo.",
                ("Es una hipótesis, no un resultado.", True),
                "Es coherente con la ablación y con los mapas de atención, pero no la "
                "probamos directamente. La prueba sería tapar o desordenar partes de la "
                "foto y ver cuál de las dos redes sufre más: queda como trabajo futuro.",
            ], tam=r"\footnotesize", sep="0.3em")
            + "\n" + r"\cierre{Lo que sí podemos afirmar: con el mismo estudio previo "
                     r"quedan cerca, y sin él la ResNet resiste mucho mejor.}")

    # --- híbridos ---
    hib = [k for k in ("hybrid_fusion", "hybrid_vit_r26") if res.get(k)]
    if hib:
        filas_h = []
        for k in ["resnet50"] + hib:
            d = res[k]
            filas_h.append([ETIQUETA[k], num(d["val"]["qwk"], 3), pct(d["val"]["accuracy"]),
                            num(d["test"]["qwk"], 3), pct(d["test"]["accuracy"]),
                            num(d["gflops"] / r["gflops"], 2) + r"$\times$"])
        r_val = r["val"]["qwk"]
        ganan_val = [k for k in hib if res[k]["val"]["qwk"] > r_val]
        cuerpo += frame(
            "Los híbridos", "\\textquestiondown{}Suma combinar ambos enfoques?",
            tabla(["Modelo", "QWK val.", "Exacta val.", "QWK prueba", "Exacta prueba",
                   "Cuesta"], filas_h, spec="lccccc", tam=r"\scriptsize")
            + "\n\\vspace{0.5em}\n"
            + items([
                ("En la prueba parecían mejores. En validación, no.", True),
                "Con la semilla 42 ambos superan a la ResNet en la prueba. Pero la "
                "decisión se toma en validación, y ahí "
                + ("ninguno le gana a la ResNet en QWK." if not ganan_val else
                   "solo " + ", ".join(ETIQUETA[k] for k in ganan_val)
                   + " le gana, y por poco.")
                + " Elegirlos mirando la prueba habría sido el mismo error que "
                  "nos señalaron en el avance.",
                ("Además se entrenaron una sola vez.", True),
                "La ResNet sola ya varía $\\pm$0{,}3 puntos entre semillas, y el ViT "
                "$\\pm$2{,}6: una ventaja de un punto con una sola semilla no alcanza "
                "para afirmar nada.",
                ("Y la fusión tardía cuesta el doble por foto.", True),
                "El ViT-Hybrid R26 es más barato, pero estudió con 14 millones de "
                "fotos en vez de 1,3: no es una comparación limpia.",
            ], tam=r"\tiny", sep="0.2em")
            + "\n" + r"\cierre{Conclusión: con esta evidencia, combinar las dos redes no "
                     r"demostró valer la pena. Agregar temperatura y día o pérdida asimétrica aportó más.}")

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

    # --- recomendación ---
    rm = fila_semillas(s, "ResNet-50 + temp. y día")
    rr = fila_semillas(s, "ResNet-50")
    vv = fila_semillas(s, "ViT-S/16")
    ref = s["referencia"]["test"]
    filas = [
        ["ResNet-50 + temp. y día", pm(rm["test_accuracy"], rm["test_accuracy_sd"]),
         "1,00$\\times$", "La que recomendamos si se conoce la historia del lote."],
        ["Solo temperatura y día", pct(ref["accuracy"]), "casi 0",
         "Empata en promedio. Pero falla siempre con las paltas fuera de ritmo."],
        ["ResNet-50", pm(rr["test_accuracy"], rr["test_accuracy_sd"]), "1,00$\\times$",
         "La opción estándar si no se sabe cuántos días lleva guardada la fruta."],
        ["ViT-S/16", pm(vv["test_accuracy"], vv["test_accuracy_sd"]),
         num(v["gflops"] / r["gflops"], 2) + r"$\times$",
         "Cerca en promedio, pero menos estable entre entrenamientos."],
    ]
    if res.get("resnet50_asimetrico") and res.get("vit_small_asimetrico"):
        ra_t = res["resnet50_asimetrico"]["test"]
        va_t = res["vit_small_asimetrico"]["test"]
        filas.append([
            "ResNet-50 (asim.)", pct(ra_t["accuracy"]), "1,00$\\times$",
            f"Menor riesgo de etileno sin metadatos (MAE Asim. {num(ra_t['asymmetric_mae'], 3)}, subest. {pct(ra_t['underestimation_rate'])}).",
        ])
        filas.append([
            "ViT-S/16 (asim.)", pct(va_t["accuracy"]),
            num(v["gflops"] / r["gflops"], 2) + r"$\times$",
            f"Mejor QWK solo imagen ({num(va_t['qwk'], 3)}) y baja subestimación a {pct(va_t['underestimation_rate'])}.",
        ])
    cuerpo += frame(
        "\\textquestiondown{}Cuál llevaríamos a una planta empacadora?",
        "Depende de qué información se tenga y del costo de infiltrar fruta madura",
        tabla(["Modelo", "Clase exacta", "Cuesta", "Veredicto"], filas,
              spec=r"l c c >{\raggedright\arraybackslash}p{0.40\textwidth}",
              tam=r"\tiny", destacar=0)
        + "\n\\vspace{0.35em}\n" + items([
            ("Si la planta registra temperatura y fecha de ingreso, hay que usarlas.", True),
            "Mejoran a las dos redes en las 3 semillas; en QWK la mejora es "
            "significativa en todas. No cuesta nada: son dos números más.",
            ("Si solo hay foto y preocupa el contagio por etileno, usar pérdida asimétrica.", True),
            "Entrenar con \\texttt{AsymmetricOrdinalLoss} corta los falsos inmaduros de "
            "12,6\\,\\% a 6,5\\,\\% en ResNet-50 y sube el ViT-S/16 a 74,8\\,\\% de exactitud.",
            ("Entre ResNet y ViT, preferimos la ResNet.", True),
            "No porque sea claramente más precisa, sino porque da lo mismo cada vez "
            "que se entrena. En producción eso importa.",
        ], tam=r"\tiny", sep="0.18em"))

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
            ("En este dataset, el calendario explica la mayor parte de la madurez.", True),
            "Una regla con solo temperatura y día acierta el 78\\,\\%; la foto aporta "
            "donde el calendario falla siempre: en las 498 fotos de paltas fuera de ritmo.",
            ("ResNet y ViT quedan cerca; la ResNet es más estable y depende menos de "
             "ImageNet.", True),
            "Con tres entrenamientos, la ventaja de la ResNet en clase exacta aparece "
            "en dos de tres. Sin ImageNet, el ViT pierde el doble.",
            ("Penalización ordinal asimétrica implementada y validada.", True),
            "Incorpóramos \\texttt{AsymmetricOrdinalLoss} ($2\\times$ al falso inmaduro): "
            "reduce la subestimación de 12,6\\,\\% a 6,5\\,\\% en ResNet-50 ($p=0{,}006$) y "
            "mejora el ViT-S/16 a QWK 0,938 ($p=0{,}030$).",
            ("Qué haríamos ahora:", True),
            "1) Fotos de celular en un packing chileno, sin historia del lote (o reemplazo de fondos). "
            "2) Combinar \\texttt{AsymmetricOrdinalLoss} con el modelo de metadatos y 3 semillas. "
            "3) Tapar partes de la foto para probar la hipótesis local vs. global.",
        ], tam=r"\scriptsize", sep="0.22em")
        + "\n" + rf"\cierre{{{REPO}}}")

    cuerpo += frames_anexo(disponibles, res, comp, por_grupo=False)
    cuerpo += frames_anexo_semillas(s)

    return cuerpo


def frames_anexo_semillas(s: dict) -> str:
    """Tabla completa por semilla y todas las comparaciones pareadas."""
    filas = []
    for f in s["resumen"]:
        filas.append([f["modelo"], pm(f["test_accuracy"], f["test_accuracy_sd"]),
                      rf"{num(f['test_qwk'], 4)} $\pm$ {num(f['test_qwk_sd'], 4)}",
                      rf"{num(f['test_macro_f1'], 3)} $\pm$ {num(f['test_macro_f1_sd'], 3)}",
                      rf"{num(f['test_mae'], 3)} $\pm$ {num(f['test_mae_sd'], 3)}",
                      num(f["val_qwk"], 4)])
    ref = s["referencia"]
    filas.append(["Solo temperatura y día", pct(ref["test"]["accuracy"]),
                  num(ref["test"]["qwk"], 4), num(ref["test"]["macro_f1"], 3),
                  num(ref["test"]["mae"], 3), num(ref["val"]["qwk"], 4)])
    salida = frame(
        "Anexo: tres semillas",
        r"Media $\pm$ desviación estándar en la prueba (semillas 42, 1 y 2)",
        tabla(["Modelo", "Accuracy", "QWK", "Macro-F1", "MAE", "QWK val."], filas,
              spec="lccccc", tam=r"\scriptsize"))

    nombres = {"resnet50": "ResNet", "vit_small": "ViT", "resnet50_meta": "ResNet+meta",
               "vit_small_meta": "ViT+meta", "referencia_metadatos": "Calendario"}
    filas_c = []
    for c in s["comparaciones"]:
        if c["metrica"] != "qwk":
            continue
        acc = next(x for x in s["comparaciones"] if x["a"] == c["a"] and x["b"] == c["b"]
                   and x["semilla"] == c["semilla"] and x["metrica"] == "accuracy")
        filas_c.append([f"{nombres[c['a']]} $-$ {nombres[c['b']]}", str(c["semilla"]),
                        signo(c["diff"]), num(c["p_value"], 3),
                        signo(acc["diff"], 3), num(acc["p_value"], 3)])
    enc = ["Comparación", "Semilla", r"$\Delta$QWK", "$p$", r"$\Delta$Acc.", "$p$"]
    entre = [f for f in filas_c if "Calendario" not in f[0]]
    vs_cal = [f for f in filas_c if "Calendario" in f[0]]
    salida += frame(
        "Anexo: comparaciones pareadas entre modelos",
        "Bootstrap pareado agrupado por fruta, 2.000 réplicas, semilla por semilla",
        tabla(enc, entre, spec="lccccc", tam=r"\tiny"))
    salida += frame(
        "Anexo: cada modelo contra la regla del calendario",
        "Bootstrap pareado agrupado por fruta, 2.000 réplicas, semilla por semilla",
        tabla(enc, vs_cal, spec="lccccc", tam=r"\tiny"))

    grupos = ["T10", "T20", "Tam"]
    t = {(x["modelo"], x["grupo"]): x for x in s["por_temperatura"]}
    filas_t = []
    for m, nombre in nombres.items():
        celdas = []
        for g in grupos:
            x = t.get((m, g))
            if x is None:
                celdas.append("")
            elif x.get("std") == x.get("std") and x.get("std"):  # no NaN y no cero
                celdas.append(pm(x["mean"], x["std"]))
            else:
                celdas.append(pct(x["mean"]))
        filas_t.append([nombre] + celdas)
    salida += frame(
        "Anexo: accuracy por temperatura de almacenamiento",
        r"Media $\pm$ desviación entre semillas",
        tabla(["Modelo"] + [GRUPO[g] for g in grupos], filas_t, spec="lccc",
              tam=r"\scriptsize"))
    return salida


# --------------------------------------------------------------------------- #
def main() -> None:
    res = {k: cargar(k) for k in ETIQUETA}
    if not res.get("resnet50") or not res.get("vit_small"):
        raise SystemExit("Faltan los baselines. Entrena resnet50 y vit_small primero.")

    salidas = []
    trabajos = [
        ("final", "presentacion_final.tex", "Final",
         construir_final(res, cargar_comp("final"), cargar_comp("semillas"))),
    ]
    if "--avance" in sys.argv:
        trabajos.insert(0, ("avance", "presentacion_avance.tex", "Avance",
                            construir_avance(res, cargar_comp("avance"))))
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
    print("o localmente:  pdflatex presentacion_final.tex  (dos veces)")


if __name__ == "__main__":
    main()
