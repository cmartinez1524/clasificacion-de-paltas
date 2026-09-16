"""Genera entregables/avance/presentacion_avance.pptx desde las metricas reales.

Uso:
    python scripts/05_deck_avance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pptx.util import Inches  # noqa: E402

from paltas.deck import (  # noqa: E402
    AZUL,
    GRIS,
    MORADO,
    ROJO,
    VERDE,
    imagen,
    imagen_centrada,
    nota,
    nueva_presentacion,
    numerar,
    slide_portada,
    slide_seccion,
    slide_titulo,
    tabla,
    tarjeta,
    vinetas,
)
from paltas.paths import FIGURES_DIR, METRICS_DIR, ROOT  # noqa: E402
from paltas.console import use_utf8  # noqa: E402

use_utf8()

SALIDA = ROOT / "entregables" / "avance" / "presentacion_avance.pptx"
AUTOR = "Cristóbal Martínez"
FECHA = "Avance · 28 de septiembre de 2026"


def cargar(nombre: str) -> dict:
    p = METRICS_DIR / f"{nombre}_test.json"
    if not p.exists():
        raise SystemExit(f"Falta {p}. Entrena primero el modelo {nombre}.")
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def main() -> None:
    r = cargar("resnet50")
    v = cargar("vit_small")
    comp_path = METRICS_DIR / "avance.json"
    comp = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else None

    prs = nueva_presentacion()

    # ------------------------------------------------------------------ 1
    slide_portada(
        prs,
        "Clasificación del estado de\nmadurez de paltas Hass",
        "ResNet-50 frente a ViT-S/16 con presupuesto equiparado",
        AUTOR, FECHA,
        "14.710 fotografías · 478 frutas · índice de madurez de 5 niveles",
    )

    # ------------------------------------------------------------------ 2
    s, y = slide_titulo(prs, "El problema", "Por qué importa acá y no solo en el paper")
    vinetas(s, Inches(0.7), y, Inches(6.3), [
        ("Chile exporta paltas Hass con una ventana de madurez muy estrecha.", True),
        "La clasificación se hace hoy a ojo y al tacto, operario por operario.",
        "Dos costos simétricos: fruta verde en góndola (el consumidor la descarta) "
        "y fruta pasada en tránsito (pérdida total).",
        ("La tarea es ordinal, no categórica: 1 < 2 < 3 < 4 < 5.", True),
        "Confundir un estado 1 con un 2 es un error de un día. Confundir un 1 con "
        "un 5 es mandar fruta podrida a la venta.",
    ])
    tabla(s, Inches(7.4), y + Inches(0.15), Inches(5.2), Inches(2.6), [
        ["Índice", "Estado", "Decisión"],
        ["1", "Verde", "Dejar madurar"],
        ["2", "Iniciando", "Lista en 2-3 días"],
        ["3", "Maduro I", "Venta inmediata"],
        ["4", "Maduro II", "Último día"],
        ["5", "Sobremaduro", "Fuera de punto"],
    ], anchos=[1, 2, 3], size=12)
    nota(s, "La métrica tiene que castigar más los errores lejanos. La accuracy plana no lo hace.")

    # ------------------------------------------------------------------ 3
    s, y = slide_titulo(prs, "Los datos",
                        "Hass Avocado Ripening Photographic Dataset · Mendeley, CC BY 4.0")
    for i, (t, val, det, c) in enumerate([
        ("IMÁGENES", "14.710", "800×800 px, JPEG", VERDE),
        ("FRUTAS", "478", "seguidas hasta 26 días", AZUL),
        ("CLASES", "5", "índice de madurez", MORADO),
        ("GRUPOS", "3", "10 °C · 20 °C · ambiente", ROJO),
    ]):
        tarjeta(s, Inches(0.7 + i * 3.08), y, Inches(2.86), Inches(1.5), t, val, det, c)

    imagen_centrada(s, FIGURES_DIR / "01_distribucion_clases.png",
                    y + Inches(1.75), Inches(11.4), Inches(3.05))
    nota(s, "Xavier, Rodrigues & Silva (2024) · DOI 10.17632/3xd9n945v8.1 · "
            "Etiquetado visual experto, sin acuerdo inter-evaluador reportado.")

    # ------------------------------------------------------------------ 4
    s, y = slide_titulo(prs, "El riesgo que define el proyecto",
                        "Fuga de datos longitudinal")
    vinetas(s, Inches(0.7), y, Inches(6.2), [
        ("La misma palta fue fotografiada a diario durante hasta 26 días.", True),
        "Dos fotos consecutivas de la fruta #173 son casi idénticas: misma piel, "
        "mismas manchas, misma forma.",
        ("Un split aleatorio por imagen pondría el día 5 en train y el día 6 en test.", True),
        "El modelo reconocería la fruta, no el estado de madurez. La métrica "
        "sería alta y completamente falsa.",
        "Hay 6.871 pares de días consecutivos de la misma fruta en el dataset.",
        ("Partimos las 478 frutas, no las 14.710 imágenes.", True),
    ], size=15)
    imagen(s, FIGURES_DIR / "04_particiones.png", Inches(7.25), y + Inches(1.0),
           Inches(5.4), Inches(2.4))
    nota(s, "Estratificado por grupo de almacenamiento · assert en el código que falla "
            "si una fruta aparece en dos particiones.")

    # ------------------------------------------------------------------ 5
    s, y = slide_titulo(prs, "Diseño de la comparación",
                        "Comparar sin igualar el presupuesto no dice nada")
    tabla(s, Inches(0.7), y, Inches(11.9), Inches(2.0), [
        ["Eje igualado", "ResNet-50", "ViT-S/16 (DeiT-S)", "Diferencia"],
        ["Parámetros", f"{r['params_M']:.2f} M", f"{v['params_M']:.2f} M",
         f"{abs(r['params_M']-v['params_M'])/max(r['params_M'],v['params_M'])*100:.1f} %"],
        ["GFLOPs @ 224 px", f"{r['gflops']:.2f}", f"{v['gflops']:.2f}",
         f"{abs(r['gflops']-v['gflops'])/max(r['gflops'],v['gflops'])*100:.1f} %"],
        ["Pre-entrenamiento", "ImageNet-1k", "ImageNet-1k", "ninguna"],
    ], anchos=[3, 2, 2.4, 1.6], size=14)
    vinetas(s, Inches(0.7), y + Inches(2.35), Inches(11.9), [
        ("El tercer eje es el que casi nadie iguala y el que más distorsiona.", True),
        "Los pesos ViT-S más usados de timm vienen de ImageNet-21k: catorce veces "
        "más datos que los de la ResNet. Con esos pesos la comparación mediría el "
        "corpus de pre-entrenamiento, no la arquitectura.",
        "Usamos DeiT-S: misma arquitectura ViT-S/16, entrenado solo en ImageNet-1k.",
        "Receta idéntica. Solo cambia lo que tiene que cambiar: el ViT usa LR 3× "
        "menor, warmup más largo y drop_path 0,1, porque diverge con la receta de la CNN.",
    ], size=15)
    nota(s, "Igualar el learning rate «por justicia» no sería más justo: solo sería peor para el ViT.")

    # ------------------------------------------------------------------ 6
    slide_seccion(prs, "01", "Resultados preliminares")

    # ------------------------------------------------------------------ 7  ResNet
    for nombre, datos, color in [("ResNet-50", r, AZUL), ("ViT-S/16", v, ROJO)]:
        s, y = slide_titulo(prs, f"Baseline {nombre}",
                            f"{datos['params_M']:.2f} M parámetros · "
                            f"{datos['gflops']:.2f} GFLOPs · "
                            f"{datos['train_minutes']:.0f} min de entrenamiento · "
                            f"mejor época {datos['best_epoch']}")
        t = datos["test"]
        for i, (lbl, val, det) in enumerate([
            ("QWK (principal)", f"{t['qwk']:.3f}", "kappa cuadrático"),
            ("ACCURACY", pct(t["accuracy"]), "5 clases exactas"),
            ("ACC. ±1 CLASE", pct(t["off_by_one"]), "utilidad operativa"),
            ("MAE", f"{t['mae']:.3f}", "escalones de madurez"),
            ("MACRO-F1", f"{t['macro_f1']:.3f}", "clases minoritarias"),
        ]):
            tarjeta(s, Inches(0.7 + i * 2.44), y, Inches(2.26), Inches(1.55), lbl, val, det, color)

        grupos = datos.get("test_por_grupo", {})
        filas = [["Grupo de almacenamiento", "n", "Accuracy", "MAE"]]
        etiquetas = {"T10": "T10 · 10 °C", "T20": "T20 · 20 °C", "Tam": "Tamb · ambiente"}
        for g, d in grupos.items():
            filas.append([etiquetas.get(g, g), f"{int(d['n'])}",
                          pct(d["accuracy"]), f"{d['mae']:.3f}"])
        tabla(s, Inches(0.7), y + Inches(1.95), Inches(5.6), Inches(1.5), filas,
              anchos=[3, 1, 1.5, 1.2], size=12)

        vinetas(s, Inches(6.8), y + Inches(1.95), Inches(5.8), [
            f"Los errores se concentran en clases adyacentes: {pct(t['off_by_one'])} "
            f"cae en la clase correcta o en una vecina.",
            f"MAE de {t['mae']:.2f} escalones sobre una escala de 5.",
            "El desempeño es homogéneo entre grupos de almacenamiento: el modelo "
            "no está explotando el sesgo «verde ⇒ refrigerada».",
        ], size=14)
        nota(s, f"Evaluado sobre 2.262 imágenes de 72 frutas que ningún modelo vio "
                f"durante el entrenamiento.")

    # ------------------------------------------------------------------ 9  Comparación
    s, y = slide_titulo(prs, "ResNet-50 vs ViT-S/16",
                        "Intervalos por bootstrap agrupado por fruta, no por imagen")
    filas = [["Modelo", "Params", "GFLOPs", "QWK", "Accuracy", "Macro-F1", "MAE", "Acc. ±1"]]
    for nom, d in [("ResNet-50", r), ("ViT-S/16", v)]:
        t = d["test"]
        filas.append([nom, f"{d['params_M']:.1f} M", f"{d['gflops']:.1f}",
                      f"{t['qwk']:.3f}", pct(t["accuracy"]), f"{t['macro_f1']:.3f}",
                      f"{t['mae']:.3f}", pct(t["off_by_one"])])
    mejor = 1 if r["test"]["qwk"] >= v["test"]["qwk"] else 2
    tabla(s, Inches(0.7), y, Inches(11.9), Inches(1.3), filas,
          anchos=[2.2, 1.2, 1.2, 1.2, 1.3, 1.3, 1, 1.2], size=13, destacar_fila=mejor)

    puntos = []
    if comp:
        for c in comp.get("comparaciones", []):
            d = c["metricas"]["qwk"]
            signo = "a favor del ViT" if d["diff"] > 0 else "a favor de la ResNet"
            puntos.append((
                f"Δ QWK = {d['diff']:+.4f}  ·  IC 95 % [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
                f"·  p = {d['p_value']:.3f} → "
                f"{'diferencia SIGNIFICATIVA ' + signo if d['significativo'] else 'diferencia NO significativa'}",
                True))
        for k, d in comp.get("desacuerdo", {}).items():
            puntos.append(f"Un oráculo que eligiera siempre el modelo correcto llegaría a "
                          f"{pct(d['oraculo_accuracy'])} de accuracy, frente a "
                          f"{pct(max(r['test']['accuracy'], v['test']['accuracy']))} del mejor "
                          f"modelo individual: hay complementariedad que explotar.")
            puntos.append(f"Ambos modelos fallan simultáneamente en solo {pct(d['ambos_fallan'])} "
                          f"de las imágenes.")
    puntos.append("Las 2.262 imágenes de test vienen de solo 72 frutas: asumir independencia "
                  "entre imágenes daría intervalos artificialmente angostos.")
    vinetas(s, Inches(0.7), y + Inches(1.65), Inches(6.2), puntos, size=12.5)

    imagen(s, FIGURES_DIR / "10_matrices_confusion_avance.png", Inches(7.25), y + Inches(1.9),
           Inches(5.4), Inches(2.9))
    nota(s, "Las dos matrices concentran el error en la diagonal vecina: el modelo confunde "
            "estados contiguos, no extremos.")

    # ------------------------------------------------------------------ 10
    s, y = slide_titulo(prs, "Curvas de entrenamiento")
    imagen_centrada(s, FIGURES_DIR / "11_curvas_entrenamiento_avance.png", y, Inches(12.0), Inches(3.5))
    vinetas(s, Inches(0.7), y + Inches(3.7), Inches(11.9), [
        f"ResNet-50 alcanza su mejor QWK de validación en la época {r['best_epoch']}; "
        f"ViT-S/16 en la {v['best_epoch']}.",
        f"Tiempo de entrenamiento: {r['train_minutes']:.0f} min vs {v['train_minutes']:.0f} min "
        f"en una RTX 4070 Laptop de 8 GB.",
        "Selección del checkpoint por QWK de validación, no por accuracy: es la métrica "
        "que refleja el costo real de los errores.",
    ], size=14)

    # ------------------------------------------------------------------ 11 Riesgos
    s, y = slide_titulo(prs, "Riesgos identificados y mitigación",
                        "Concretos de este dataset, no genéricos")
    filas = [
        ["Riesgo", "Por qué es real acá", "Mitigación"],
        ["Fuga longitudinal",
         "6.871 pares de días consecutivos de la misma fruta",
         "Split por fruta + assert automático"],
        ["Domain gap laboratorio → terreno",
         "Fondo blanco, fruta centrada, luz difusa controlada",
         "Declarado en la data card; recolectar fotos de celular"],
        ["Color = señal, no ruido",
         "Un ColorJitter estándar vuelve una clase 2 en una clase 4",
         "Jitter suave: hue 0,02 en vez de 0,1"],
        ["Sesgo T10 ⇒ clase 1",
         "T10 aporta 60 % de las imágenes con 40 % de las frutas",
         "Pesos de clase + métricas desglosadas por grupo"],
        ["Etiquetas subjetivas",
         "Sin acuerdo inter-evaluador: techo humano desconocido",
         "QWK y MAE en vez de accuracy; no perseguir 100 %"],
        ["Ventaja oculta de pre-entrenamiento",
         "Los pesos ViT habituales vienen de ImageNet-21k",
         "DeiT-S: ambos modelos solo ImageNet-1k"],
    ]
    tabla(s, Inches(0.7), y, Inches(11.9), Inches(4.2), filas,
          anchos=[2.4, 4.3, 4.2], size=12)
    nota(s, "El domain gap es el riesgo que decide si esto sirve en un packing o solo en un paper.")

    # ------------------------------------------------------------------ 12
    s, y = slide_titulo(prs, "Qué viene para el entregable final")
    vinetas(s, Inches(0.7), y, Inches(11.9), [
        ("Ablación sin pre-entrenamiento.", True),
        "Los mismos dos modelos entrenados desde cero. Es la evidencia empírica propia "
        "de por qué ResNet y ViT difieren: el transformer no tiene sesgo inductivo de "
        "localidad y tiene que aprenderlo de los datos.",
        ("Dos modelos híbridos.", True),
        "Fusión tardía de los dos backbones ya afinados, y el ViT-Hybrid R26+S/32 del "
        "paper original de ViT, donde un stem convolucional alimenta al transformer.",
        ("Interpretabilidad comparada.", True),
        "Grad-CAM sobre la ResNet y attention rollout sobre el ViT, en los casos donde "
        "los dos modelos discrepan.",
        ("Demo funcional en Gradio", True),
        "y decisión explícita de velocidad frente a precisión.",
    ], size=16)
    nota(s, "Repositorio: github.com/cmartinez1524/clasificacion-de-paltas")

    numerar(prs)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    prs.save(SALIDA)
    print(f"Presentación de avance: {SALIDA}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
