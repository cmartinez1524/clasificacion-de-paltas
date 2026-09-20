"""Genera entregables/final/presentacion_final.pptx desde las metricas reales.

Uso:
    python scripts/06_deck_final.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pptx.util import Inches, Pt  # noqa: E402

from paltas.deck import (  # noqa: E402
    AZUL,
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
from paltas.slides_comunes import (  # noqa: E402
    slide_como_miran,
    slide_por_que_entrenan_distinto,
)

use_utf8()

SALIDA = ROOT / "entregables" / "final" / "presentacion_final.pptx"
AUTOR = "Cristóbal Martínez"
FECHA = "Entrega final · 23 de noviembre de 2026"

ETIQUETA = {
    "resnet50": "ResNet-50",
    "vit_small": "ViT-S/16",
    "resnet50_scratch": "ResNet-50 (scratch)",
    "vit_small_scratch": "ViT-S/16 (scratch)",
    "hybrid_fusion": "Híbrido · fusión tardía",
    "hybrid_vit_r26": "ViT-Hybrid R26+S/32",
}


def cargar(nombre: str) -> dict | None:
    p = METRICS_DIR / f"{nombre}_test.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def main() -> None:
    res = {k: cargar(k) for k in ETIQUETA}
    disponibles = [k for k, v in res.items() if v]
    if "resnet50" not in disponibles or "vit_small" not in disponibles:
        raise SystemExit("Faltan los baselines. Entrena resnet50 y vit_small primero.")

    r, v = res["resnet50"], res["vit_small"]
    comp_path = METRICS_DIR / "final.json"
    comp = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else None

    prs = nueva_presentacion()

    # ------------------------------------------------------------------ portada
    slide_portada(
        prs,
        "Clasificación del estado de\nmadurez de paltas Hass",
        "ResNet-50, ViT-S/16 e híbridos con presupuesto equiparado",
        AUTOR, FECHA,
        "14.710 fotografías · 478 frutas · índice ordinal de 5 niveles",
    )

    # ------------------------------------------------------------------ problema
    s, y = slide_titulo(prs, "Problema y motivación",
                        "Una ventana de madurez estrecha y una clasificación manual")
    vinetas(s, Inches(0.7), y, Inches(6.3), [
        ("Chile es el tercer exportador mundial de paltas Hass.", True),
        "La clasificación por madurez se hace a ojo y al tacto, operario por operario, "
        "sin trazabilidad ni criterio homogéneo.",
        "Fruta verde en góndola: el consumidor la descarta. Fruta pasada en tránsito: "
        "pérdida total. Los dos errores cuestan, y cuestan distinto.",
        ("La tarea es ordinal.", True),
        "Confundir un 1 con un 2 es un error de un día; confundir un 1 con un 5 es "
        "mandar fruta podrida a la venta. La métrica debe reflejarlo.",
    ])
    tabla(s, Inches(7.4), y + Inches(0.15), Inches(5.2), Inches(2.6), [
        ["Índice", "Estado", "Decisión"],
        ["1", "Verde", "Dejar madurar"],
        ["2", "Iniciando", "Lista en 2-3 días"],
        ["3", "Maduro I", "Venta inmediata"],
        ["4", "Maduro II", "Último día"],
        ["5", "Sobremaduro", "Fuera de punto"],
    ], anchos=[1, 2, 3], size=12)
    nota(s, "Métrica principal: QWK (kappa cuadrático), que penaliza el error según el "
            "cuadrado de la distancia entre clases.")

    # ------------------------------------------------------------------ datos
    s, y = slide_titulo(prs, "Datos y limitaciones",
                        "Hass Avocado Ripening Photographic Dataset · Mendeley · CC BY 4.0")
    for i, (t, val, det, c) in enumerate([
        ("IMÁGENES", "14.710", "800×800 px", VERDE),
        ("FRUTAS", "478", "hasta 26 días c/u", AZUL),
        ("TRAIN / VAL / TEST", "334 / 72 / 72", "frutas, no imágenes", MORADO),
        ("LICENCIA", "CC BY 4.0", "uso libre con atribución", ROJO),
    ]):
        tarjeta(s, Inches(0.7 + i * 3.08), y, Inches(2.86), Inches(1.5), t, val, det, c)
    vinetas(s, Inches(0.7), y + Inches(1.75), Inches(11.9), [
        ("Fuga longitudinal — resuelta.", True),
        "6.871 pares de días consecutivos de la misma fruta. Partimos frutas, no imágenes, "
        "con un assert que falla si una fruta aparece en dos particiones.",
        ("Domain gap laboratorio → terreno — no resuelta, declarada.", True),
        "Fondo blanco uniforme, una sola fruta centrada, iluminación difusa controlada, "
        "distancia fija. Una foto de celular en una feria está fuera de distribución. "
        "Es la limitación que decide si esto sirve en un packing o solo en un paper.",
        ("Un lote, una cosecha, un cultivar, etiquetas subjetivas.", True),
        "478 frutas portuguesas de 2022, solo Hass, etiquetado visual sin acuerdo "
        "inter-evaluador reportado: no conocemos el techo humano de esta tarea.",
    ], size=15)
    nota(s, "Detalle completo en la data card (entregables/final/data_card.md).")

    # --------------------------------------------- slides conceptuales
    # Compartidas con la presentación de avance y espejo de los frames del
    # generador Beamer: la explicación de las redes es una sola.
    slide_como_miran(prs)
    slide_por_que_entrenan_distinto(prs)

    # ------------------------------------------------------------------ diseño
    s, y = slide_titulo(prs, "Diseño de la comparación",
                        "Tres ejes igualados, no solo los parámetros")
    tabla(s, Inches(0.7), y, Inches(11.9), Inches(2.0), [
        ["Eje igualado", "ResNet-50", "ViT-S/16 (DeiT-S)", "Diferencia"],
        ["Parámetros", f"{r['params_M']:.2f} M", f"{v['params_M']:.2f} M",
         f"{abs(r['params_M']-v['params_M'])/max(r['params_M'],v['params_M'])*100:.1f} %"],
        ["GFLOPs @ 224 px", f"{r['gflops']:.2f}", f"{v['gflops']:.2f}",
         f"{abs(r['gflops']-v['gflops'])/max(r['gflops'],v['gflops'])*100:.1f} %"],
        ["Pre-entrenamiento", "ImageNet-1k", "ImageNet-1k", "ninguna"],
    ], anchos=[3, 2, 2.4, 1.6], size=14)
    vinetas(s, Inches(0.7), y + Inches(2.35), Inches(11.9), [
        ("El pre-entrenamiento es el eje que casi nadie iguala.", True),
        "Los pesos ViT-S habituales de timm vienen de ImageNet-21k: catorce veces más "
        "datos. Con esos pesos la comparación mediría el corpus, no la arquitectura. "
        "Usamos DeiT-S, misma arquitectura entrenada solo en ImageNet-1k.",
        "Receta idéntica; el ViT solo cambia LR (3× menor), warmup y drop_path, porque "
        "diverge con la receta de la CNN.",
    ], size=15)
    nota(s, "Todo el código de la comparación está en el repositorio y se reproduce con un comando.")

    # ------------------------------------------------------------------ resultados
    slide_seccion(prs, "01", "Resultados")

    s, y = slide_titulo(prs, "Tabla comparativa completa",
                        "Test: 2.262 imágenes de 72 frutas nunca vistas · IC 95 % "
                        "por bootstrap agrupado por fruta")
    filas = [["Modelo", "Params", "GFLOPs", "QWK", "Accuracy", "Macro-F1", "MAE", "Acc. ±1", "min"]]
    for k in disponibles:
        d = res[k]
        t = d["test"]
        filas.append([ETIQUETA[k], f"{d['params_M']:.1f} M", f"{d['gflops']:.1f}",
                      f"{t['qwk']:.3f}", pct(t["accuracy"]), f"{t['macro_f1']:.3f}",
                      f"{t['mae']:.3f}", pct(t["off_by_one"]), f"{d['train_minutes']:.0f}"])
    mejor = 1 + max(range(len(disponibles)), key=lambda i: res[disponibles[i]]["test"]["qwk"])
    tabla(s, Inches(0.7), y, Inches(11.9), Inches(0.5 + 0.38 * len(filas)), filas,
          anchos=[2.6, 1.1, 1.1, 1.1, 1.2, 1.2, 0.9, 1.1, 0.8], size=12,
          destacar_fila=mejor)
    nota(s, "Fila destacada: mejor QWK. El híbrido ViT-R26 usa pesos de ImageNet-21k "
            "y más parámetros: es cota superior de referencia, no competidor equiparable.")

    s, y = slide_titulo(prs, "ResNet vs ViT: la comparación, no solo el número final",
                        "Todas las diferencias contra ResNet-50, bootstrap pareado "
                        "agrupado por fruta")
    imagen(s, FIGURES_DIR / "10_matrices_confusion_final.png", Inches(0.7), y,
           Inches(6.3), Inches(4.6))
    puntos = []
    if comp:
        for c in comp.get("comparaciones", []):
            d = c["metricas"]["qwk"]
            puntos.append((
                f"{ETIQUETA.get(c['modelo_a'], c['modelo_a'])}:  Δ QWK = {d['diff']:+.4f}  "
                f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  p = {d['p_value']:.3f}  → "
                f"{'significativa' if d['significativo'] else 'NO significativa'}", True))
    puntos.append("Todas las matrices son bandeadas: los errores de distancia ≥ 2 son casi "
                  "inexistentes. Ningún modelo confunde una palta verde con una sobremadura.")
    puntos.append("El cuello de botella son las clases intermedias (2, 3 y 4), donde la "
                  "frontera es un corte continuo y las etiquetas son más subjetivas.")
    puntos.append("Los dos modelos desde cero pierden la estructura: el ViT colapsa hacia las "
                  "clases altas y predice 4 para el 68 % de las paltas que son clase 5.")
    vinetas(s, Inches(7.3), y, Inches(5.4), puntos, size=11.5, espacio=Pt(9))

    s, y = slide_titulo(prs, "Por qué difieren (o no) ResNet y ViT",
                        "La ablación sin pre-entrenamiento es la evidencia")
    if res.get("resnet50_scratch") and res.get("vit_small_scratch"):
        rs, vs = res["resnet50_scratch"], res["vit_small_scratch"]
        tabla(s, Inches(0.7), y, Inches(11.9), Inches(1.7), [
            ["", "ResNet-50", "ViT-S/16", "Lectura"],
            ["QWK con ImageNet-1k", f"{r['test']['qwk']:.3f}", f"{v['test']['qwk']:.3f}", ""],
            ["QWK desde cero", f"{rs['test']['qwk']:.3f}", f"{vs['test']['qwk']:.3f}", ""],
            # Delta con signo: quitar el pre-entrenamiento BAJA la metrica.
            ["Δ sin pre-entrenamiento",
             f"{rs['test']['qwk'] - r['test']['qwk']:+.4f}",
             f"{vs['test']['qwk'] - v['test']['qwk']:+.4f}",
             f"el ViT pierde "
             f"{(v['test']['qwk'] - vs['test']['qwk']) / max(r['test']['qwk'] - rs['test']['qwk'], 1e-9):.1f}× más"],
            ["Δ accuracy sin pre-entren.",
             f"{100 * (rs['test']['accuracy'] - r['test']['accuracy']):+.1f} pts",
             f"{100 * (vs['test']['accuracy'] - v['test']['accuracy']):+.1f} pts",
             f"el ViT pierde "
             f"{(v['test']['accuracy'] - vs['test']['accuracy']) / max(r['test']['accuracy'] - rs['test']['accuracy'], 1e-9):.1f}× más"],
        ], anchos=[3.4, 2, 2, 3], size=13)
        yy = y + Inches(2.5)
    else:
        yy = y
    vinetas(s, Inches(0.7), yy, Inches(11.9), [
        ("La ResNet trae el sesgo inductivo incorporado en la arquitectura.", True),
        "La convolución asume localidad y equivarianza a la traslación: una mancha café "
        "significa lo mismo esté donde esté en la fruta. Eso no hay que aprenderlo.",
        ("El ViT tiene que aprender ese sesgo de los datos.", True),
        "La auto-atención es global desde la primera capa y no asume nada sobre estructura "
        "espacial. Con 10.208 imágenes de 334 frutas no alcanza para descubrirlo desde cero; "
        "con ImageNet-1k detrás, sí.",
        ("Es un problema de textura local, no de forma global.", True),
        "El índice de madurez se lee en el color y las manchas de la cáscara, justo donde "
        "la convolución es fuerte. El campo receptivo global del ViT aporta poco.",
    ], size=15)

    # ------------------------------------------------------------------ híbridos
    if res.get("hybrid_fusion") or res.get("hybrid_vit_r26"):
        s, y = slide_titulo(prs, "Los híbridos", "¿Suma combinar ambos enfoques?")
        hib = [k for k in ("hybrid_fusion", "hybrid_vit_r26") if res.get(k)]
        for i, k in enumerate(hib):
            d = res[k]
            tarjeta(s, Inches(0.7 + i * 6.1), y, Inches(5.8), Inches(1.6),
                    ETIQUETA[k].upper(), f"QWK {d['test']['qwk']:.3f}",
                    f"{d['params_M']:.1f} M params · {d['gflops']:.1f} GFLOPs · "
                    f"acc {pct(d['test']['accuracy'])}", MORADO)
        puntos = []
        if comp:
            for k, dd in comp.get("desacuerdo", {}).items():
                puntos.append(f"Techo de un oráculo que eligiera siempre el modelo correcto: "
                              f"{pct(dd['oraculo_accuracy'])}. Ambos fallan a la vez en "
                              f"{pct(dd['ambos_fallan'])} de las imágenes.")
                break
        puntos += [
            ("La fusión tardía cuesta el doble de cómputo por imagen.", True),
            "Dos backbones completos en el forward. La pregunta no es si mejora, sino si "
            "mejora lo suficiente para justificar 2× de latencia.",
            "El ViT-Hybrid R26 no es equiparable: 36 M de parámetros y pesos de ImageNet-21k. "
            "Va como cota superior de referencia, no como competidor.",
        ]
        vinetas(s, Inches(0.7), y + Inches(1.95), Inches(6.2), puntos, size=12.5)
        if (FIGURES_DIR / "13_costo_beneficio_final.png").exists():
            imagen(s, FIGURES_DIR / "13_costo_beneficio_final.png",
                   Inches(7.1), y + Inches(2.1), Inches(5.6), Inches(2.4))

    # ------------------------------------------------------------------ interpretabilidad
    fig_int = FIGURES_DIR / "20_interpretabilidad_desacuerdo.png"
    if not fig_int.exists():
        fig_int = FIGURES_DIR / "20_interpretabilidad_aleatorio.png"
    if fig_int.exists():
        s, y = slide_titulo(prs, "Dónde mira cada arquitectura",
                            "Grad-CAM en la ResNet · attention rollout en el ViT")
        # La figura es muy alta (5 filas): va como columna izquierda angosta,
        # no centrada, o el texto se le monta encima.
        imagen(s, fig_int, Inches(0.7), y - Inches(0.35), Inches(3.1), Inches(5.2))
        vinetas(s, Inches(4.3), y, Inches(8.3), [
            "No usamos la misma técnica en ambos a propósito: Grad-CAM necesita un mapa "
            "de activaciones con estructura espacial, y en un ViT ese mapa es una secuencia "
            "de tokens que produce resultados ruidosos.",
            "Attention rollout acumula las matrices de atención de todas las capas para "
            "estimar cuánto aporta cada parche al token CLS.",
            ("La ResNet se activa sobre texturas locales: manchas, arrugas, zonas oscuras.", True),
            ("El ViT reparte la atención sobre regiones más extensas de la cáscara.", True),
            "Ambas responden la misma pregunta por caminos distintos, y esa diferencia es "
            "coherente con el resultado cuantitativo.",
        ], size=13)

    # ------------------------------------------------------------------ velocidad vs precisión
    s, y = slide_titulo(prs, "Decisión: velocidad frente a precisión",
                        "Qué modelo llevaríamos a una línea de empaque")
    # Solo los modelos pre-entrenados: las corridas desde cero son una ablacion
    # explicativa, no candidatas a despliegue. Costo relativo a la ResNet-50,
    # que es la referencia de la decision.
    candidatos = [k for k in disponibles if "scratch" not in k]
    base_gf = res["resnet50"]["gflops"]
    base_qwk = res["resnet50"]["test"]["qwk"]
    veredicto = {
        "resnet50": "Referencia. La recomendación operativa.",
        "vit_small": "Empata en QWK al mismo costo y converge 2× más rápido.",
        "hybrid_fusion": "Mejora real pero marginal: +0,6 % de QWK por 2× de cómputo.",
        "hybrid_vit_r26": "Mejor QWK y 16 % más barato, pero con ventaja de ImageNet-21k.",
    }
    filas = [["Modelo", "QWK", "Δ QWK", "GFLOPs", "Costo rel.", "Veredicto"]]
    for k in candidatos:
        d = res[k]
        dq = d["test"]["qwk"] - base_qwk
        filas.append([ETIQUETA[k], f"{d['test']['qwk']:.4f}",
                      "—" if k == "resnet50" else f"{dq:+.4f}",
                      f"{d['gflops']:.2f}", f"{d['gflops']/base_gf:.2f}×",
                      veredicto.get(k, "")])
    fila_rec = 1 + candidatos.index("resnet50")
    tabla(s, Inches(0.7), y, Inches(11.9), Inches(0.5 + 0.38 * len(filas)), filas,
          anchos=[2.4, 1.1, 1.1, 1.1, 1.1, 5.4], size=11.5, destacar_fila=fila_rec)
    vinetas(s, Inches(0.7), y + Inches(0.6 + 0.38 * len(filas)), Inches(11.9), [
        ("El criterio no es el QWK máximo, es el QWK por unidad de cómputo.", True),
        "En un packing la inferencia corre sobre miles de frutas por hora, probablemente "
        "en un equipo sin GPU dedicada.",
        "Una diferencia de QWK que no es estadísticamente significativa no justifica "
        "duplicar el cómputo: si los intervalos se solapan, gana el modelo más barato.",
        "Con accuracy ±1 clase por sobre el 99 %, el error residual está dentro del ruido "
        "de etiquetado humano. Optimizar más el modelo no es donde está el retorno; "
        "cerrar el domain gap sí.",
    ], size=15)

    # ------------------------------------------------------------------ demo
    s, y = slide_titulo(prs, "Demo", "Gradio · casos nuevos del conjunto de test")
    vinetas(s, Inches(0.7), y, Inches(11.9), [
        ("Selección de modelo, distribución de probabilidad sobre las 5 clases y "
         "recomendación operativa.", True),
        "Muestra además el mapa de atención del modelo: la demo no es una caja negra.",
        "Reporta el índice esperado (promedio ponderado), más informativo que el argmax "
        "cuando el modelo duda entre estados adyacentes, que es el caso frecuente.",
        "Avisa explícitamente cuando la confianza es baja y conviene revisión humana.",
        ("Los ejemplos precargados salen solo del conjunto de test: son frutas que ningún "
         "modelo vio durante el entrenamiento.", True),
        "Advierte en pantalla sobre el domain gap: fue entrenado con fotos de laboratorio.",
    ], size=16)
    nota(s, "python app/gradio_app.py  →  http://localhost:7860")

    # ------------------------------------------------------------------ cierre
    s, y = slide_titulo(prs, "Conclusiones y trabajo futuro")
    vinetas(s, Inches(0.7), y, Inches(11.9), [
        ("Con presupuesto equiparado y pre-entrenamiento equiparado, las dos "
         "arquitecturas resuelven esta tarea a un nivel comparable.", True),
        "La diferencia relevante no está entre CNN y transformer, sino entre tener o no "
        "tener pre-entrenamiento: ahí sí se separan, y se separan de forma asimétrica.",
        ("El techo de esta tarea lo pone el etiquetado, no la arquitectura.", True),
        "Sin acuerdo inter-evaluador reportado no sabemos cuánto del error residual es "
        "del modelo y cuánto es ruido de la etiqueta.",
        ("Trabajo futuro, en orden de retorno esperado:", True),
        "1) Recolectar fotos de celular en condiciones reales chilenas y medir la caída. "
        "2) Pérdida ordinal explícita (CORAL) en vez de entropía cruzada plana. "
        "3) Predicción de vida útil restante en días, que es la pregunta que de verdad "
        "importa en la cadena de frío.",
    ], size=15)
    nota(s, "github.com/cmartinez1524/clasificacion-de-paltas")

    numerar(prs)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    prs.save(SALIDA)
    print(f"Presentación final: {SALIDA}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
