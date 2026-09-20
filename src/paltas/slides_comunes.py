"""Slides conceptuales compartidas por las dos presentaciones .pptx.

Viven en un solo lugar porque el avance y la final tienen que explicar las redes
exactamente igual. Son el espejo de los frames homonimos del generador Beamer
(scripts/07_deck_latex.py): si se cambia la explicacion, se cambia en los dos.
"""

from __future__ import annotations

from pptx.util import Inches

from .deck import (
    AZUL,
    ROJO,
    caja_texto,
    imagen_centrada,
    nota,
    slide_titulo,
    vinetas,
)
from .paths import FIGURES_DIR


def slide_como_miran(prs) -> None:
    """Convolucion frente a atencion, con la figura didactica."""
    s, y = slide_titulo(prs, "Las dos redes, en una imagen",
                        "La diferencia es CÓMO miran la foto")
    imagen_centrada(s, FIGURES_DIR / "30_como_mira_cada_red.png", y,
                    Inches(10.2), Inches(3.3))
    vinetas(s, Inches(0.8), y + Inches(3.55), Inches(5.5), [
        "Va de a pedacitos y reutiliza el mismo detector en toda la foto "
        "(convolución).",
        "Buena para detalles: manchas, arrugas, tono.",
    ], size=12)
    vinetas(s, Inches(6.9), y + Inches(3.55), Inches(5.6), [
        "Parte la foto en 196 cuadraditos y los compara todos contra todos "
        "(auto-atención).",
        "Buena para relacionar zonas distantes.",
    ], size=12)
    nota(s, "Ninguna se programó desde cero: las dos vienen de timm, la biblioteca "
            "estándar de modelos de visión.")


def slide_por_que_entrenan_distinto(prs) -> None:
    """Sesgo inductivo en criollo, y su consecuencia sobre el entrenamiento."""
    s, y = slide_titulo(prs, "Por qué entrenan tan distinto",
                        "Todo se explica por lo que cada red ya sabe antes de empezar")
    # 2,45" de alto: con menos, el pie en cursiva se monta sobre la última viñeta
    caja_texto(s, Inches(0.7), y, Inches(5.85), Inches(2.45),
               "ResNet-50 trae dos reglas de fábrica",
               ["Lo que importa está cerca",
                "No importa en qué parte de la foto esté"],
               "Una mancha café es una mancha café, arriba o abajo de la fruta. "
               "Eso no lo aprende: ya lo trae.", AZUL)
    caja_texto(s, Inches(6.75), y, Inches(5.85), Inches(2.45),
               "ViT-S/16 no trae ninguna",
               ["Todos los cuadraditos le parecen iguales",
                "Ni siquiera sabe cuáles son vecinos"],
               "Tiene que deducir de los ejemplos hasta la noción de «al lado».", ROJO)
    vinetas(s, Inches(0.7), y + Inches(2.75), Inches(11.9), [
        ("El ViT necesita muchísimos más ejemplos.", True),
        "La ResNet arranca sabiendo mirar imágenes; el ViT tiene que descubrir primero "
        "cómo se mira una imagen, y recién después aprender de paltas.",
        ("Por eso los dos parten desde ImageNet.", True),
        "Antes de ver una sola palta ya vieron 1,3 millones de fotos de cosas "
        "cotidianas. Ahí el ViT compensa: llega con el oficio aprendido.",
        ("Y por eso el ViT es más delicado de entrenar.", True),
        "Entrenar es bajar un cerro a ciegas y el learning rate es el tamaño del paso. "
        "La ResNet aguanta pasos grandes; el ViT se cae.",
    ], size=12.5, espacio=Inches(0.06))
