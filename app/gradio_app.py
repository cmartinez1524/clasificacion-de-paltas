"""Demo interactiva: clasificador del estado de madurez de paltas Hass.

Carga los modelos entrenados disponibles, predice sobre una foto nueva y muestra
ademas el mapa de atencion del modelo, para que la demo no sea una caja negra.

Uso:
    python app/gradio_app.py
    python app/gradio_app.py --share        # enlace publico temporal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.data import build_transforms  # noqa: E402
from paltas.interpret import GradCAM, attention_rollout, resnet_target_layer  # noqa: E402
from paltas.models import build_model  # noqa: E402
from paltas.paths import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    IMAGES_DIR,
    SPLITS_CSV,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF = build_transforms(224, train=False)

#: checkpoint -> (arquitectura, etiqueta en la interfaz)
DISPONIBLES = {
    "resnet50": ("resnet50", "ResNet-50"),
    "vit_small": ("vit_small", "ViT-S/16"),
    "hybrid_fusion": ("hybrid_fusion", "Hibrido (fusion tardia)"),
    "hybrid_vit_r26": ("hybrid_vit_r26", "ViT-Hybrid R26+S/32"),
}

ACCION = {
    1: "Aun no comestible. Dejar madurar varios dias a temperatura ambiente.",
    2: "Empezando a madurar. Lista en 2-3 dias.",
    3: "Lista para consumo. Optima para venta inmediata.",
    4: "Madura, ultimo dia de vida util. Consumir hoy.",
    5: "Sobremadura. Fuera de su punto de consumo.",
}

_cache: dict[str, torch.nn.Module] = {}


def cargar_modelo(clave: str) -> torch.nn.Module:
    if clave in _cache:
        return _cache[clave]
    arq, _ = DISPONIBLES[clave]
    ckpt = CHECKPOINT_DIR / f"{clave}.pt"
    model = build_model(arq, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    model.to(DEVICE).eval()
    _cache[clave] = model
    return model


def modelos_entrenados() -> list[str]:
    return [k for k in DISPONIBLES if (CHECKPOINT_DIR / f"{k}.pt").exists()]


def superponer(base: Image.Image, mapa: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Superpone un mapa de calor [0,1] sobre la imagen, sin depender de matplotlib."""
    import matplotlib.cm as cm

    calor = (cm.jet(mapa)[:, :, :3] * 255).astype(np.uint8)
    calor = Image.fromarray(calor).resize(base.size)
    return Image.blend(base.convert("RGB"), calor, alpha)


def predecir(imagen: Image.Image, clave_modelo: str, mostrar_mapa: bool):
    if imagen is None:
        return {}, None, "Sube o selecciona una imagen."

    clave = next(k for k, (_, lbl) in DISPONIBLES.items() if lbl == clave_modelo)
    model = cargar_modelo(clave)
    base = imagen.convert("RGB").resize((224, 224))
    x = TF(imagen.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    pred = int(probs.argmax()) + 1
    etiquetas = {f"{i+1} · {CLASS_NAMES[i+1]}": float(probs[i]) for i in range(5)}

    # Valor esperado del indice: mas informativo que el argmax cuando el modelo
    # duda entre dos estados adyacentes, que es el caso frecuente.
    esperado = float((probs * np.arange(1, 6)).sum())

    texto = (
        f"### Indice predicho: **{pred}** — {CLASS_NAMES[pred]}\n\n"
        f"**Recomendacion:** {ACCION[pred]}\n\n"
        f"Indice esperado (promedio ponderado): **{esperado:.2f}**  ·  "
        f"confianza {probs.max():.1%}  ·  modelo: {clave_modelo}"
    )
    if probs.max() < 0.55:
        texto += ("\n\n> El modelo no esta seguro: la probabilidad esta repartida entre "
                  "estados vecinos. Conviene revision humana.")

    overlay = None
    if mostrar_mapa:
        try:
            if clave == "resnet50":
                with GradCAM(model, resnet_target_layer(model)) as cam:
                    mapa, _ = cam(x)
            elif clave == "vit_small":
                mapa, _ = attention_rollout(model, x)
            else:
                mapa = None
            if mapa is not None:
                overlay = superponer(base, mapa)
        except Exception as e:  # la demo no debe caerse por el mapa
            print(f"[aviso] no se pudo generar el mapa: {e}")

    return etiquetas, overlay, texto


def ejemplos_de_test(n_por_clase: int = 2, seed: int = 11) -> list[str]:
    """Ejemplos tomados SOLO del conjunto de test: son frutas que el modelo nunca vio."""
    if not SPLITS_CSV.exists():
        return []
    df = pd.read_csv(SPLITS_CSV)
    test = df[df["split"] == "test"]
    carpeta = CACHE_DIR if any(CACHE_DIR.glob("*.jpg")) else IMAGES_DIR
    rng = np.random.default_rng(seed)
    rutas = []
    for c in range(1, 6):
        sub = test[test["ripening_index"] == c]
        if len(sub) == 0:
            continue
        for _, r in sub.iloc[rng.choice(len(sub), min(n_por_clase, len(sub)), replace=False)].iterrows():
            p = carpeta / r["image_file"]
            if p.exists():
                rutas.append(str(p))
    return rutas


def construir_interfaz() -> gr.Blocks:
    entrenados = modelos_entrenados()
    if not entrenados:
        raise SystemExit(
            "No hay checkpoints en checkpoints/. Entrena al menos un modelo:\n"
            "  python scripts/train.py --config configs/resnet50.yaml"
        )
    opciones = [DISPONIBLES[k][1] for k in entrenados]

    with gr.Blocks(title="Madurez de paltas Hass", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Clasificacion del estado de madurez de paltas Hass\n"
            "Sube una foto de una palta y el modelo estima su estado en el indice "
            "de madurez de 5 niveles.\n\n"
            "> **Importante:** el modelo fue entrenado con fotografias de laboratorio "
            "(fondo blanco uniforme, fruta unica y centrada, iluminacion controlada). "
            "En fotos de celular tomadas en una feria o supermercado el desempeno cae: "
            "es el *domain gap* documentado en el informe."
        )
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Fotografia", height=320)
                modelo = gr.Radio(opciones, value=opciones[0], label="Modelo")
                mapa_chk = gr.Checkbox(
                    True, label="Mostrar mapa de atencion (Grad-CAM / attention rollout)"
                )
                btn = gr.Button("Clasificar", variant="primary")
            with gr.Column(scale=1):
                salida_txt = gr.Markdown()
                probs = gr.Label(num_top_classes=5, label="Distribucion de probabilidad")
                overlay = gr.Image(label="Donde mira el modelo", height=280)

        ejemplos = ejemplos_de_test()
        if ejemplos:
            gr.Examples(
                examples=[[e] for e in ejemplos],
                inputs=[img],
                label="Ejemplos del conjunto de test (frutas que el modelo nunca vio)",
                examples_per_page=10,
            )

        btn.click(predecir, [img, modelo, mapa_chk], [probs, overlay, salida_txt])
        img.change(predecir, [img, modelo, mapa_chk], [probs, overlay, salida_txt])

    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="genera un enlace publico temporal")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    print(f"Dispositivo: {DEVICE}")
    print(f"Modelos disponibles: {modelos_entrenados()}")
    construir_interfaz().launch(share=args.share, server_port=args.port)
