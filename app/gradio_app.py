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
from threading import Lock
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.data import build_transforms  # noqa: E402
from paltas.interpret import GradCAM, attention_rollout, denormalize, resnet_target_layer  # noqa: E402
from paltas.metrics import bayes_optimal_predict  # noqa: E402
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
    "resnet50_app": ("resnet50", "ResNet-50 · nueva app"),
    "resnet50_app_s1_v2": ("resnet50", "ResNet-50 · v2 experimental (semilla 1)"),
    "resnet50": ("resnet50", "ResNet-50"),
    "vit_small": ("vit_small", "ViT-S/16"),
    "resnet50_asimetrico": ("resnet50", "ResNet-50 (asimétrico)"),
    "vit_small_asimetrico": ("vit_small", "ViT-S/16 (asimétrico)"),
    "hybrid_fusion": ("hybrid_fusion", "Hibrido (fusion tardia)"),
    "hybrid_vit_r26": ("hybrid_vit_r26", "ViT-Hybrid R26+S/32"),
}
for semilla in (1, 2):
    for arquitectura, etiqueta in (("resnet50", "ResNet-50"), ("vit_small", "ViT-S/16")):
        DISPONIBLES[f"{arquitectura}_asimetrico_s{semilla}"] = (
            arquitectura, f"{etiqueta} (asimétrico, semilla {semilla})")

ACCION = {
    1: "Aun no comestible. Dejar madurar varios dias a temperatura ambiente.",
    2: "Empezando a madurar. Lista en 2-3 dias.",
    3: "Lista para consumo. Optima para venta inmediata.",
    4: "Madura, ultimo dia de vida util. Consumir hoy.",
    5: "Sobremadura. Fuera de su punto de consumo.",
}

_cache: dict[str, torch.nn.Module] = {}
_inference_lock = Lock()


def cargar_modelo(clave: str) -> torch.nn.Module:
    if clave in _cache:
        return _cache[clave]
    arq, _ = DISPONIBLES[clave]
    ckpt = CHECKPOINT_DIR / f"{clave}.pt"
    model = build_model(arq, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
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
    # Serializa los hooks de Grad-CAM también en peticiones HTTP sin cola.
    with _inference_lock:
        if imagen is None:
            return {}, None, "Sube o selecciona una imagen."

        clave = next((k for k, (_, lbl) in DISPONIBLES.items() if lbl == clave_modelo), None)
        if clave is None:
            raise gr.Error("Selecciona un modelo entrenado disponible.")
        model = cargar_modelo(clave)
        imagen = ImageOps.exif_transpose(imagen).convert("RGB")
        x = TF(imagen).unsqueeze(0).to(DEVICE)
        # El mapa corresponde al recorte que realmente vio la red, no a la foto estirada.
        base = Image.fromarray((denormalize(x[0]) * 255).round().astype(np.uint8))

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

        if "_asimetrico" in clave:
            pred = int(bayes_optimal_predict(probs[None, :], underripe_penalty=2.0, overripe_penalty=1.0)[0]) + 1
        else:
            pred = int(probs.argmax()) + 1
        etiquetas = {f"{i+1} · {CLASS_NAMES[i+1]}": float(probs[i]) for i in range(5)}

        # Valor esperado del indice: mas informativo que el argmax cuando el modelo
        # duda entre dos estados adyacentes, que es el caso frecuente.
        esperado = float((probs * np.arange(1, 6)).sum())

        texto = (
            f"### Indice predicho: **{pred}** — {CLASS_NAMES[pred]}\n\n"
            f"**Recomendacion:** {ACCION[pred]}\n\n"
            f"Indice esperado (promedio ponderado): **{esperado:.2f}**  ·  "
            f"probabilidad del estado elegido {probs[pred - 1]:.1%}  ·  modelo: {clave_modelo}"
        )
        if probs.max() < 0.55:
            texto += ("\n\n> El modelo no esta seguro: la probabilidad esta repartida entre "
                      "estados vecinos. Conviene revision humana.")

        overlay = None
        if mostrar_mapa:
            try:
                mapa = mapa_de_atencion(model, clave, x, pred - 1)
                if mapa is not None:
                    overlay = superponer(base, mapa)
                    texto += ("\n\n**Mapa de calor:** rojo indica mayor intensidad. Se superpone al "
                              "recorte que vio el modelo. Es una aproximación, no una segmentación de la fruta.")
                    if DISPONIBLES[clave][0] in ("vit_small", "hybrid_vit_r26"):
                        texto += " En ViT muestra atención entre parches; no es específico de la clase predicha."
            except Exception as e:  # la demo no debe caerse por el mapa
                print(f"[aviso] no se pudo generar el mapa: {e}")
                texto += "\n\nNo se pudo generar el mapa de calor para esta imagen."

        return etiquetas, overlay, texto


def mapa_de_atencion(model, clave: str, x, class_idx: int | None = None):
    """Devuelve el mapa de saliencia adecuado a cada arquitectura.

    Para la fusion tardia usamos Grad-CAM sobre su ResNet interna: el gradiente
    viaja desde el logit final, a traves de la cabeza y de la proyeccion, hasta
    las activaciones convolucionales, asi que el mapa refleja lo que el modelo
    COMPLETO usa de la rama convolucional, no lo que usaria la ResNet sola.
    """
    if clave.startswith("resnet50"):
        with GradCAM(model, resnet_target_layer(model)) as cam:
            return cam(x, class_idx=class_idx)[0]
    if clave.startswith("vit_small"):
        return attention_rollout(model, x)[0]
    if clave == "hybrid_fusion":
        with GradCAM(model, model.cnn.layer4[-1]) as cam:
            return cam(x, class_idx=class_idx)[0]
    if clave == "hybrid_vit_r26":
        # Arquitectura ViT de timm con stem convolucional: expone .blocks igual
        # que un ViT puro, asi que el rollout se aplica sin cambios.
        return attention_rollout(model, x)[0]
    return None


def ejemplos_de_test(n_por_clase: int = 2, seed: int = 11) -> list[str]:
    """Ejemplos tomados SOLO del conjunto de test: son frutas que el modelo nunca vio."""
    if not SPLITS_CSV.exists():
        return sorted(str(p) for p in (Path(__file__).resolve().parent / "ejemplos").glob("*.jpg"))
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
            "Falta el modelo entrenado en checkpoints/. Usa el ZIP completo de la demo, "
            "que incluye resnet50_app.pt; no necesitas reentrenar."
        )
    opciones = [DISPONIBLES[k][1] for k in entrenados]

    with gr.Blocks(title="Madurez de paltas Hass", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Clasificacion del estado de madurez de paltas Hass\n"
            "Conecta una cámara al computador o sube una foto. La clasificación se calcula "
            "en este equipo y estima el estado en el índice "
            "de madurez de 5 niveles.\n\n"
            "> **Importante:** el modelo fue entrenado con fotografias de laboratorio "
            "(fondo blanco uniforme, fruta unica y centrada, iluminacion controlada). "
            "En fotos de celular tomadas en una feria o supermercado el desempeno cae: "
            "es el *domain gap* documentado en el informe."
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Foto"):
                    img = gr.Image(type="pil", sources=["upload"], label="Subir foto", height=320)
                    btn = gr.Button("Clasificar foto", variant="primary")
                with gr.Tab("Cámara en vivo"):
                    camara = gr.Image(
                        type="pil", sources=["webcam"], streaming=True,
                        webcam_options=gr.WebcamOptions(mirror=False, constraints={
                            "video": {"facingMode": {"ideal": "environment"}}}),
                        label="Cámara conectada a este computador", height=320)
                    gr.Markdown("Activa la cámara para analizar imágenes aproximadamente cada segundo. "
                                "Usa el control de la cámara para detener el envío.")
                modelo = gr.Radio(opciones, value=opciones[0], label="Modelo", visible=len(opciones) > 1)
                mapa_chk = gr.Checkbox(
                    True, label="Mostrar mapa de calor"
                )
            with gr.Column(scale=1):
                salida_txt = gr.Markdown()
                probs = gr.Label(num_top_classes=5, label="Distribucion de probabilidad")
                overlay = gr.Image(label="Mapa de calor sobre el recorte analizado", height=320,
                                   format="png", buttons=["download", "fullscreen"],
                                   interactive=False)

        ejemplos = ejemplos_de_test()
        if ejemplos:
            gr.Examples(
                examples=[[e] for e in ejemplos],
                inputs=[img],
                label="Ejemplos del conjunto de test (frutas que el modelo nunca vio)",
                examples_per_page=10,
            )

        gr.on([btn.click, img.change], predecir,
              [img, modelo, mapa_chk], [probs, overlay, salida_txt],
              queue=False, trigger_mode="always_last")
        camara.stream(predecir, [camara, modelo, mapa_chk], [probs, overlay, salida_txt],
                      stream_every=1.0, queue=False, trigger_mode="always_last")

    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="genera un enlace publico temporal")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 permite acceso desde la red local")
    ap.add_argument("--cpu", action="store_true", help="fuerza CPU aunque haya una GPU disponible")
    ap.add_argument("--open-browser", action="store_true", help="abre la app en el navegador del equipo")
    args = ap.parse_args()
    if args.cpu:
        DEVICE = torch.device("cpu")

    print(f"Dispositivo: {DEVICE}")
    print(f"Modelos disponibles: {modelos_entrenados()}")
    construir_interfaz().launch(share=args.share, server_port=args.port,
                               server_name=args.host, pwa=True, theme=gr.themes.Soft(),
                               inbrowser=args.open_browser)
