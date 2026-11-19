"""Demo interactiva: clasificador del estado de madurez de paltas Hass.

Carga los modelos entrenados disponibles, predice sobre una foto nueva y muestra
ademas el mapa de atencion del modelo, para que la demo no sea una caja negra.

Uso:
    python app/gradio_app.py
    python app/gradio_app.py --share        # enlace publico temporal
"""

from __future__ import annotations

import argparse
import html
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
    "resnet101_app_v3": ("resnet101", "ResNet-101 · v3 experimental"),
    "resnet101_app_parcial_v5": ("resnet101", "ResNet-101 · v5 parcial experimental"),
    "vit_base_app_capas_v5": ("vit_base", "ViT-Base/16 · v5 por capas experimental"),
    "resnet101_app_refinado_v4": ("resnet101", "ResNet-101 · v4 refinada experimental"),
    "resnet101_app_regularizado_v4": ("resnet101", "ResNet-101 · v4 regularizada experimental"),
    "vit_base_app_v3": ("vit_base", "ViT-Base/16 · v3 experimental"),
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

for semilla in range(1, 6):
    for prefijo, arquitectura, etiqueta in (
        ('resnet50_app', 'resnet50', 'ResNet-50'),
        ('vit_base_app', 'vit_base', 'ViT-Base'),
        ('resnet101_app', 'resnet101', 'ResNet-101 refinada'),
        ('resnet101_app_parcial', 'resnet101', 'ResNet-101 parcial'),
        ('vit_base_app_capas', 'vit_base', 'ViT-Base por capas'),
    ):
        DISPONIBLES[f'{prefijo}_s{semilla}_v6'] = (
            arquitectura, f'{etiqueta} · v6 experimental (semilla {semilla})')

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

        segunda = int(np.argsort(probs)[-2]) + 1
        duda = probs.max() < 0.55
        aviso_mapa = ""
        overlay = None
        if mostrar_mapa:
            try:
                mapa = mapa_de_atencion(model, clave, x, pred - 1)
                if mapa is not None:
                    overlay = superponer(base, mapa)
            except Exception as e:  # la demo no debe caerse por el mapa
                print(f"[aviso] no se pudo generar el mapa: {e}")
                aviso_mapa = "No se pudo generar el mapa de calor para esta imagen."

        texto = tarjeta(pred, probs, esperado, clave_modelo,
                        segunda if duda else None, aviso_mapa)
        return etiquetas, overlay, texto


def mapa_de_atencion(model, clave: str, x, class_idx: int | None = None):
    """Devuelve el mapa de saliencia adecuado a cada arquitectura.

    Para la fusion tardia usamos Grad-CAM sobre su ResNet interna: el gradiente
    viaja desde el logit final, a traves de la cabeza y de la proyeccion, hasta
    las activaciones convolucionales, asi que el mapa refleja lo que el modelo
    COMPLETO usa de la rama convolucional, no lo que usaria la ResNet sola.
    """
    if clave.startswith(("resnet50", "resnet101")):
        with GradCAM(model, resnet_target_layer(model)) as cam:
            return cam(x, class_idx=class_idx)[0]
    if clave.startswith(("vit_small", "vit_base")):
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


#: Cómo se ve cada etapa en el stand: (nombre corto, consejo, fondo, tinta).
#: Los fondos siguen la cáscara de la Hass: de verde a casi negro.
ETAPA = {
    1: ("Todavía dura", "Déjala unos días a temperatura ambiente.", "#8DB04A", "#1E2618"),
    2: ("Casi lista", "Le faltan dos o tres días.", "#5F8530", "#F4F7E6"),
    3: ("Lista para comer", "Está en su punto: ábrela hoy o mañana.", "#3F5F2A", "#F4F7E6"),
    4: ("Bien madura", "Cómela hoy, es su último buen día.", "#3B2F2C", "#F1E9DF"),
    5: ("Pasada", "Ya pasó su punto de consumo.", "#22191C", "#E9DFD8"),
}


def regla(esperado: float | None) -> str:
    """Regla de madurez: la cáscara de verde a negro y el carozo marcando la etapa."""
    carozo = ""
    if esperado is not None:
        pos = (min(max(esperado, 1.0), 5.0) - 1) / 4 * 100
        carozo = f'<span class="carozo" style="left:{pos:.1f}%"></span>'
    return (f'<div class="regla"><div class="cascara">{carozo}</div>'
            '<div class="extremos"><span>dura</span><span>pasada</span></div></div>')


def tarjeta_vacia() -> str:
    return ('<div class="resultado vacio">'
            '<p class="etiqueta">Esperando una palta</p>'
            '<p class="nombre">Muéstrale una palta a la cámara o sube una foto</p>'
            f'{regla(None)}'
            '<p class="consejo">Funciona mejor con una sola palta, sobre fondo blanco '
            'y con buena luz.</p></div>')


def tarjeta(pred: int, probs: np.ndarray, esperado: float, modelo: str,
            duda_con: int | None, aviso: str = "") -> str:
    nombre, consejo, fondo, tinta = ETAPA[pred]
    barras = "".join(
        f'<div class="barra{" elegida" if i + 1 == pred else ""}">'
        f'<div class="pista"><span class="relleno" style="height:{max(p * 100, 3):.0f}%"></span></div>'
        f'<span class="pct">{i + 1} · {p:.0%}</span></div>'
        for i, p in enumerate(probs))
    duda = ""
    if duda_con is not None:
        duda = (f'<p class="duda">Duda entre «{ETAPA[pred][0].lower()}» y '
                f'«{ETAPA[duda_con][0].lower()}». Mejor que la revise una persona.</p>')
    extra = f'<p class="duda">{html.escape(aviso)}</p>' if aviso else ""
    return (f'<div class="resultado" style="--fondo:{fondo};--tinta:{tinta}">'
            f'<p class="etiqueta">Etapa {pred} de 5 · {CLASS_NAMES[pred]}</p>'
            f'<p class="nombre">{nombre}</p>'
            f'<p class="consejo">{consejo}</p>'
            f'{regla(esperado)}{duda}{extra}'
            f'<div class="barras" aria-label="Probabilidad por etapa">{barras}</div>'
            f'<p class="pie">Seguridad {probs[pred - 1]:.1%} · {html.escape(modelo)}</p>'
            '</div>')


ENCABEZADO = """
<header class="marca">
  <h1>Palt<span>IA</span></h1>
  <p>¿Está lista tu palta? Una foto y te decimos si ya está para el pan.</p>
</header>
"""

COMO_FUNCIONA = """
Una red neuronal aprendió con **14.710 fotos de 478 paltas Hass**, fotografiadas día a día
desde que estaban duras hasta pasarse. Mira el color y la textura de la cáscara: no toca la
fruta ni ve por dentro.

Acierta la etapa exacta unas **3 de cada 4 veces**, y cuando falla casi siempre es por una
sola etapa. Aprendió con fotos de laboratorio, así que funciona mejor con **una sola palta,
centrada, sobre fondo blanco y con buena luz**.

El mapa muestra en rojo las zonas de la foto que más pesaron en la decisión.
"""

PIE = ("Aprendizaje Automático 2 · PUCV · Cristóbal Martínez, Benjamín Espinoza, "
       "Felipe Alvarez, Felipe Torres y Felipe Rojo")

FUENTES = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;'
    '12..96,800&family=Atkinson+Hyperlegible:wght@400;700&family=JetBrains+Mono:wght@500'
    '&display=swap" rel="stylesheet">')

CSS = r"""
:root, .dark {
  --pulpa: #EDF1D6; --hueso: #FBFBF3; --cascara: #1E2618; --hoja: #3F5F2A;
  --carozo: #8A5A3B; --linea: #CBD3A8;
  --titulo: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
  --cuerpo: "Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif;
  --datos: "JetBrains Mono", Consolas, monospace;
}
body, gradio-app, .gradio-container { background: var(--pulpa) !important; color: var(--cascara); }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important;
  font-family: var(--cuerpo) !important; }
.gradio-container .prose, .gradio-container label, .gradio-container button {
  font-family: var(--cuerpo) !important; }

.marca { padding: 28px 4px 8px; }
.marca h1 { font-family: var(--titulo); font-weight: 800; font-size: clamp(56px, 8vw, 104px);
  line-height: .9; letter-spacing: -.04em; margin: 0; color: var(--cascara); }
.marca h1 span { color: var(--hoja); }
.marca p { font-size: clamp(18px, 2vw, 24px); margin: 12px 0 0; max-width: 36ch; color: #3A4630; }

#boton-clasificar { font-size: 20px !important; padding: 16px !important; border-radius: 999px !important;
  background: var(--cascara) !important; color: var(--pulpa) !important; border: 0 !important; }
#boton-clasificar:hover { background: var(--hoja) !important; }
#boton-clasificar:focus-visible { outline: 3px solid var(--carozo); outline-offset: 3px; }

.resultado { --fondo: var(--hueso); --tinta: var(--cascara);
  background: var(--fondo); color: var(--tinta); border-radius: 28px; padding: 28px 30px 22px;
  min-height: 420px;
  display: flex; flex-direction: column; }
.resultado.vacio { border: 2px dashed var(--linea); }
.resultado.vacio .consejo { margin-top: 18px; font-size: 18px; }
button[role="tab"] { font-size: 17px !important; padding: 10px 16px !important; }
.resultado p { margin: 0; color: inherit; }
.resultado .etiqueta { font-family: var(--datos); font-size: 14px; letter-spacing: .04em;
  text-transform: uppercase; opacity: .75; }
.resultado .nombre { font-family: var(--titulo); font-weight: 800; font-size: clamp(44px, 5.4vw, 76px);
  line-height: .95; letter-spacing: -.03em; margin: 10px 0 12px; }
.resultado.vacio .nombre { font-size: clamp(28px, 3vw, 38px); font-weight: 600; letter-spacing: -.01em; }
.resultado .consejo { font-size: 22px; line-height: 1.3; max-width: 30ch; }
.resultado .duda { margin-top: 14px; font-size: 16px; padding: 10px 14px; border-radius: 12px;
  background: color-mix(in srgb, var(--tinta) 14%, transparent); }
.resultado .pie { margin-top: auto; padding-top: 18px; font-family: var(--datos); font-size: 13px;
  opacity: .65; }

.regla { margin: 26px 0 4px; }
.cascara { position: relative; height: 22px; border-radius: 999px;
  background: linear-gradient(90deg, #8DB04A, #5F8530 25%, #3F5F2A 50%, #3B2F2C 75%, #22191C);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--tinta) 25%, transparent); }
.carozo { position: absolute; top: 50%; width: 38px; height: 46px; transform: translate(-50%, -50%);
  border-radius: 50% 50% 46% 46% / 56% 56% 44% 44%;
  background: radial-gradient(circle at 35% 30%, #C99A6E, var(--carozo) 60%, #5E3A22);
  border: 4px solid #F3E9C9; box-shadow: 0 6px 14px rgba(0,0,0,.35);
  transition: left .7s cubic-bezier(.2,.8,.2,1); }
.resultado .extremos span { color: var(--tinta) !important; }
#ejemplos { margin-top: 12px; }
#ejemplos .label, #ejemplos label { font-size: 15px !important; }
.extremos { display: flex; justify-content: space-between; margin-top: 10px; font-size: 14px; opacity: .7; }

.barras { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 22px; }
.barra { display: flex; flex-direction: column; gap: 6px; }
.pista { height: 64px; display: flex; align-items: flex-end; border-radius: 10px; overflow: hidden;
  background: color-mix(in srgb, var(--tinta) 12%, transparent); }
.relleno { display: block; width: 100%; background: var(--tinta); opacity: .45;
  border-radius: 8px 8px 0 0; transition: height .5s ease; }
.barra.elegida .relleno { opacity: 1; }
.resultado .pct { font-family: var(--datos); font-size: 13px; text-align: center; color: var(--tinta); opacity: .8; }
.barra.elegida .pct { opacity: 1; font-weight: 700; }

.entrada .block, .salida .block { border-radius: 22px !important; }
.como-funciona { font-size: 17px; }
.pie-equipo { text-align: center; font-size: 14px; color: #55613F; padding: 18px 0 8px; }

@media (max-width: 760px) {
  .resultado { padding: 22px 18px; min-height: 0; }
  .resultado .consejo { font-size: 18px; }
}
@media (prefers-reduced-motion: reduce) {
  .carozo, .relleno { transition: none; }
}
"""

_C = {
    "body_background_fill": "#EDF1D6", "body_text_color": "#1E2618",
    "body_text_color_subdued": "#55613F", "block_background_fill": "#FBFBF3",
    "block_border_color": "#CBD3A8", "block_label_background_fill": "#EDF1D6",
    "block_label_text_color": "#3F5F2A", "block_title_text_color": "#1E2618",
    "background_fill_primary": "#FBFBF3", "background_fill_secondary": "#EDF1D6",
    "border_color_primary": "#CBD3A8", "input_background_fill": "#FBFBF3",
    "button_primary_background_fill": "#1E2618", "button_primary_text_color": "#EDF1D6",
    "checkbox_background_color_selected": "#3F5F2A", "color_accent_soft": "#DDE6BE",
}
#: Mismo aspecto con el sistema en modo claro u oscuro: la pantalla del stand no cambia.
TEMA = gr.themes.Base(
    primary_hue=gr.themes.colors.lime, neutral_hue=gr.themes.colors.stone, radius_size="lg",
    font=["Atkinson Hyperlegible", "Segoe UI", "system-ui", "sans-serif"],
    font_mono=["JetBrains Mono", "Consolas", "monospace"],
).set(color_accent="#3F5F2A", **_C, **{f"{k}_dark": v for k, v in _C.items()})

#: Modelo que se muestra primero si sus pesos están en checkpoints/.
PREFERIDOS = ("vit_base_app_s2_v6", "resnet50_app_s1_v2", "resnet50_app", "resnet50")


def construir_interfaz() -> gr.Blocks:
    entrenados = modelos_entrenados()
    if not entrenados:
        raise SystemExit(
            "Falta el modelo entrenado en checkpoints/. Usa el ZIP completo de la demo, "
            "que incluye resnet50_app.pt; no necesitas reentrenar."
        )
    entrenados.sort(key=lambda k: PREFERIDOS.index(k) if k in PREFERIDOS else len(PREFERIDOS))
    opciones = [DISPONIBLES[k][1] for k in entrenados]

    with gr.Blocks(title="PaltIA · madurez de paltas", analytics_enabled=False) as demo:
        gr.HTML(ENCABEZADO)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="entrada"):
                with gr.Tab("Foto"):
                    img = gr.Image(type="pil", sources=["upload"], label="Tu palta", height=380)
                    btn = gr.Button("Ver madurez", variant="primary", elem_id="boton-clasificar")
                    ejemplos = ejemplos_de_test()
                    if ejemplos:
                        gr.Examples(
                            examples=[[e] for e in ejemplos],
                            inputs=[img],
                            label="¿No trajiste palta? Prueba con una de estas (la IA nunca las vio)",
                            examples_per_page=10, elem_id="ejemplos",
                        )
                with gr.Tab("Cámara en vivo"):
                    camara = gr.Image(
                        type="pil", sources=["webcam"], streaming=True,
                        webcam_options=gr.WebcamOptions(mirror=False, constraints={
                            "video": {"facingMode": {"ideal": "environment"}}}),
                        label="Cámara", height=380)
                    gr.Markdown("Activa la cámara y apunta a una palta: el resultado se "
                                "actualiza cada segundo.")
            with gr.Column(scale=6, elem_classes="salida"):
                salida_txt = gr.HTML(tarjeta_vacia())
                overlay = gr.Image(label="Dónde miró la IA", height=300,
                                   format="png", buttons=["download", "fullscreen"],
                                   interactive=False)

        with gr.Accordion("Cómo funciona", open=False):
            gr.Markdown(COMO_FUNCIONA, elem_classes="como-funciona")
            modelo = gr.Radio(opciones, value=opciones[0], label="Modelo",
                              visible=len(opciones) > 1)
            mapa_chk = gr.Checkbox(True, label="Mostrar dónde miró la IA")
        gr.HTML(f'<p class="pie-equipo">{PIE}</p>')

        def a_pantalla(imagen, modelo_elegido, mostrar):
            if imagen is None:
                return tarjeta_vacia(), None
            _, mapa, texto = predecir(imagen, modelo_elegido, mostrar)
            return texto, mapa

        gr.on([btn.click, img.change], a_pantalla,
              [img, modelo, mapa_chk], [salida_txt, overlay],
              queue=False, trigger_mode="always_last")
        camara.stream(a_pantalla, [camara, modelo, mapa_chk], [salida_txt, overlay],
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
                               server_name=args.host, pwa=True, theme=TEMA, css=CSS, head=FUENTES,
                               inbrowser=args.open_browser)
