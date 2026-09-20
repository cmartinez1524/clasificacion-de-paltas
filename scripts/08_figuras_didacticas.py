"""Figuras explicativas para las presentaciones.

No son figuras de resultados: son figuras para que alguien que no trabaja en
vision por computador entienda (a) en que se diferencian una CNN y un
transformer al mirar una imagen, y (b) por que eso hace que uno dependa mucho
mas del pre-entrenamiento que el otro.

Solo van aca las figuras que son *visuales*. Las comparaciones que son puro
texto (por ejemplo "que sabe cada red antes de empezar") se arman directamente
en el slide con LaTeX: una imagen de texto queda ilegible apenas hay que
reducirla para que entre.

Uso:
    python scripts/08_figuras_didacticas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.console import use_utf8  # noqa: E402
from paltas.paths import (  # noqa: E402
    CACHE_DIR,
    FIGURES_DIR,
    IMAGES_DIR,
    METRICS_DIR,
    SPLITS_CSV,
    ensure_dirs,
)
from paltas.viz import save, set_style  # noqa: E402

use_utf8()

AZUL = "#1f6fb2"
ROJO = "#d1495b"
VERDE = "#2E6B2F"
GRIS = "#8A9098"


def _una_palta(clase: int = 3, seed: int = 4) -> np.ndarray:
    """Una imagen de test de la clase pedida, a 224x224."""
    df = pd.read_csv(SPLITS_CSV)
    sub = df[(df["split"] == "test") & (df["ripening_index"] == clase)]
    fila = sub.iloc[np.random.default_rng(seed).integers(len(sub))]
    carpeta = CACHE_DIR if any(CACHE_DIR.glob("*.jpg")) else IMAGES_DIR
    return np.array(Image.open(carpeta / fila["image_file"]).convert("RGB").resize((224, 224)))


def _bbox_fruta(img: np.ndarray, umbral: int = 205) -> tuple[int, int, int, int]:
    """Caja que encierra la palta. El fondo del dataset es gris claro uniforme.

    Se calcula en vez de codificarse a mano para que la figura siga quedando
    bien si se cambia la imagen de ejemplo.
    """
    oscuro = img.mean(axis=2) < umbral
    ys, xs = np.where(oscuro)
    if len(xs) == 0:
        return 0, 0, img.shape[1], img.shape[0]
    return xs.min(), ys.min(), xs.max(), ys.max()


def fig_como_mira_cada_red() -> None:
    """La figura central: la lupa que recorre vs el rompecabezas que conversa."""
    img = _una_palta()
    fx0, fy0, fx1, fy1 = _bbox_fruta(img)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))

    # ---------------- Panel izquierdo: convolución ----------------
    ax = axes[0]
    ax.imshow(img)
    lado = 38
    # zigzag descendente sobre la fruta (no sobre el fondo), bien separado
    cx = (fx0 + fx1) / 2
    izq, der = cx - lado * 1.05, cx + lado * 0.05
    alturas = np.linspace(fy0 + 6, fy1 - lado - 6, 4)
    posiciones = [(izq if k % 2 == 0 else der, h) for k, h in enumerate(alturas)]
    for k, (x, y) in enumerate(posiciones):
        borde = AZUL if k < len(posiciones) - 1 else "#0d3f6b"
        ancho = 2.0 if k < len(posiciones) - 1 else 3.0
        ax.add_patch(mpatches.Rectangle((x, y), lado, lado, fill=False,
                                        edgecolor=borde, lw=ancho,
                                        linestyle="--" if k < len(posiciones) - 1 else "-"))
    # flechas entre posiciones: la misma ventana se desplaza
    for (ax0, ay0), (ax1, ay1) in zip(posiciones, posiciones[1:]):
        ax.annotate("", xy=(ax1 + lado / 2, ay1 + lado / 2),
                    xytext=(ax0 + lado / 2, ay0 + lado / 2),
                    arrowprops=dict(arrowstyle="->", color=AZUL, lw=1.4, alpha=0.65,
                                    connectionstyle="arc3,rad=0.25"))
    ax.set_title("ResNet-50  ·  una lupa que recorre la fruta", color=AZUL,
                 fontsize=13, fontweight="bold", pad=10)

    # ---------------- Panel derecho: atención ----------------
    ax = axes[1]
    ax.imshow(img)
    n = 7                      # 7x7 para que se vea; el modelo real usa 14x14
    paso = 224 / n
    for i in range(1, n):
        ax.axhline(i * paso, color="white", lw=1.1, alpha=0.85)
        ax.axvline(i * paso, color="white", lw=1.1, alpha=0.85)

    centro = lambda i, j: ((j + 0.5) * paso, (i + 0.5) * paso)  # noqa: E731
    # celdas que caen sobre la fruta, elegidas a partir de su caja real
    ci0, ci1 = int(fy0 // paso), int(fy1 // paso)
    cj0, cj1 = int(fx0 // paso), int(fx1 // paso)
    cim, cjm = (ci0 + ci1) // 2, (cj0 + cj1) // 2
    # en diamante: arriba, izquierda, derecha y abajo de la fruta, para que se
    # vea que dialogan aunque esten lejos
    destacados = [(ci0, cjm), (cim, cj0), (cim, cj1), (ci1, cjm)]
    destacados = [(min(max(i, 0), n - 1), min(max(j, 0), n - 1)) for i, j in destacados]
    for i, j in destacados:
        ax.add_patch(mpatches.Rectangle((j * paso, i * paso), paso, paso, fill=False,
                                        edgecolor=ROJO, lw=2.4))
    # todos con todos, desde la primera capa
    for a in range(len(destacados)):
        for b in range(a + 1, len(destacados)):
            bx0, by0 = centro(*destacados[a])
            bx1, by1 = centro(*destacados[b])
            ax.annotate("", xy=(bx1, by1), xytext=(bx0, by0),
                        arrowprops=dict(arrowstyle="<->", color=ROJO, lw=1.3, alpha=0.75,
                                        connectionstyle="arc3,rad=0.2"))
    ax.set_title("ViT-S/16  ·  un rompecabezas que conversa", color=ROJO,
                 fontsize=13, fontweight="bold", pad=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlim(0, 224)
        ax.set_ylim(224, 0)
        for s in ax.spines.values():
            s.set_visible(False)

    # sin leyendas al pie: el slide las pone como vinetas y no conviene duplicarlas
    fig.subplots_adjust(bottom=0.02, top=0.90, wspace=0.08)
    save(fig, FIGURES_DIR / "30_como_mira_cada_red.png")


def fig_dependencia_pretraining() -> None:
    """Cuánto cae cada red al quitarle ImageNet. Es el resultado clave del informe."""
    datos = {}
    for n in ("resnet50", "vit_small", "resnet50_scratch", "vit_small_scratch"):
        p = METRICS_DIR / f"{n}_test.json"
        if not p.exists():
            print(f"  [aviso] falta {p.name}; omito la figura de dependencia")
            return
        datos[n] = json.loads(p.read_text(encoding="utf-8"))["test"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = [0, 1]
    etiquetas = ["Con ImageNet\n(1,3 millones de fotos\nantes de ver paltas)",
                 "Desde cero\n(solo las 10.208 fotos\nde paltas)"]

    for ax, (met, nombre, fmt) in zip(axes, [("qwk", "QWK", lambda v: f"{v:.3f}".replace(".", ",")),
                                             ("accuracy", "Accuracy",
                                              lambda v: f"{100*v:.1f} %".replace(".", ","))]):
        for base, scratch, color, nom in [("resnet50", "resnet50_scratch", AZUL, "ResNet-50"),
                                          ("vit_small", "vit_small_scratch", ROJO, "ViT-S/16")]:
            y = [datos[base][met], datos[scratch][met]]
            ax.plot(x, y, "-o", color=color, lw=2.6, ms=9, label=nom, zorder=3)
            ax.annotate(fmt(y[0]), (0, y[0]), textcoords="offset points", xytext=(-12, 6),
                        ha="right", fontsize=9.5, color=color, fontweight="bold")
            ax.annotate(fmt(y[1]), (1, y[1]), textcoords="offset points", xytext=(12, -2),
                        ha="left", fontsize=9.5, color=color, fontweight="bold")
            caida = y[0] - y[1]
            # recuadro blanco y desplazamiento amplio: si no, la etiqueta queda
            # encima de la propia linea y no se lee
            ax.annotate(f"$-${fmt(caida).lstrip('-')}", (0.5, (y[0] + y[1]) / 2),
                        textcoords="offset points",
                        xytext=(0, 18 if color == AZUL else -24),
                        ha="center", fontsize=10.5, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_xlim(-0.35, 1.35)
        ax.margins(y=0.16)
        ax.set_title(nombre, fontsize=12)
        ax.grid(axis="x", visible=False)
        if met == "accuracy":     # el eje en % para que hable el mismo idioma
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
        else:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))

    axes[0].legend(frameon=False, loc="lower left", fontsize=10)
    fig.suptitle("Si le quitas ImageNet, el ViT se cae mucho más que la ResNet",
                 fontsize=13, y=1.02)
    save(fig, FIGURES_DIR / "32_dependencia_pretraining.png")


def fig_matrices_didactica() -> None:
    """Las matrices de confusion, pero legibles para alguien que nunca vio una.

    La version tecnica (compare_models.py) muestra las seis matrices sin
    anotaciones. Esta muestra solo las dos principales y encierra la franja
    diagonal +/-1 para que se vea de un golpe que los errores graves no existen.
    """
    datos = {}
    for n in ("resnet50", "vit_small"):
        p = METRICS_DIR / f"{n}_test.json"
        if not p.exists():
            print(f"  [aviso] falta {p.name}; omito la figura de matrices")
            return
        datos[n] = json.loads(p.read_text(encoding="utf-8"))["test"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax, (clave, nombre, color) in zip(axes, [("resnet50", "ResNet-50", AZUL),
                                                 ("vit_small", "ViT-S/16", ROJO)]):
        cm = np.array(datos[clave]["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                if cmn[i, j] > 0.005:
                    ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                            fontsize=10, color="white" if cmn[i, j] > 0.55 else "#222")

        # contorno de la franja diagonal +/-1: donde caen los errores leves
        for i in range(5):
            for j in range(5):
                if abs(i - j) <= 1:
                    ax.add_patch(mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor="#1a7f37", lw=2.4, zorder=4))
        # esquinas: los errores graves
        for (i, j) in [(0, 4), (4, 0)]:
            ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor="#b3261e", lw=2.2,
                                            linestyle="--", zorder=4))

        ax.set_xticks(range(5), range(1, 6))
        ax.set_yticks(range(5), range(1, 6))
        ax.set_xlabel("Lo que dijo el modelo")
        ax.set_ylabel("Lo que la palta era en realidad")
        pm1 = datos[clave]["off_by_one"]
        ax.set_title(f"{nombre}\nacierta o se pasa por uno: {100*pm1:.1f} %".replace(".", ","),
                     color=color, fontsize=12, fontweight="bold")
        ax.grid(False)

    # leyenda al pie y no sobre las celdas: dentro de la matriz tapaba los numeros
    fig.text(0.5, 0.045,
             "marco verde: acertó o se pasó por un solo escalón",
             ha="center", va="center", fontsize=10.5, color="#1a7f37", fontweight="bold")
    fig.text(0.5, 0.005,
             "recuadro rojo punteado: confundir una palta verde con una podrida "
             "— cero casos en las dos redes",
             ha="center", va="center", fontsize=10.5, color="#b3261e", fontweight="bold")

    fig.subplots_adjust(wspace=0.28, top=0.86, bottom=0.17)
    save(fig, FIGURES_DIR / "33_matrices_didactica.png")


def main() -> None:
    ensure_dirs()
    set_style()
    fig_como_mira_cada_red()
    fig_dependencia_pretraining()
    fig_matrices_didactica()
    print("Listo.")


if __name__ == "__main__":
    main()
