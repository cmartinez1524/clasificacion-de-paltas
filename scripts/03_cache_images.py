"""Pre-reescala el dataset a 256x256 en data/cache/.

Motivacion: entrenamos a 224 px, pero cada epoca decodifica 14.710 JPEG de
800x800 y los reduce en CPU. En una RTX 4070 Laptop la GPU termina esperando al
dataloader. Reescalar una sola vez a 256 px (el lado justo que necesita el
RandomResizedCrop/CenterCrop de 224) elimina ese cuello de botella y baja el
dataset de ~485 MB a ~60 MB, lo que ademas permite que quede en la cache de
disco del sistema operativo.

El cache es un artefacto derivado: esta en .gitignore y se regenera con este
script.

Uso:
    python scripts/03_cache_images.py [--size 256] [--workers 12]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.paths import CACHE_DIR, IMAGES_DIR, ensure_dirs  # noqa: E402
from paltas.console import use_utf8  # noqa: E402

use_utf8()


def _resize_one(args: tuple[Path, Path, int]) -> int:
    src, dst, size = args
    if dst.exists():
        return 0
    with Image.open(src) as im:
        im = im.convert("RGB").resize((size, size), Image.LANCZOS)
        im.save(dst, "JPEG", quality=92, optimize=True)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tareas = [(p, CACHE_DIR / p.name, args.size) for p in sorted(IMAGES_DIR.glob("*.jpg"))]
    print(f"{len(tareas)} imagenes -> {args.size}x{args.size} en {CACHE_DIR}")

    t0 = time.time()
    hechas = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_resize_one, tareas, chunksize=64), 1):
            hechas += r
            if i % 2000 == 0:
                print(f"  {i}/{len(tareas)}  ({time.time() - t0:.0f} s)")

    mb_src = sum(p.stat().st_size for p in IMAGES_DIR.glob("*.jpg")) / 1e6
    mb_dst = sum(p.stat().st_size for p in CACHE_DIR.glob("*.jpg")) / 1e6
    print(f"\nNuevas: {hechas} | ya existian: {len(tareas) - hechas}")
    print(f"Tamano: {mb_src:.0f} MB -> {mb_dst:.0f} MB  ({mb_src / max(mb_dst, 1):.1f}x menos)")
    print(f"Tiempo: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
