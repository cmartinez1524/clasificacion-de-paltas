"""Compila los .tex y mapea cada aviso de desborde al titulo de su slide.

Sin esto hay que cruzar a mano numeros de linea contra `\\begin{frame}`, que es
lento y propenso a error. Es una herramienta de desarrollo, no parte del
pipeline de entregables.

Uso:
    python scripts/_check_overfull.py <ruta-a-tectonic.exe>
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DECKS = [
    RAIZ / "entregables" / "avance" / "presentacion_avance.tex",
    RAIZ / "entregables" / "final" / "presentacion_final.tex",
]


def titulos_por_linea(tex: Path) -> list[tuple[int, str]]:
    """(linea_de_inicio, titulo) de cada frame, en orden."""
    out = []
    for i, ln in enumerate(tex.read_text(encoding="utf-8").split("\n"), 1):
        m = re.match(r"\\begin\{frame\}\{(.*)", ln)
        if m:
            out.append((i, m.group(1)[:58]))
        elif ln.startswith(r"\begin{frame}[plain]"):
            out.append((i, "(portada / separador)"))
    return out


def frame_de(linea: int, indice: list[tuple[int, str]]) -> str:
    anterior = "(preambulo)"
    for ini, tit in indice:
        if ini > linea:
            break
        anterior = tit
    return anterior


def main() -> None:
    tectonic = sys.argv[1] if len(sys.argv) > 1 else "tectonic"
    problemas = 0
    for tex in DECKS:
        if not tex.exists():
            print(f"falta {tex}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([tectonic, "-X", "compile", tex.name, "--outdir", tmp],
                               cwd=tex.parent, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        salida = (r.stdout or "") + (r.stderr or "")
        avisos = sorted(set(re.findall(
            r"^warning: .*?:(\d+): Overfull \\([hv])box \(([\d.]+)pt", salida, re.M)))
        errores = [l for l in salida.split("\n") if l.startswith("error:")]

        indice = titulos_por_linea(tex)
        print(f"\n=== {tex.name} ===")
        if errores:
            problemas += len(errores)
            for e in errores:
                print(f"  ERROR  {e}")
        if not avisos:
            print("  sin desbordes")
        for linea, tipo, pts in avisos:
            pts_f = float(pts)
            marca = "  " if pts_f < 5 else "!!"
            if pts_f >= 5:
                problemas += 1
            print(f"  {marca} {tipo}box {pts_f:6.2f}pt  linea {linea:>4}  "
                  f"{frame_de(int(linea), indice)}")
    print(f"\nDesbordes relevantes (>=5pt) o errores: {problemas}")


if __name__ == "__main__":
    main()
