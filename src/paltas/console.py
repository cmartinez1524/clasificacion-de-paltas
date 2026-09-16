"""Salida de consola en UTF-8.

La consola de Windows usa cp1252 por defecto, que no puede codificar los simbolos
que usamos en los reportes (Delta, +/-, comillas tipograficas). Sin esto, un
script que termino de calcular correctamente igual se cae al imprimir.
"""

from __future__ import annotations

import sys


def use_utf8() -> None:
    """Reconfigura stdout/stderr a UTF-8, degradando a '?' si algo no se puede."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
