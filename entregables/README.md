# Entregables

Cada presentación existe en tres formatos, generados **desde los mismos JSON de métricas**
(`reports/metrics/`). Ningún número está escrito a mano: si se reentrena un modelo, se
regeneran y las cifras se actualizan solas.

| Archivo | Qué es |
|---|---|
| `presentacion_*.pdf` | **Listo para presentar.** Compilado desde el `.tex` |
| `presentacion_*.tex` | Fuente LaTeX (Beamer). Editable |
| `presentacion_*.pptx` | Versión PowerPoint, por si se prefiere editar ahí |
| `figuras/` | Las imágenes que usa el `.tex`, ya renombradas |

## Cómo compilar el LaTeX

### Opción A — Overleaf (lo más simple)

1. Entrar a [overleaf.com](https://overleaf.com) → *New Project* → *Upload Project*.
2. Comprimir la carpeta (`avance/` o `final/`) completa y subir el `.zip`.
   **Tiene que incluir la subcarpeta `figuras/`**: el `.tex` la referencia con
   `\graphicspath{{figuras/}}`.
3. Overleaf detecta el `.tex` principal y compila con pdfLaTeX. Listo.

### Opción B — local

Necesitas una distribución TeX (MiKTeX en Windows, TeX Live en Linux/macOS):

```bash
cd entregables/avance
pdflatex presentacion_avance.tex
pdflatex presentacion_avance.tex   # dos veces: la segunda fija el total de páginas
```

## Cómo regenerarlas

```bash
python scripts/07_deck_latex.py    # ambas versiones .tex + sus figuras
python scripts/05_deck_avance.py   # .pptx del avance
python scripts/06_deck_final.py    # .pptx del final
```

## Notas sobre el LaTeX

- **Sin paquetes exóticos.** El preámbulo usa solo `graphicx`, `booktabs`, `array`, `xcolor`,
  `textcomp` y `amssymb`, todos presentes en cualquier distribución y en Overleaf. Nada de
  `metropolis`, fuentes externas ni `minted`: un deck que no compila en la máquina del
  profesor no sirve.
- **Notas del orador.** El `.tex` trae la línea `% \setbeameroption{show notes}` comentada al
  final del preámbulo. Descoméntala para imprimir con notas.
- **Colores y estilo** están definidos como comandos al inicio del `.tex`
  (`\cierre`, `\kpi`, `\sig`, `\nosig`), así que se pueden cambiar en un solo lugar.
- **Nombres de figuras con guion medio, no guion bajo.** `\includegraphics{a_b}` falla en
  LaTeX porque el guion bajo es el operador de subíndice; por eso la copia en `figuras/` va
  renombrada.
