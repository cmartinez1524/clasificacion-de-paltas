"""Constructores de presentaciones Beamer (LaTeX).

Misma filosofia que `deck.py` para .pptx: el .tex se genera desde los JSON de
metricas, nunca se escribe un numero a mano. La diferencia es que el .tex
resultante es texto plano legible y editable, asi que despues se puede retocar
a mano sin volver a pasar por Python.

El preambulo usa solo paquetes que vienen en cualquier distribucion TeX y en
Overleaf: sin metropolis, sin fuentes externas, sin minted. Es una eleccion
deliberada -- un deck que no compila en la maquina del profesor no sirve.
"""

from __future__ import annotations

PREAMBULO = r"""% ---------------------------------------------------------------------------
%  GENERADO AUTOMATICAMENTE por scripts/07_deck_latex.py
%  Las cifras provienen de reports/metrics/*.json. Si reentrenas un modelo,
%  regenera este archivo en vez de editar los numeros a mano:
%      python scripts/07_deck_latex.py
%  (Igual puedes editarlo a mano: es LaTeX normal.)
% ---------------------------------------------------------------------------
\documentclass[aspectratio=169,11pt,table]{beamer}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{textcomp}
\usepackage{amssymb}

\graphicspath{{figuras/}}

% --- Paleta del proyecto ---------------------------------------------------
\definecolor{verdeosc}{HTML}{2E6B2F}
\definecolor{verdecla}{HTML}{8FB339}
\definecolor{grisTexto}{HTML}{444A52}
\definecolor{grisSuave}{HTML}{8A9098}
\definecolor{grisFondo}{HTML}{F6F7F4}
\definecolor{azul}{HTML}{1F6FB2}
\definecolor{rojo}{HTML}{D1495B}
\definecolor{morado}{HTML}{6A4C93}

% --- Tema ------------------------------------------------------------------
\usetheme{default}
\usefonttheme{professionalfonts}
\setbeamertemplate{navigation symbols}{}
\setbeamercolor{structure}{fg=verdeosc}
\setbeamercolor{normal text}{fg=grisTexto}
\setbeamercolor{frametitle}{fg=verdeosc}
\setbeamercolor{itemize item}{fg=verdecla}
\setbeamercolor{itemize subitem}{fg=verdecla}
\setbeamerfont{frametitle}{size=\Large,series=\bfseries}
\setbeamerfont{framesubtitle}{size=\scriptsize}
\setbeamertemplate{itemize item}{\raisebox{0.15ex}{\tiny$\blacksquare$}}
\setbeamertemplate{itemize subitem}{\raisebox{0.15ex}{\tiny$\square$}}

% Usa \par + \vspace en vez de \\ a proposito: con un subtitulo vacio, un \\
% despues de un grupo vacio provoca "There's no line here to end".
\setbeamertemplate{frametitle}{%
  \vspace*{0.35em}%
  \begin{beamercolorbox}[wd=\paperwidth,leftskip=0.7cm,rightskip=0.7cm]{frametitle}%
    \setlength{\parskip}{0pt}\setlength{\parindent}{0pt}%
    {\usebeamerfont{frametitle}\insertframetitle}\par
    {\usebeamerfont{framesubtitle}\color{grisSuave}\insertframesubtitle}\par
    \vspace{0.2em}%
    {\color{verdecla}\rule{1.7em}{2pt}}%
  \end{beamercolorbox}%
}

\setbeamertemplate{footline}{%
  \hfill{\tiny\color{grisSuave}\insertframenumber\,/\,\inserttotalframenumber}%
  \hspace{0.7cm}\vspace{0.25cm}%
}

% --- Atajos ----------------------------------------------------------------
% Linea de cierre al pie de un slide: el "take-away".
\newcommand{\cierre}[1]{%
  \vfill
  {\color{verdecla}\rule{2.5pt}{0.9em}}\hspace{0.4em}%
  {\footnotesize\color{grisSuave}#1}%
}
% Tarjeta tipo KPI: un numero grande con etiqueta.
\newcommand{\kpi}[4][verdeosc]{%
  \begin{minipage}[t]{#2}%
    \setlength{\fboxsep}{6pt}%
    \colorbox{grisFondo}{\begin{minipage}[t]{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}%
      {\tiny\color{grisSuave}\bfseries #3}\\[0.12em]%
      % \large y no \Large: con cinco tarjetas en una fila, un valor como
      % "99,2 %" en \Large desborda la caja.
      {\large\bfseries\color{#1}#4}%
    \end{minipage}}%
  \end{minipage}%
}
\newcommand{\sig}{\textcolor{verdeosc}{\textbf{significativa}}}
\newcommand{\nosig}{\textcolor{rojo}{\textbf{NO significativa}}}

\renewcommand{\arraystretch}{1.25}
\setlength{\tabcolsep}{5pt}

% Descomenta para imprimir con notas del orador:
% \setbeameroption{show notes}
"""


def portada(titulo: str, subtitulo: str, autor: str, fecha: str, pie: str) -> str:
    return rf"""
{{
\setbeamercolor{{background canvas}}{{bg=verdeosc}}
\begin{{frame}}[plain]
  \vspace{{1.4cm}}
  % linespread: a \Huge el interlineado por defecto deja los descendentes de una
  % linea tocando la siguiente.
  {{\color{{white}}\Huge\bfseries\linespread{{1.15}}\selectfont {titulo}}}

  \vspace{{0.55cm}}
  {{\color{{verdecla}}\rule{{\textwidth}}{{1.6pt}}}}

  \vspace{{0.55cm}}
  {{\color{{white!85!verdecla}}\large {subtitulo}}}

  \vfill
  {{\color{{white!70!verdecla}}\small {autor} \quad\textbullet\quad {fecha}}}

  \vspace{{0.2em}}
  {{\color{{white!60!verdecla}}\footnotesize {pie}}}
  \vspace{{0.6cm}}
\end{{frame}}
}}
"""


def seccion(numero: str, titulo: str) -> str:
    return rf"""
{{
\setbeamercolor{{background canvas}}{{bg=grisFondo}}
\begin{{frame}}[plain]
  \vfill
  \hspace{{0.6cm}}{{\color{{verdecla}}\fontsize{{54}}{{58}}\selectfont\bfseries {numero}}}%
  \hspace{{0.5cm}}{{\color{{verdeosc}}\Huge\bfseries {titulo}}}
  \vfill
\end{{frame}}
}}
"""


def frame(titulo: str, subtitulo: str, cuerpo: str, nota: str = "") -> str:
    """Un slide de contenido. `cuerpo` es LaTeX crudo."""
    n = f"\n\\note{{{nota}}}" if nota else ""
    return rf"""
\begin{{frame}}{{{titulo}}}{{{subtitulo}}}
{cuerpo}
\end{{frame}}{n}
"""


def items(lista, tam: str = r"\normalsize", sep: str = "0.35em") -> str:
    """Lista con vinetas. Cada elemento: str, o (texto, True) para negrita."""
    filas = []
    for it in lista:
        texto, negrita = it if isinstance(it, tuple) else (it, False)
        filas.append(rf"  \item {{\bfseries {texto}}}" if negrita else rf"  \item {texto}")
    cuerpo = "\n".join(filas)
    return (f"{{{tam}\n"
            f"\\setlength{{\\itemsep}}{{{sep}}}\n"
            f"\\begin{{itemize}}\n{cuerpo}\n\\end{{itemize}}\n}}")


def tabla(encabezados, filas, spec: str, tam: str = r"\small",
          destacar: int | None = None) -> str:
    """Tabla con encabezado verde. `destacar` resalta una fila (0-indexada)."""
    # \rowcolor debe ser lo primero de la fila; el color de texto va celda a celda.
    head = " & ".join(rf"\color{{white}}\bfseries {h}" for h in encabezados)
    cuerpo = [r"\rowcolor{verdeosc}" + head + r" \\"]
    for k, f in enumerate(filas):
        pre = r"\rowcolor{verdecla!22}" if destacar == k else ""
        cuerpo.append(pre + " & ".join(str(c) for c in f) + r" \\")
    return (f"{{{tam}\n\\centering\n"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
            + "\n\\midrule\n".join([cuerpo[0], "\n".join(cuerpo[1:])])
            + "\n\\bottomrule\n\\end{tabular}\n}")


def kpis(tarjetas, ancho: str = "") -> str:
    """Fila de tarjetas KPI. Cada una: (etiqueta, valor, color).

    El ancho deja holgura para los \\hfill entre tarjetas y para el redondeo de
    \\fboxsep; con el reparto exacto la linea quedaba desbordada.
    """
    n = len(tarjetas)
    w = ancho or f"{(0.94 - 0.015 * (n - 1)) / n:.3f}" + r"\textwidth"
    partes = [rf"\kpi[{c}]{{{w}}}{{{lbl}}}{{{val}}}" for lbl, val, c in tarjetas]
    return "\\hfill\n".join(partes)


def nombre_seguro(nombre: str) -> str:
    """Nombre de figura apto para \\includegraphics.

    LaTeX lee el argumento de \\includegraphics en modo texto, donde el guion
    bajo es el operador de subindice: `\\includegraphics{a_b}` falla con
    "Missing $ inserted". Copiamos las figuras con guion medio para evitarlo,
    en vez de depender de grffile o de \\detokenize.
    """
    return nombre.replace("_", "-")


def figura(nombre: str, ancho: str = r"0.85\textwidth", alto: str = "") -> str:
    opts = f"width={ancho}" + (f",height={alto}" if alto else "") + ",keepaspectratio"
    return (f"\\begin{{center}}\n  \\includegraphics[{opts}]"
            f"{{{nombre_seguro(nombre)}}}\n\\end{{center}}")


def columnas(izq: str, der: str, w_izq: str = "0.48", w_der: str = "0.48") -> str:
    return (f"\\begin{{columns}}[T]\n"
            f"\\begin{{column}}{{{w_izq}\\textwidth}}\n{izq}\n\\end{{column}}\n"
            f"\\begin{{column}}{{{w_der}\\textwidth}}\n{der}\n\\end{{column}}\n"
            f"\\end{{columns}}")


#: Caracteres que se ven bien en el editor pero se imprimen mal segun el motor
#: (pdfLaTeX/XeTeX) y la codificacion. Se reemplazan por su comando LaTeX.
CARACTERES_RIESGOSOS = {
    "«": r"\guillemotleft{}",
    "»": r"\guillemotright{}",
    "¿": r"\textquestiondown{}",
    "¡": r"\textexclamdown{}",
    "‘": "`", "’": "'",
    "“": "``", "”": "''",
    "…": r"\ldots{}",
}


def validar_tex(tex: str) -> None:
    """Falla si quedo un caracter que no imprime igual en todos los motores.

    Las vocales acentuadas y la enye funcionan bien via utf8+T1, pero los
    guillemets y los signos de apertura invertidos no: se imprimian como
    'ń', 'ż' y '£'. Es un error silencioso --compila sin quejarse y sale mal en
    el PDF-- asi que lo convertimos en un fallo ruidoso.
    """
    encontrados = {c: CARACTERES_RIESGOSOS[c] for c in CARACTERES_RIESGOSOS if c in tex}
    if encontrados:
        detalle = ", ".join(f"{c!r} -> usa {cmd}" for c, cmd in encontrados.items())
        raise ValueError(f"Caracteres que no imprimen bien en LaTeX: {detalle}")


def documento(preambulo_extra: str, titulo_pdf: str, autor: str, cuerpo: str) -> str:
    validar_tex(cuerpo + titulo_pdf + autor)
    return (PREAMBULO
            + preambulo_extra
            + f"\n\\title{{{titulo_pdf}}}\n\\author{{{autor}}}\n"
            + "\n\\begin{document}\n"
            + cuerpo
            + "\n\\end{document}\n")


# --- Utilidades de formato numerico (convencion es-CL: coma decimal) --------
def num(x: float, dec: int = 4) -> str:
    return f"{x:.{dec}f}".replace(".", "{,}")


def pct(x: float, dec: int = 1) -> str:
    return f"{100 * x:.{dec}f}".replace(".", "{,}") + r"\,\%"


def signo(x: float, dec: int = 4) -> str:
    return f"{x:+.{dec}f}".replace(".", "{,}").replace("-", r"$-$")
