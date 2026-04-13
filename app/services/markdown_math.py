"""Markdown → HTML for lecture outputs, with light normalization for math-heavy text."""

from __future__ import annotations

import re

import markdown

_MD_EXTENSIONS = ["fenced_code", "tables"]

# Models sometimes emit an extra backslash before common LaTeX macros (\\mathbb vs \mathbb).
_KNOWN_TEX_MACROS = (
    "mathbb",
    "mathrm",
    "mathcal",
    "mathit",
    "mathbf",
    "mathsf",
    "mathtt",
    "frac",
    "sqrt",
    "binom",
    "sum",
    "int",
    "infty",
    "cap",
    "cup",
    "subseteq",
    "supseteq",
    "in",
    "notin",
    "forall",
    "exists",
    "neg",
    "land",
    "lor",
    "rightarrow",
    "Rightarrow",
    "Leftrightarrow",
    "cdots",
    "ldots",
    "vdots",
    "ddots",
    "times",
    "cdot",
    "div",
    "pm",
    "mp",
    "leq",
    "geq",
    "neq",
    "approx",
    "equiv",
    "sim",
    "partial",
    "nabla",
    "ell",
    "hbar",
    "quad",
    "qquad",
    "text",
    "operatorname",
    "binom",
    "vec",
    "hat",
    "bar",
    "tilde",
    "overline",
    "underline",
    "left",
    "right",
    "middle",
    "big",
    "Big",
    "bigg",
    "Bigg",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "lambda",
    "mu",
    "pi",
    "sigma",
    "omega",
    "Omega",
    "Delta",
    "Sigma",
    "Pi",
)

_DOUBLE_BACKSLASH_MACRO = re.compile(
    r"\\\\(" + "|".join(re.escape(m) for m in _KNOWN_TEX_MACROS) + r")\b"
)


def normalize_lecture_markdown(raw: str) -> str:
    """Strip BOM, normalize newlines, and fix doubled backslashes before known macros."""
    if not raw:
        return raw
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    text = _DOUBLE_BACKSLASH_MACRO.sub(r"\\\1", text)
    return text


def markdown_to_lecture_html(md: str) -> str:
    """Convert stored Markdown to HTML; math stays as $...$ / $$...$$ for KaTeX in the browser."""
    normalized = normalize_lecture_markdown(md)
    return markdown.markdown(normalized, extensions=_MD_EXTENSIONS)
