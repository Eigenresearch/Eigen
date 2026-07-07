"""Preprocess paper.md for pandoc LaTeX PDF build — Eigen 2.7 Meridian."""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "paper.md"
OUT = ROOT / "paper_build.md"
ABSTRACT_TEX = ROOT / "abstract_block.tex"
FIG = ROOT / "figures"

FIGURE_MAP = {
    f"Figure {i}": f"fig{i}_" for i in range(1, 16)
}
# Build full mapping by scanning actual files
import os
for f in os.listdir(FIG):
    if f.endswith(".png"):
        for i in range(1, 16):
            if f.startswith(f"fig{i}_"):
                FIGURE_MAP[f"Figure {i}"] = f

def extract_abstract(text):
    m = re.search(r"## Abstract\s*\n\n([\s\S]*?)\n\n## ", text)
    if not m: raise ValueError("Abstract not found")
    return m.group(1).strip(), text[m.end():].lstrip()

def strip_front_matter(text):
    idx = text.find("## Abstract")
    if idx == -1: raise ValueError("## Abstract not found")
    return text[idx:]

def replace_figure_blocks(text):
    pattern = re.compile(
        r"\*\*(Figure[^*]+)\*\*[^\n]*(?:\n(?!\*\*Figure)[^\n]*)*\n\n```(?:python|mermaid)[\s\S]*?```",
        re.MULTILINE)
    def repl(m):
        caption = m.group(1).strip()
        for fig_key in sorted(FIGURE_MAP.keys(), key=len, reverse=True):
            if caption.startswith(fig_key):
                path = f"figures/{FIGURE_MAP[fig_key]}"
                return f"\n\n![{caption}]({path})\n\n"
        return m.group(0)
    return pattern.sub(repl, text)

def fix_display_math(text):
    def clean(m):
        lines = [ln.rstrip() for ln in m.group(1).splitlines() if ln.strip()]
        return "$$\n" + "\n".join(lines) + "\n$$"
    return re.sub(r"\$\$([\s\S]*?)\$\$", clean, text)

def latex_escape(s):
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    out = s
    for k, v in repl.items(): out = out.replace(k, v)
    return out

def write_abstract_tex(abstract):
    parts = [latex_escape(p.strip()) + "\n\n" for p in abstract.split("\n\n") if p.strip()]
    ABSTRACT_TEX.write_text(
        "\\begin{quote}\n\\small\n\\justifying\n" + "".join(parts) + "\\end{quote}\n",
        encoding="utf-8")

def build():
    raw = SRC.read_text(encoding="utf-8")
    raw = strip_front_matter(raw)
    abstract, body = extract_abstract(raw)
    write_abstract_tex(abstract)
    body = replace_figure_blocks(body)
    body = fix_display_math(body)
    OUT.write_text(body, encoding="utf-8")
    print(f"Wrote {OUT} ({len(OUT.read_text(encoding='utf-8'))} chars)")

if __name__ == "__main__":
    build()
