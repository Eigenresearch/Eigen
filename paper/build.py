"""Full LaTeX PDF build: figures + pandoc + xelatex."""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
BUILD_MD = ROOT / "paper_build.md"
OUT_PDF = ROOT / "paper.pdf"
HEADER = ROOT / "tex_header.tex"
BEFORE = ROOT / "title_block.tex"

def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)

def main():
    run([sys.executable, str(ROOT / "prepare_paper_tex.py")])
    pandoc_cmd = [
        "pandoc", str(BUILD_MD), "-o", str(OUT_PDF),
        "--pdf-engine=xelatex", "--number-sections", "--toc", "--toc-depth=2",
        "-V", "documentclass=article", "-V", "fontsize=11pt",
        "-V", "geometry:margin=1in", "-V", "geometry:letterpaper",
        "-V", "linestretch=1.15", "-V", "colorlinks=true",
        "-V", "linkcolor=blue", "-V", "citecolor=green", "-V", "urlcolor=blue",
        "-H", str(HEADER), "--include-before-body", str(BEFORE),
        "--resource-path", f"{ROOT};{FIG}",
    ]
    run(pandoc_cmd)
    size_mb = OUT_PDF.stat().st_size / 1024 / 1024
    print(f"\nDone: {OUT_PDF} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
