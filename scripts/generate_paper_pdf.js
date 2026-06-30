/**
 * Build paper.pdf via LaTeX (pandoc + pdflatex).
 * Delegates to scripts/build_paper_pdf.py
 */
const { execSync } = require("child_process");
const path = require("path");

const script = path.join(__dirname, "build_paper_pdf.py");
execSync(`python "${script}"`, { stdio: "inherit", cwd: path.join(__dirname, "..") });
