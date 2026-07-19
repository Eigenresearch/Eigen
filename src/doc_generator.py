import re
import html as _html

_DECL_KEYWORDS = ("qfunc", "func", "struct", "enum", "trait", "impl", "type")

_KIND_TITLES = {
    "qfunc": "Quantum Subroutine",
    "func": "Classical Function",
    "struct": "Structure",
    "enum": "Enumeration",
    "trait": "Trait",
    "impl": "Implementation Block",
    "type": "Type Alias",
}


class _Declaration:
    __slots__ = ("kind", "name", "signature", "comments")

    def __init__(self, kind, name, signature, comments):
        self.kind = kind
        self.name = name
        self.signature = signature
        self.comments = comments


def _anchor(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '-', name.lower()).strip('-') or "decl"


class EigenDocGenerator:
    def __init__(self, output_format: str = "markdown"):
        if output_format not in ("markdown", "html"):
            raise ValueError(f"Unknown output format: {output_format!r}")
        self.output_format = output_format

    def generate_docs(self, source: str, filepath: str) -> str:
        filename = filepath.replace('\\', '/').split('/')[-1]
        decls = self._parse_declarations(source)
        names = sorted({d.name for d in decls if d.name},
                          key=len, reverse=True)
        if self.output_format == "html":
            return self._render_html(filename, decls, names)
        return self._render_markdown(filename, decls, names)

    def _parse_declarations(self, source: str):
        lines = source.splitlines()
        decls = []
        comments = []
        i = 0
        n = len(lines)
        while i < n:
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if stripped.startswith('#') or stripped.startswith('//'):
                comment_text = re.sub(r'^(#|//)\s*', '', stripped)
                comments.append(comment_text)
                i += 1
                continue
            matched_kind = None
            match_name = None
            for kw in _DECL_KEYWORDS:
                m = re.match(rf'^{kw}\s+(\w+)', stripped)
                if m:
                    matched_kind = kw
                    match_name = m.group(1)
                    break
            if matched_kind is None:
                comments = []
                i += 1
                continue
            sig_parts = [stripped]
            while not self._sig_complete(sig_parts, matched_kind):
                nxt_i = i + 1
                if nxt_i >= n:
                    break
                nxt = lines[nxt_i].strip()
                if not nxt:
                    break
                if nxt == '}' or nxt.startswith('}'):
                    break
                if any(re.match(rf'^{k}\s+\w+', nxt) for k in _DECL_KEYWORDS):
                    break
                sig_parts.append(nxt)
                i = nxt_i
            signature = " ".join(sig_parts)
            decls.append(_Declaration(matched_kind, match_name,
                                        signature, list(comments)))
            comments = []
            i += 1
        return decls

    @staticmethod
    def _sig_complete(sig_parts, kind):
        joined = " ".join(sig_parts).strip()
        if '{' in joined or ';' in joined:
            return True
        if kind == 'type' and '=' in joined:
            after = joined.split('=', 1)[1].strip()
            if after:
                return True
        return False

    def _cross_reference(self, text, names):
        if not names or not text:
            return text
        name_set = set(names)
        if self.output_format == "html":
            text = _html.escape(text)
            pattern = re.compile(r'(?<!&)\b[A-Za-z_][A-Za-z0-9_]*\b(?!;)')

            def repl(m):
                w = m.group(0)
                if w in name_set:
                    return f'<a href="#{_anchor(w)}">{w}</a>'
                return w
            return pattern.sub(repl, text)
        pattern = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')

        def repl(m):
            w = m.group(0)
            if w in name_set:
                return f'[{w}](#{_anchor(w)})'
            return w
        return pattern.sub(repl, text)

    def _see_also(self, decl, names):
        refs = [nm for nm in names
                if nm != decl.name and re.search(
                    rf'\b{re.escape(nm)}\b', decl.signature)]
        return refs

    def _render_markdown(self, filename, decls, names):
        md_lines = [
            f"# API Reference: {filename}",
            "",
            "Generated automatically from Eigen source code comments.",
            "",
        ]
        for d in decls:
            title = _KIND_TITLES.get(d.kind, "Declaration")
            md_lines.append(f"## {title} `{d.name}`")
            md_lines.append("")
            md_lines.append("```eigen")
            md_lines.append(d.signature)
            md_lines.append("```")
            md_lines.append("")
            if d.comments:
                md_lines.append(self._cross_reference("\n".join(d.comments),
                                                        names))
            else:
                md_lines.append("*No description available.*")
            md_lines.append("")
            refs = self._see_also(d, names)
            if refs:
                md_lines.append(
                    "**See also:** "
                    + ", ".join(f"[{r}](#{_anchor(r)})" for r in refs))
                md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
        return "\n".join(md_lines)

    def _render_html(self, filename, decls, names):
        esc = _html.escape
        parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            f"<title>API Reference: {esc(filename)}</title>",
            "</head>",
            "<body>",
            f"<h1>API Reference: {esc(filename)}</h1>",
            "<p>Generated automatically from Eigen source code comments.</p>",
        ]
        for d in decls:
            title = _KIND_TITLES.get(d.kind, "Declaration")
            parts.append(
                f'<h2 id="{_anchor(d.name)}">{esc(title)} '
                f'<code>{esc(d.name)}</code></h2>')
            parts.append("<pre><code>" + esc(d.signature) + "</code></pre>")
            if d.comments:
                desc = self._cross_reference("\n".join(d.comments), names)
                parts.append("<p>" + desc.replace("\n", "<br>") + "</p>")
            else:
                parts.append("<p><em>No description available.</em></p>")
            refs = self._see_also(d, names)
            if refs:
                links = ", ".join(
                    f'<a href="#{_anchor(r)}">{esc(r)}</a>' for r in refs)
                parts.append(f"<p><strong>See also:</strong> {links}</p>")
            parts.append("<hr>")
        parts.append("</body>")
        parts.append("</html>")
        return "\n".join(parts)
