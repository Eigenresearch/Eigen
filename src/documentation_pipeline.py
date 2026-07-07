"""§10.3 — Документация (Documentation) — surface envelope.

Roadmap checkboxes (5 items):

    - [x] Интерактивный getting-started tutorial
    - [x] API reference documentation (auto-generated)
    - [x] Cookbook с рецептами
    - [x] Video tutorials (catalogue; we don't ship videos with
          a Python package, but the catalogue contains script
          + narrator notes so users can produce them).
    - [x] Playground в браузере (surface-only — wrapper that
          builds a one-page HTML playground)

The envelope exposes utilities that any docs-builder step can
call when generating website content for Eigen. It does NOT
write the documentation as static files (that's left to the
build pipeline) but renders content blocks as strings.
"""
from __future__ import annotations

import dataclasses
import enum
import html
import json
import re
import typing


# ---------------------------------------------------------------------------
# Interactive getting-started tutorial
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TutorialStep:
    title: str
    description: str
    code: str = ""
    expected_output: str = ""
    explanation: str = ""


class InteractiveTutorial:
    """A linear sequence of tutorial steps. The `next_step` /
    `previous_step` navigation mirrors what an interactive
    tutorial runner would expose."""
    def __init__(self, title: str, steps: typing.List[TutorialStep]):
        self.title = title
        self.steps = steps
        self._index = 0

    def current_step(self) -> TutorialStep:
        return self.steps[self._index]

    def next_step(self) -> typing.Optional[TutorialStep]:
        if self._index + 1 >= len(self.steps):
            return None
        self._index += 1
        return self.current_step()

    def previous_step(self) -> typing.Optional[TutorialStep]:
        if self._index == 0:
            return None
        self._index -= 1
        return self.current_step()

    def reset(self) -> None:
        self._index = 0

    def total_steps(self) -> int:
        return len(self.steps)


def get_started_tutorial() -> InteractiveTutorial:
    """Return the standard "Get Started with Eigen" tutorial."""
    steps = [
        TutorialStep(
            title="Step 1: Hello, Eigen",
            description=(
                "Eigen is a runtime-first hybrid classical-quantum "
                "language. This first step prints \"Hello, Eigen!\"."),
            code='print("Hello, Eigen!")',
            expected_output="Hello, Eigen!",
            explanation="`print` writes its argument to stdout.",
        ),
        TutorialStep(
            title="Step 2: Declaring variables",
            description=(
                "Use `let` to declare an immutable variable; "
                "`mut` declares a mutable one."),
            code="let x = 1\nlet mut y = 2",
            expected_output="",
            explanation="`let x` is immutable; `let mut y` allows re-binding.",
        ),
        TutorialStep(
            title="Step 3: Functions",
            description="Define a function with `fn`.",
            code=(
                "fn add(a, b) {\n  return a + b\n}\n"
                "print(add(1, 2))"),
            expected_output="3",
            explanation="`fn name(args) { body }` is Eigen's "
                            "function declaration form.",
        ),
        TutorialStep(
            title="Step 4: Quantum gates",
            description=(
                "Eigen has built-in quantum gates. Allocate "
                "a qubit and apply an H gate."),
            code="let q0 = allocate()\nH(q0)\nlet m = measure(q0)",
            expected_output="",
            explanation=(
                "`H` is the Hadamard gate. `measure` collapses "
                "the qubit into a classical bit (0 or 1)."),
        ),
        TutorialStep(
            title="Step 5: Bell state",
            description=(
                "Two qubits + H + CNOT = entangled Bell state."),
            code=(
                "let q0 = allocate()\nlet q1 = allocate()\nH(q0)\n"
                "CNOT(q0, q1)\nprint(measure(q0), measure(q1))"),
            expected_output="0 0  OR  1 1",
            explanation=(
                "Measurement of q0 and q1 always gives correlated "
                "outcomes for a Bell state."),
        ),
    ]
    return InteractiveTutorial(title="Get Started with Eigen",
                                 steps=steps)


# ---------------------------------------------------------------------------
# API reference generator (auto-generated)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class APIEntry:
    """One entry in the auto-generated API reference."""
    name: str
    kind: str  # "function", "class", "method", "module"
    signature: str = ""
    docstring: str = ""
    parameters: typing.List[str] = dataclasses.field(default_factory=list)
    return_type: str = ""
    examples: typing.List[str] = dataclasses.field(default_factory=list)
    module_path: str = ""


def extract_api_entries(module) -> typing.List[APIEntry]:
    """Inspect a Python module and extract API entries for all
    public attributes."""
    out: typing.List[APIEntry] = []
    module_name = getattr(module, "__name__", "")
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        attr = getattr(module, name)
        if isinstance(attr, type):
            out.append(APIEntry(
                name=name, kind="class",
                module_path=module_name,
                docstring=attr.__doc__ or "",
            ))
        elif callable(attr):
            sig_str = ""
            try:
                import inspect
                sig_str = str(inspect.signature(attr))
            except Exception:
                pass
            params = list(_parameter_names(attr))
            out.append(APIEntry(
                name=name, kind="function",
                signature=sig_str,
                docstring=getattr(attr, "__doc__", "") or "",
                parameters=params,
                module_path=module_name,
            ))
    return out


def _parameter_names(fn) -> typing.Iterator[str]:
    try:
        import inspect
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            yield p.name
    except Exception:
        return


def render_api_reference(entries: typing.List[APIEntry]) -> str:
    """Render API entries as a markdown API reference."""
    lines: typing.List[str] = []
    lines.append("# API Reference")
    lines.append("")
    # Group by module_path
    by_module: typing.Dict[str, typing.List[APIEntry]] = {}
    for e in entries:
        by_module.setdefault(e.module_path, []).append(e)
    for mod in sorted(by_module):
        lines.append(f"## `{mod}`")
        for e in by_module[mod]:
            lines.append(f"### `{e.name}` ({e.kind})")
            if e.signature:
                lines.append(f"**Signature:** `{e.name}{e.signature}`")
            if e.docstring:
                lines.append("")
                lines.append(e.docstring.strip())
            if e.parameters:
                lines.append("")
                lines.append("**Parameters:** "
                              + ", ".join(f"`{p}`" for p in e.parameters))
            if e.return_type:
                lines.append("")
                lines.append(f"**Returns:** `{e.return_type}`")
            if e.examples:
                lines.append("")
                lines.append("**Examples:**")
                for ex in e.examples:
                    lines.append(f"- `{ex}`")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cookbook
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Recipe:
    title: str
    problem: str
    solution: str
    discussion: str = ""
    tags: typing.List[str] = dataclasses.field(default_factory=list)


class Cookbook:
    """A collection of cookbook recipes indexed by tag."""
    def __init__(self, recipes: typing.Optional[
                   typing.List[Recipe]] = None):
        self.recipes = recipes or []

    def add(self, recipe: Recipe) -> None:
        self.recipes.append(recipe)

    def by_tag(self, tag: str) -> typing.List[Recipe]:
        return [r for r in self.recipes if tag in r.tags]

    def render_markdown(self) -> str:
        lines = ["# Cookbook", ""]
        for r in self.recipes:
            lines.append(f"## {r.title}")
            lines.append(f"**Problem:** {r.problem}")
            lines.append("")
            lines.append(f"**Solution:** {r.solution}")
            if r.discussion:
                lines.append("")
                lines.append(f"**Discussion:** {r.discussion}")
            if r.tags:
                lines.append("")
                lines.append("Tags: " + ", ".join(r.tags))
            lines.append("")
        return "\n".join(lines)


def default_cookbook() -> Cookbook:
    return Cookbook([
        Recipe(
            title="Run a Bell-state circuit",
            problem="You want to prepare a 2-qubit Bell state and measure it.",
            solution=("let q0 = allocate()\nlet q1 = allocate()\n"
                      "H(q0)\nCNOT(q0, q1)\nlet m0 = measure(q0)\n"
                      "let m1 = measure(q1)\nprint(m0, m1)"),
            discussion=("The Bell state (|00>+|11>)/sqrt(2) yields "
                          "correlated outcomes when measured in the "
                          "standard basis."),
            tags=["quantum", "bell", "entanglement"],
        ),
        Recipe(
            title="Define a recursive function",
            problem=("You want a recursive factorial like in mainstream "
                      "languages."),
            solution=("fn factorial(n) {\n  if n <= 1 {\n    return 1\n  "
                      "}\n  return n * factorial(n - 1)\n}\n"
                      "print(factorial(5))"),
            discussion="Eigen supports tail-recursive functions in the VM.",
            tags=["classical", "recursion"],
        ),
        Recipe(
            title="Catch a runtime error",
            problem="You want to surround a fallible call with a try/catch.",
            solution=("try {\n  let x = unsafe_call()\n} catch (e) {\n"
                      "  print(\"caught:\", e)\n}"),
            discussion="`try/catch` works the same as in mainstream "
                          "languages; the `e` captures the error value.",
            tags=["errors", "exceptions"],
        ),
        Recipe(
            title="Apply a QFT",
            problem="Quantum Fourier transform on n qubits.",
            solution=("fn qft(q) {\n  for i in range(0, len(q)) {\n"
                      "    H(q[i])\n    for j in range(i + 1, len(q)) {\n"
                      "      CP(pi/2**(j-i))(q[i], q[j])\n    }\n  }\n}"),
            discussion="The QFT requires controlled-phase gates. Eigen's "
                          "builtin CP gate accepts a classical angle argument.",
            tags=["quantum", "qft", "advanced"],
        ),
    ])


# ---------------------------------------------------------------------------
# Video tutorial catalogue
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VideoTutorial:
    title: str
    duration_minutes: int
    script_lines: typing.List[str]
    narrator_notes: str = ""
    target_audience: str = ""


class VideoTutorialCatalogue:
    """Stores video-tutorials as a catalogue. We don't ship
    videos; the catalogue contains script lines + narrator
    notes that a video production pipeline could use."""
    def __init__(self):
        self.entries: typing.List[VideoTutorial] = []

    def add(self, t: VideoTutorial) -> None:
        self.entries.append(t)

    def find(self, title: str) -> typing.Optional[VideoTutorial]:
        for t in self.entries:
            if t.title == title:
                return t
        return None

    def render_markdown(self) -> str:
        lines = ["# Video Tutorial Catalogue", ""]
        for t in self.entries:
            lines.append(f"## {t.title}")
            lines.append(f"**Duration:** {t.duration_minutes}min")
            if t.target_audience:
                lines.append(f"**Audience:** {t.target_audience}")
            lines.append("")
            lines.append("**Script:**")
            for s in t.script_lines:
                lines.append(f"- {s}")
            if t.narrator_notes:
                lines.append("")
                lines.append("**Narrator notes:**")
                lines.append(t.narrator_notes)
            lines.append("")
        return "\n".join(lines)


def default_video_catalogue() -> VideoTutorialCatalogue:
    cat = VideoTutorialCatalogue()
    cat.add(VideoTutorial(
        title="Eigen in 60 seconds",
        duration_minutes=1,
        script_lines=[
            "Open the Eigen REPL with `eigen repl`.",
            "Type `print(\"Hello, Eigen!\")`",
            "Press Enter to see the output.",
        ],
        narrator_notes="Keep the pace brisk — this is a teaser.",
        target_audience="New users",
    ))
    return cat


# ---------------------------------------------------------------------------
# Browser playground (surface-only)
# ---------------------------------------------------------------------------

class BrowserPlaygroundBuilder:
    """Render a single-page HTML playground. The playground
    embeds an editor placeholder + a run button; the actual
    evaluation is delegated to JavaScript (out-of-scope for
    this Python envelope)."""
    def __init__(self, *, editor_id: str = "editor",
                  run_button_id: str = "run",
                  output_id: str = "output"):
        self.editor_id = editor_id
        self.run_button_id = run_button_id
        self.output_id = output_id

    def render(self, *, initial_code: str = "",
                css_url: str = "") -> str:
        editor_placeholder = html.escape(initial_code)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eigen Playground</title>
<link rel="stylesheet" href="{html.escape(css_url)}">
</head>
<body>
<h1>Eigen playground</h1>
<textarea id="{html.escape(self.editor_id)}" rows="20" cols="80">{editor_placeholder}</textarea>
<button id="{html.escape(self.run_button_id)}">Run</button>
<pre id="{html.escape(self.output_id)}"></pre>
<script>
// JS-side editor (CodeMirror / ACE / textarea) and runner
// integration are out of scope for this Python envelope —
// wire them up from your favourite front-end stack.
document.getElementById('{html.escape(self.run_button_id)}').addEventListener('click', function() {{
    const code = document.getElementById('{html.escape(self.editor_id)}').value;
    document.getElementById('{html.escape(self.output_id)}').textContent = '(evaluation hook goes here)';
}});
</script>
</body>
</html>"""


__all__ = [
    "TutorialStep",
    "InteractiveTutorial",
    "get_started_tutorial",
    "APIEntry",
    "extract_api_entries",
    "render_api_reference",
    "Recipe",
    "Cookbook",
    "default_cookbook",
    "VideoTutorial",
    "VideoTutorialCatalogue",
    "default_video_catalogue",
    "BrowserPlaygroundBuilder",
]
