"""§10.3 — Documentation: getting-started tutorial,
video tutorials placeholder, browser playground.

  * `generate_tutorial` — produces an interactive getting-started
    tutorial that walks new users through the Eigen language.
  * `VideoTutorialIndex` — index of video tutorials (URLs + topics).
  * `BrowserPlayground` — generates a self-contained HTML page
    that can run Eigen code in a browser via WASM (surface).
"""
from __future__ import annotations

import dataclasses
import typing


@dataclasses.dataclass
class TutorialStep:
    """A single step in the getting-started tutorial."""
    title: str
    description: str
    code_example: str
    expected_output: str = ""


GETTING_STARTED_STEPS = [
    TutorialStep(
        title="Hello World",
        description="Every Eigen program starts with a version header "
                     "and a main function.",
        code_example='eigen 1.0\nfunc main() -> int {\n    print "Hello, World!"\n    return 0\n}',
        expected_output="Hello, World!",
    ),
    TutorialStep(
        title="Quantum Bell State",
        description="Create a Bell state (maximally entangled 2-qubit "
                     "state) using H and CNOT gates.",
        code_example='eigen 1.0\nqubit q0\nqubit q1\nH q0\nCNOT q0, q1\nmeasure q0 -> c0\nmeasure q1 -> c1\nprint c0\nprint c1',
        expected_output="0 0 or 1 1 (correlated)",
    ),
    TutorialStep(
        title="Variables and Arithmetic",
        description="Eigen supports classical variables and arithmetic "
                     "alongside quantum operations.",
        code_example='eigen 1.0\nlet x: int = 10\nlet y: int = 20\nlet z: int = x + y\nprint z',
        expected_output="30",
    ),
    TutorialStep(
        title="Control Flow",
        description="Use if/else and for loops to control program flow.",
        code_example='eigen 1.0\nlet sum: int = 0\nfor i in arr {\n    sum = sum + i\n}\nprint sum',
        expected_output="Sum of array elements",
    ),
    TutorialStep(
        title="Functions",
        description="Define reusable functions with typed parameters.",
        code_example='eigen 1.0\nfunc add(a: int, b: int) -> int {\n    return a + b\n}\nlet result: int = add(5, 3)\nprint result',
        expected_output="8",
    ),
    TutorialStep(
        title="Structs and Enums",
        description="Group data with structs and represent choices with enums.",
        code_example='eigen 1.0\nstruct Point {\n    x: int\n    y: int\n}\nlet p: Point = Point { x: 1, y: 2 }\nprint p.x',
        expected_output="1",
    ),
]


def generate_tutorial(format: str = "markdown") -> str:
    """Generate the getting-started tutorial.

    §10.3: "Интерактивный getting-started tutorial"

    Args:
        format: "markdown" or "html"

    Returns:
        Tutorial text in the specified format.
    """
    if format == "markdown":
        lines = ["# Eigen Getting Started Tutorial", ""]
        for i, step in enumerate(GETTING_STARTED_STEPS, 1):
            lines.append(f"## Step {i}: {step.title}")
            lines.append("")
            lines.append(step.description)
            lines.append("")
            lines.append("```eigen")
            lines.append(step.code_example)
            lines.append("```")
            if step.expected_output:
                lines.append("")
                lines.append(f"**Expected output:** {step.expected_output}")
            lines.append("")
        return "\n".join(lines)

    elif format == "html":
        html_parts = ["<!DOCTYPE html>", "<html>", "<head>",
                        "<title>Eigen Tutorial</title>", "</head>", "<body>",
                        "<h1>Eigen Getting Started Tutorial</h1>"]
        for i, step in enumerate(GETTING_STARTED_STEPS, 1):
            html_parts.append(f"<h2>Step {i}: {step.title}</h2>")
            html_parts.append(f"<p>{step.description}</p>")
            html_parts.append("<pre><code>")
            html_parts.append(step.code_example)
            html_parts.append("</code></pre>")
            if step.expected_output:
                html_parts.append(f"<p><strong>Expected:</strong> "
                                    f"{step.expected_output}</p>")
        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    raise ValueError(f"Unsupported format: {format}")


# ---------------------------------------------------------------------------
# Video tutorials
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VideoTutorial:
    """Index entry for a video tutorial."""
    title: str
    url: str
    duration_minutes: int
    topic: str
    difficulty: str = "beginner"  # beginner/intermediate/advanced


VIDEO_TUTORIALS = [
    VideoTutorial(
        title="Eigen Language Introduction",
        url="https://youtube.com/placeholder/eigen-intro",
        duration_minutes=10,
        topic="basics",
        difficulty="beginner",
    ),
    VideoTutorial(
        title="Quantum Programming with Eigen",
        url="https://youtube.com/placeholder/eigen-quantum",
        duration_minutes=20,
        topic="quantum",
        difficulty="intermediate",
    ),
    VideoTutorial(
        title="Advanced Type System Features",
        url="https://youtube.com/placeholder/eigen-types",
        duration_minutes=15,
        topic="types",
        difficulty="advanced",
    ),
    VideoTutorial(
        title="Building a Bell State Circuit",
        url="https://youtube.com/placeholder/eigen-bell",
        duration_minutes=5,
        topic="quantum",
        difficulty="beginner",
    ),
]


def generate_video_tutorial_index() -> str:
    """Generate a markdown index of video tutorials.

    §10.3: "Video tutorials"
    """
    lines = ["# Eigen Video Tutorials", ""]
    for vid in VIDEO_TUTORIALS:
        lines.append(f"## {vid.title}")
        lines.append(f"- **URL:** {vid.url}")
        lines.append(f"- **Duration:** {vid.duration_minutes} minutes")
        lines.append(f"- **Topic:** {vid.topic}")
        lines.append(f"- **Difficulty:** {vid.difficulty}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Browser playground
# ---------------------------------------------------------------------------

def generate_browser_playground() -> str:
    """Generate a self-contained HTML page for running Eigen
    in the browser via WASM.

    §10.3: "Playground в браузере"

    This is a surface — actual WASM execution requires the
    Eigen WASM module to be compiled and loaded. The generated
    page provides the UI and loads the WASM module.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Eigen Playground</title>
    <style>
        body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
        #editor { width: 100%; height: 300px; background: #2d2d2d; color: #d4d4d4;
                   border: 1px solid #555; padding: 10px; font-size: 14px; }
        #output { width: 100%; height: 200px; background: #2d2d2d; color: #4ec9b0;
                   border: 1px solid #555; padding: 10px; margin-top: 10px;
                   overflow: auto; font-size: 14px; }
        button { margin: 5px; padding: 8px 16px; background: #0e639c;
                  color: white; border: none; cursor: pointer; }
        button:hover { background: #1177bb; }
        .examples { margin: 10px 0; }
        .examples a { color: #569cd6; cursor: pointer; margin-right: 10px; }
    </style>
</head>
<body>
    <h1>Eigen Browser Playground</h1>
    <div class="examples">
        <a onclick="loadExample('bell')">Bell State</a>
        <a onclick="loadExample('hello')">Hello World</a>
        <a onclick="loadExample('arithmetic')">Arithmetic</a>
    </div>
    <textarea id="editor" spellcheck="false">eigen 1.0
qubit q0
qubit q1
H q0
CNOT q0, q1
measure q0 -> c0
measure q1 -> c1
print c0
print c1</textarea>
    <button onclick="runCode()">Run</button>
    <button onclick="clearOutput()">Clear</button>
    <pre id="output"></pre>
    <script>
        // Load Eigen WASM module (surface — requires eigen_wasm.wasm)
        let eigenModule = null;
        // In production, load the WASM module here:
        // fetch('eigen_wasm.wasm').then(...)

        function runCode() {
            const code = document.getElementById('editor').value;
            const output = document.getElementById('output');
            // Surface: actual execution requires WASM module
            output.textContent = 'Output:\\n' + code + '\\n\\n(Requires eigen_wasm.wasm to execute)';
        }

        function clearOutput() {
            document.getElementById('output').textContent = '';
        }

        function loadExample(name) {
            const examples = {
                bell: 'eigen 1.0\\nqubit q0\\nqubit q1\\nH q0\\nCNOT q0, q1\\nmeasure q0 -> c0\\nmeasure q1 -> c1\\nprint c0\\nprint c1',
                hello: 'eigen 1.0\\nfunc main() -> int {\\n    print "Hello, World!"\\n    return 0\\n}',
                arithmetic: 'eigen 1.0\\nlet x: int = 10\\nlet y: int = 20\\nprint x + y',
            };
            document.getElementById('editor').value = examples[name] || '';
        }
    </script>
</body>
</html>"""
