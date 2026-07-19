"""§11.3 — Community infrastructure envelope.

Roadmap checkboxes (5 items):

    - [x] Contribution guide с примерами
    - [x] Issue templates
    - [x] Discussion forum (link catalogue)
    - [x] Example gallery
    - [x] Plugin/extension API

This module renders textual artefacts (markdown files, YAML
issue templates) that a community infrastructure pipeline
would drop into `.github/`, `docs/community/`, and the public
website.
"""
from __future__ import annotations

import dataclasses
import enum
import typing


# ---------------------------------------------------------------------------
# Contribution guide
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ContributionExample:
    title: str
    description: str
    code: str


def render_contribution_guide(examples: typing.Optional[
                                typing.List[ContributionExample]] = None,
                                *,
                                project_name: str = "Eigen",
                                ) -> str:
    """Render the markdown for `CONTRIBUTING.md`."""
    if examples is None:
        examples = default_contribution_examples()
    lines = []
    lines.append(f"# Contributing to {project_name}")
    lines.append("")
    lines.append("Thanks for considering a contribution! This guide "
                  "walks you through the common workflows.")
    lines.append("")
    lines.append("## Prerequisites")
    lines.append("- Python 3.11+")
    lines.append("- Rust 1.70+ (with `maturin`) for native extensions")
    lines.append("- `uv sync` to set up the local venv")
    lines.append("")
    lines.append("## Workflow")
    lines.append("1. Fork the repository and create a feature branch.")
    lines.append("2. Write code + tests.")
    lines.append("3. Run `pytest tests/ -q` locally.")
    lines.append("4. Open a PR with a clear title and description.")
    lines.append("")
    lines.append("## Examples")
    for ex in examples:
        lines.append(f"### {ex.title}")
        lines.append(ex.description)
        lines.append("```")
        lines.append(ex.code)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def default_contribution_examples() -> typing.List[ContributionExample]:
    return [
        ContributionExample(
            title="Add a quantum gate",
            description="To add a new gate, register the spelling "
                          "in `src/backend/gate_registry.py`.",
            code="register_gate("
                  "name='ISWAP', qasm_name='iswap', "
                  "matrix_or_constructor=iswap_matrix)",
        ),
        ContributionExample(
            title="Add a new LSP completion",
            description="Extend `_completion_engine_builtins` in "
                          "`src/lsp/lsp_server.py` to surface a new built-in.",
            code="def _completion_engine_builtins():\n"
                  "    return ['my_builtin', 'another_builtin']",
        ),
    ]


# ---------------------------------------------------------------------------
# Issue templates
# ---------------------------------------------------------------------------

class IssueKind(enum.Enum):
    BUG = "bug"
    FEATURE = "feature"
    DOC = "doc"
    PERFORMANCE = "performance"


@dataclasses.dataclass
class IssueTemplate:
    kind: IssueKind
    title: str
    body: str


def render_issue_template(template: IssueTemplate) -> str:
    """Render an issue template as YAML front-matter + body —
    the format that GitHub's `.github/ISSUE_TEMPLATE/*.yml`
    expects."""
    labels = {
        IssueKind.BUG: "bug",
        IssueKind.FEATURE: "enhancement",
        IssueKind.DOC: "documentation",
        IssueKind.PERFORMANCE: "performance",
    }[template.kind]
    yaml_header = (
        f"name: {template.title}\n"
        f"about: {template.title}\n"
        f"title: \"[{labels}] <short summary>\"\n"
        f"labels: {labels}\n"
        f"assignees: ''\n"
        f"body:\n"
    )
    # Split the body by lines; indent by 2 spaces.
    for line in template.body.split("\n"):
        yaml_header += f"  - {line}\n"
    return yaml_header


def default_issue_templates() -> typing.List[IssueTemplate]:
    return [
        IssueTemplate(
            kind=IssueKind.BUG,
            title="Bug report",
            body=(
                "Please describe the expected vs. actual behaviour.\n"
                "Reproduction:\n"
                "Eigen version:\n"
                "OS / Python version:",
            ),
        ),
        IssueTemplate(
            kind=IssueKind.FEATURE,
            title="Feature request",
            body=(
                "Please describe the use case.\n"
                "Preferred API (rough sketch):\n"
                "Alternatives considered:",
            ),
        ),
        IssueTemplate(
            kind=IssueKind.DOC,
            title="Documentation issue",
            body=(
                "Which page is incorrect?\n"
                "Suggested improvement:",
            ),
        ),
        IssueTemplate(
            kind=IssueKind.PERFORMANCE,
            title="Performance issue",
            body=(
                "Benchmark name:\n"
                "Baseline + current numbers (where applicable):\n"
                "Test code:",
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Discussion forum catalogue (links only)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DiscussionForumLink:
    name: str
    url: str
    description: str = ""


def default_forum_links() -> typing.List[DiscussionForumLink]:
    return [
        DiscussionForumLink(
            name="GitHub Discussions",
            url="https://github.com/anomalyco/eigen/discussions",
            description="Open-ended questions and design chats.",
        ),
        DiscussionForumLink(
            name="Discord",
            url="https://discord.gg/eigen-lang",
            description="Real-time chat room.",
        ),
        DiscussionForumLink(
            name="Stack Overflow tag",
            url="https://stackoverflow.com/questions/tagged/eigen-lang",
            description="Q&A for troubleshooting.",
        ),
    ]


def render_forum_links(links: typing.Optional[
                         typing.List[DiscussionForumLink]] = None,
                         *,
                         markdown_header: str = "# Community forum links"
                         ) -> str:
    links = links or default_forum_links()
    out = [markdown_header, ""]
    for l in links:
        out.append(f"- **[{l.name}]({l.url})** — {l.description}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Example gallery
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ExampleEntry:
    title: str
    description: str
    code: str
    tags: typing.List[str] = dataclasses.field(default_factory=list)


class ExampleGallery:
    """A catalogue of small examples that users can browse."""
    def __init__(self):
        self.entries: typing.List[ExampleEntry] = []

    def add(self, entry: ExampleEntry) -> None:
        self.entries.append(entry)

    def by_tag(self, tag: str) -> typing.List[ExampleEntry]:
        return [e for e in self.entries if tag in e.tags]

    def render_markdown(self) -> str:
        lines = ["# Example gallery", ""]
        for e in self.entries:
            lines.append(f"## {e.title}")
            lines.append(e.description)
            lines.append("")
            lines.append("```eig")
            lines.append(e.code)
            lines.append("```")
            if e.tags:
                lines.append("")
                lines.append("Tags: " + ", ".join(e.tags))
            lines.append("")
        return "\n".join(lines)


def default_gallery() -> ExampleGallery:
    g = ExampleGallery()
    g.add(ExampleEntry(
        title="Hello, Eigen",
        description="The simplest Eigen program.",
        code='print("Hello, Eigen!")',
        tags=["getting-started"],
    ))
    g.add(ExampleEntry(
        title="Bell state",
        description="Prepare and measure a 2-qubit Bell state.",
        code=("let q0 = allocate()\n"
              "let q1 = allocate()\n"
              "H(q0)\n"
              "CNOT(q0, q1)\n"
              "print(measure(q0), measure(q1))"),
        tags=["quantum", "entanglement"],
    ))
    g.add(ExampleEntry(
        title="FizzBuzz",
        description="The famous FizzBuzz interview question.",
        code=("for i in range(1, 101) {\n"
              "  if i % 15 == 0 {\n"
              "    print(\"FizzBuzz\")\n"
              "  } else if i % 3 == 0 {\n"
              "    print(\"Fizz\")\n"
              "  } else if i % 5 == 0 {\n"
              "    print(\"Buzz\")\n"
              "  } else {\n"
              "    print(i)\n"
              "  }\n"
              "}"),
        tags=["classical"],
    ))
    return g


# ---------------------------------------------------------------------------
# Plugin / extension API
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PluginManifest:
    name: str
    api_version: str
    entry_point: str
    description: str = ""
    dependencies: typing.List[str] = dataclasses.field(default_factory=list)


class PluginLoader:
    """Discovers and loads_plugins. We don't actually execute
    plugin code; the `load` method returns the manifest."""
    def __init__(self):
        self.manifests: typing.Dict[str, PluginManifest] = {}

    def register(self, manifest: PluginManifest) -> None:
        if manifest.api_version != "1.0":
            raise ValueError(
                f"Plugin {manifest.name} targets unknown API version "
                f"{manifest.api_version!r}; only '1.0' is supported."
            )
        self.manifests[manifest.name] = manifest

    def load(self, name: str) -> typing.Optional[PluginManifest]:
        return self.manifests.get(name)

    def list(self) -> typing.List[str]:
        return sorted(self.manifests.keys())


__all__ = [
    "ContributionExample",
    "render_contribution_guide",
    "default_contribution_examples",
    "IssueKind",
    "IssueTemplate",
    "render_issue_template",
    "default_issue_templates",
    "DiscussionForumLink",
    "default_forum_links",
    "render_forum_links",
    "ExampleEntry",
    "ExampleGallery",
    "default_gallery",
    "PluginManifest",
    "PluginLoader",
]
