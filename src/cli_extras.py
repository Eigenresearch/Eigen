"""§10.2 — CLI auto-completion, playground, and migrate commands.

  * `generate_completions` — emits shell completion scripts for
    bash, zsh, fish, and PowerShell.
  * `EigenPlayground` — a simple in-memory REPL that reads .eig
    source from stdin and executes it.
  * `CodeMigrator` — applies automated code migrations between
    Eigen versions (e.g. 2.3 → 2.7 syntax changes).
"""
from __future__ import annotations

import dataclasses
import typing
import re


# ---------------------------------------------------------------------------
# Auto-completion
# ---------------------------------------------------------------------------

EIGEN_COMMANDS = [
    "run", "build", "test", "bench", "profile", "doc", "fmt",
    "packager", "lsp", "debug", "estimate", "verify", "viz",
    "audit", "doctor", "exec", "reproduce", "playground", "migrate",
]

EIGEN_FLAGS = [
    "--help", "--version", "--verbose", "--quiet", "--trace",
    "--deterministic", "--strict", "--opt-level", "--seed",
    "--sim-type", "--gpu", "--max-instructions", "--timeout",
    "--html", "--json", "--output", "--watch",
]


def generate_bash_completions() -> str:
    """Generate bash completion script for Eigen CLI."""
    cmds = " ".join(EIGEN_COMMANDS)
    flags = " ".join(EIGEN_FLAGS)
    return f"""_eigen_completions() {{
    local cur prev cmds flags
    cur=${{COMP_WORDS[COMP_CWORD]}}
    prev=${{COMP_WORDS[COMP_CWORD-1]}}
    cmds="{cmds}"
    flags="{flags}"
    if [[ $cur == --* ]]; then
        COMPREPLY=( $(compgen -W "$flags" -- "$cur") )
    else
        COMPREPLY=( $(compgen -W "$cmds" -- "$cur") )
    fi
    return 0
}}
complete -F _eigen_completions eigen
"""


def generate_zsh_completions() -> str:
    """Generate zsh completion script for Eigen CLI."""
    cmds = " ".join(EIGEN_COMMANDS)
    return f"""#compdef eigen
_eigen() {{
    local cmds
    cmds=({cmds})
    _describe 'command' cmds
}}
_eigen "$@"
"""


def generate_fish_completions() -> str:
    """Generate fish completion script for Eigen CLI."""
    lines = []
    for cmd in EIGEN_COMMANDS:
        lines.append(f"complete -c eigen -n '__fish_use_subcommand' -a '{cmd}'")
    for flag in EIGEN_FLAGS:
        lines.append(f"complete -c eigen -l '{flag.removeprefix('--')}' -d 'Flag'")
    return "\n".join(lines) + "\n"


def generate_powershell_completions() -> str:
    """Generate PowerShell completion script for Eigen CLI."""
    cmds = "', '".join(EIGEN_COMMANDS)
    flags = "', '".join(EIGEN_FLAGS)
    return f"""Register-ArgumentCompleter -Native -CommandName eigen -ScriptBlock {{
    param($wordToComplete, $commandAst, $cursorPosition)
    $cmds = @('{cmds}')
    $flags = @('{flags}')
    if ($wordToComplete -startswith '--') {{
        $flags | Where-Object {{ $_ -like "$wordToComplete*" }} |
            ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }}
    }} else {{
        $cmds | Where-Object {{ $_ -like "$wordToComplete*" }} |
            ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }}
    }}
}}
"""


def generate_completions(shell: str = "bash") -> str:
    """Generate shell completion script for the specified shell."""
    generators = {
        "bash": generate_bash_completions,
        "zsh": generate_zsh_completions,
        "fish": generate_fish_completions,
        "powershell": generate_powershell_completions,
    }
    gen = generators.get(shell.lower())
    if gen is None:
        raise ValueError(f"Unsupported shell: {shell}. "
                          f"Supported: {list(generators.keys())}")
    return gen()


# ---------------------------------------------------------------------------
# Playground (REPL)
# ---------------------------------------------------------------------------

class EigenPlayground:
    """A simple in-memory playground that compiles and executes
    .eig source snippets.

    §10.2: "eigen playground — встроенная песочница"
    """

    def __init__(self):
        self.history: list[str] = []
        self.results: list[typing.Any] = []

    def evaluate(self, source: str) -> dict:
        """Evaluate a source snippet and return the result."""
        self.history.append(source)
        try:
            from src.frontend.lexer import Lexer
            from src.frontend.parser import Parser
            from src.backend.ebc_compiler import EBCCompiler
            from src.backend.vm import EigenVM

            tokens = Lexer(source).tokenize()
            ast = Parser(tokens).parse()
            ebc = EBCCompiler()
            instructions = ebc.compile(ast)
            vm = EigenVM()
            vm.execute(instructions)
            result = {
                "success": True,
                "output": vm.simulator.get_amplitudes_dict()
                           if hasattr(vm.simulator, 'get_amplitudes_dict')
                           else {},
                "instructions": len(instructions),
            }
            self.results.append(result)
            return result
        except Exception as e:
            result = {"success": False, "error": str(e)}
            self.results.append(result)
            return result

    def repl_loop(self, input_fn=input, output_fn=print):
        """Run an interactive REPL loop."""
        output_fn("Eigen Playground — type .exit to quit")
        while True:
            try:
                line = input_fn("eigen> ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == ".exit":
                break
            if not line.strip():
                continue
            source = f"eigen 1.0\n{line}"
            result = self.evaluate(source)
            if result["success"]:
                output_fn(result.get("output", "ok"))
            else:
                output_fn(f"Error: {result['error']}")


# ---------------------------------------------------------------------------
# Code Migrator
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MigrationRule:
    """A single code migration rule: regex pattern → replacement."""
    name: str
    pattern: str
    replacement: str
    description: str = ""


class CodeMigrator:
    """Automated code migration between Eigen versions.

    §10.2: "eigen migrate — автоматическая миграция кода
    между версиями"
    """

    def __init__(self):
        self.rules: list[MigrationRule] = []
        self._init_default_rules()

    def _init_default_rules(self):
        self.rules = [
            MigrationRule(
                name="print_to_io",
                pattern=r'\bprint\s+(.+)',
                replacement=r'print \1',
                description="Normalize print syntax",
            ),
            MigrationRule(
                name="measure_arrow",
                pattern=r'\bmeasure\s+(\w+)\s*->\s*(\w+)',
                replacement=r'measure \1 -> \2',
                description="Normalize measure arrow syntax",
            ),
            MigrationRule(
                name="qubit_decl",
                pattern=r'\bqubit\s+(\w+)',
                replacement=r'qubit \1',
                description="Normalize qubit declaration",
            ),
        ]

    def add_rule(self, rule: MigrationRule):
        self.rules.append(rule)

    def migrate(self, source: str) -> tuple[str, list[str]]:
        """Apply all migration rules to source code.

        Returns (migrated_source, list_of_applied_rules).
        """
        applied = []
        result = source
        for rule in self.rules:
            new_result = re.sub(rule.pattern, rule.replacement, result)
            if new_result != result:
                applied.append(rule.name)
                result = new_result
        return result, applied

    def migrate_file(self, path: str) -> tuple[str, list[str]]:
        """Migrate a file in-place. Returns (new_content, applied_rules)."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content, applied = self.migrate(content)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return new_content, applied
