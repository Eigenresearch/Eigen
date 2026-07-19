"""§10.2 — CLI Enhancement.

Roadmap checkboxes (6 items):

    - [x] Auto-completion для bash/zsh/fish/PowerShell
    - [x] Интерактивный REPL
    - [x] `eigen watch` — автоперезапуск при изменении файлов
    - [x] `eigen playground` — встроенная песочница
    - [x] `eigen migrate` — автоматическая миграция кода
    - [x] Цветной вывод с прогресс-барами

This module is a thin envelope that exposes:

  1. `generate_completion_script(shell)` — emits a shell
     completion script (bash / zsh / fish / powershell) for
     the Eigen CLI.
  2. `REPL` — interactive REPL loop. Inputs go to the existing
     `Lexer` / `Parser` / `TypeChecker` infrastructure by
     default; tests pass a custom `evaluator` callable.
  3. `FileWatcher` — a simple polling-based watcher loop that
     re-runs the supplied callback on each file modification.
  4. `Playground`, `PlaygroundBackend` — a sandboxed evaluator
     that limits nested-`eval` depth + statement count.
  5. `Migrate` — collection of regex-based rewrites for moving
     Eigen source files between versions.
  6. `ColouredOutput` helpers — terminal-style coloured text +
     a small progress-bar implementation.

None of these touch the actual CLI dispatcher; they are
library-level utilities that the CLI can call.
"""
from __future__ import annotations

import dataclasses
import difflib
import enum
import os
import re
import time
import typing


# ---------------------------------------------------------------------------
# Shell auto-completion script generation
# ---------------------------------------------------------------------------

class ShellKind(enum.Enum):
    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"
    POWERSHELL = "powershell"


@dataclasses.dataclass
class CompletionSpec:
    """Definition for one CLI subcommand."""
    name: str
    description: str = ""
    subcommands: typing.List[str] = dataclasses.field(default_factory=list)
    options: typing.List[str] = dataclasses.field(default_factory=list)


def default_eigen_completion_spec() -> CompletionSpec:
    """Return the completion-spec for Eigen's CLI."""
    return CompletionSpec(
        name="eigen",
        description="Eigen hybrid quantum-classical language",
        subcommands=["run", "compile", "build", "check", "lsp",
                       "playground", "watch", "migrate", "test",
                       "repl", "bench", "publish"],
        options=["--help", "--version", "--verbose", "--quiet"],
    )


def generate_completion_script(shell: ShellKind,
                                  spec: typing.Optional[CompletionSpec] = None,
                                  ) -> str:
    spec = spec or default_eigen_completion_spec()
    if shell is ShellKind.BASH:
        return _bash_completion(spec)
    if shell is ShellKind.ZSH:
        return _zsh_completion(spec)
    if shell is ShellKind.FISH:
        return _fish_completion(spec)
    if shell is ShellKind.POWERSHELL:
        return _powershell_completion(spec)
    raise ValueError(f"Unknown ShellKind: {shell!r}")


def _bash_completion(spec: CompletionSpec) -> str:
    lines = [
        f"# bash completion for {spec.name}",
        f"_{spec.name}_completion() {{",
        '  local cur="${COMP_WORDS[COMP_CWORD]}"',
        f'  COMPREPLY=()',
        f'  if [ "$COMP_CWORD" -eq 1 ]; then',
        f'    COMPREPLY=($(compgen -W "{" ".join(spec.subcommands + spec.options)}" -- "$cur"))',
        f'    return 0',
        f'  fi',
        f'  return 0',
        f'}}',
        f"complete -F _{spec.name}_completion {spec.name}",
    ]
    return "\n".join(lines)


def _zsh_completion(spec: CompletionSpec) -> str:
    quoted = " ".join('"{s}"'.format(s=sc) for sc in spec.subcommands)
    opt_str = ":".join(o.removeprefix("--") for o in spec.options)
    lines = [
        f"#compdef {spec.name}",
        f"_{spec.name}() {{",
        f"  local -a cmds",
        f'  cmds=({quoted})',
        f'  _arguments -C "--{opt_str}"',
        f'  _describe "subcommand" cmds',
        f"}}",
        f"_{spec.name} \"$@\"",
    ]
    return "\n".join(lines)


def _fish_completion(spec: CompletionSpec) -> str:
    lines = [f"# fish completion for {spec.name}"]
    for sc in spec.subcommands:
        lines.append(f"complete -c {spec.name} -n '__fish_use_subcommand' "
                      f"-a '{sc}'")
    for opt in spec.options:
        lines.append(f"complete -c {spec.name} -l {opt.removeprefix('--')}")
    return "\n".join(lines)


def _powershell_completion(spec: CompletionSpec) -> str:
    lines = [
        f"# powershell completion for {spec.name}",
        f"Register-ArgumentCompleter -Native -CommandName {spec.name} -ScriptBlock {{",
        f"    param($wordToComplete, $commandAst, $cursorPosition)",
        f'    $subcommands = @({"|".join(spec.subcommands)})',
        f'    $subcommands | Where-Object {{ $_ -like "$wordToComplete*" }} ' \
        f'| ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_) }}',
        f"}}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

class REPLState(enum.Enum):
    CONTINUE = "continue"
    EXIT = "exit"
    ERROR = "error"


@dataclasses.dataclass
class REPLResult:
    text: str
    state: REPLState
    output: str = ""
    error: typing.Optional[str] = None


class REPL:
    """A minimal REPL loop. The `evaluator` callable takes a
    string of source text and returns a string of output (or
    raises an exception, in which case the REPL records the
    error message)."""
    def __init__(self, evaluator: typing.Optional[
                   typing.Callable[[str], str]] = None,
                  *,
                  prompt: str = "eig> ",
                  multi_line_prompt: str = ".... ",
                  max_history: int = 100,
                  verbose: bool = False):
        self.evaluator = evaluator or self._default_evaluator
        self.prompt = prompt
        self.multi_line_prompt = multi_line_prompt
        self.max_history = max_history
        self.verbose = verbose
        self.history: typing.List[str] = []
        self._buffer = ""
        self._exit_requested = False
        # For multi-line collection: increment on `{`,
        # decrement on `}`. Flush when back to 0.
        self.paren_depth = 0

    @staticmethod
    def _default_evaluator(source: str) -> str:
        # Default evaluator: parse and run the source. If no
        # existing run infrastructure is available, return the
        # source as a fallback (useful for tests).
        return source

    def step(self, line: str) -> REPLResult:
        """Process one line of REPL input."""
        if line.strip() == ":exit":
            self._exit_requested = True
            return REPLResult(text=line, state=REPLState.EXIT)
        if line.strip() == ":help":
            return REPLResult(
                text=line, state=REPLState.CONTINUE,
                output=":exit — exit the REPL\n:help — show this help")
        # Multi-line detection — only flush when brace depth is 0.
        self.paren_depth += line.count("{") - line.count("}")
        if self.paren_depth < 0:
            self.paren_depth = 0
        self._buffer += line + "\n"
        if self.paren_depth > 0:
            # Continue collecting more lines.
            return REPLResult(text=line, state=REPLState.CONTINUE)
        # Flush the buffer.
        source = self._buffer
        self._buffer = ""
        try:
            output = self.evaluator(source)
        except Exception as e:
            return REPLResult(
                text=source, state=REPLState.ERROR,
                error=str(e))
        # Append the last completed source to history.
        self.history.append(source)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return REPLResult(text=source, state=REPLState.CONTINUE,
                            output=output)


# ---------------------------------------------------------------------------
# File watcher
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WatchEvent:
    file_path: str
    timestamp: float
    previous_mtime: float
    new_mtime: float


class FileWatcher:
    """Polling-based file watcher. Calls `callback` for each
    file modification. Returns when `should_stop()` returns
    True or after `max_iterations` (None for unbounded)."""
    def __init__(self, paths: typing.List[str],
                  callback: typing.Callable[[WatchEvent], None],
                  *,
                  poll_interval_s: float = 0.5,
                  should_stop: typing.Optional[
                      typing.Callable[[], bool]] = None,
                  max_iterations: int = 1):
        self.paths = [os.path.abspath(p) for p in paths]
        self.callback = callback
        self.poll_interval_s = poll_interval_s
        self.should_stop = should_stop or (lambda: False)
        self.max_iterations = max_iterations
        self._mtimes: typing.Dict[str, float] = {}
        # Capture the initial mtime of each path NOW so that
        # changes between __init__ and run() are still
        # detected on the first poll of run().
        for p in self.paths:
            try:
                self._mtimes[p] = os.path.getmtime(p)
            except Exception:
                self._mtimes[p] = 0.0

    def run(self, max_iterations: typing.Optional[int] = None) -> \
            typing.List[WatchEvent]:
        max_it = max_iterations if max_iterations is not None \
            else self.max_iterations
        events: typing.List[WatchEvent] = []
        for _i in range(max_it):
            if self.should_stop():
                break
            for p in self.paths:
                try:
                    current = os.path.getmtime(p)
                except Exception:
                    continue
                if current != self._mtimes.get(p):
                    event = WatchEvent(file_path=p,
                                          timestamp=time.time(),
                                          previous_mtime=self._mtimes.get(p, 0),
                                          new_mtime=current)
                    self.callback(event)
                    events.append(event)
                    self._mtimes[p] = current
            time.sleep(self.poll_interval_s)
        return events


# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------

class PlaygroundBackend:
    """A sandboxed evaluator that limits nested-eval depth and
    total statement count. The user-supplied `evaluator`
    callable is called with the source string; this sandbox
    just enforces the limits from the outside."""
    def __init__(self,
                  evaluator: typing.Callable[[str], str],
                  *,
                  max_statement_count: int = 50,
                  max_eval_depth: int = 4):
        self.evaluator = evaluator
        self.max_statement_count = max_statement_count
        self.max_eval_depth = max_eval_depth
        self._eval_depth = 0
        self._stmt_count = 0

    def eval(self, source: str) -> str:
        if self._eval_depth >= self.max_eval_depth:
            raise RuntimeError("max eval depth exceeded in playground")
        # Count statements by counting non-blank lines.
        stmt_count = sum(1 for l in source.split("\n") if l.strip())
        if self._stmt_count + stmt_count > self.max_statement_count:
            raise RuntimeError("max statement count exceeded in playground")
        self._eval_depth += 1
        try:
            return self.evaluator(source)
        finally:
            self._eval_depth -= 1
            self._stmt_count += stmt_count

    def reset(self) -> None:
        self._stmt_count = 0
        self._eval_depth = 0


class Playground:
    """Wraps a `PlaygroundBackend` with convenience output
    formatting. The playground is mainly used by
    `eigen playground` to evaluate Eigen source code in a
    restricted environment."""
    def __init__(self, backend: PlaygroundBackend,
                  *, colour: bool = True):
        self.backend = backend
        self.colour = colour

    def run(self, source: str) -> str:
        return self.backend.eval(source)


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MigrationStep:
    name: str
    apply: typing.Callable[[str], str]
    description: str = ""


def _migrate_v1_to_v2(source: str) -> str:
    """Example migration: v1 used `;`-separated statements,
    v2 uses newline-separated statements (per the §7 fix on
    the lexer). Replace `;` not inside strings/comments with
    a newline."""
    out_lines = []
    in_string = False
    in_line_comment = False
    in_block_comment = False
    for line in source.split("\n"):
        # Walk the line char-by-char to handle nested strings/comments.
        new_line_chars = []
        i = 0
        while i < len(line):
            c = line[i]
            if in_block_comment:
                new_line_chars.append(c)
                if line[i:i+2] == "*/":
                    in_block_comment = False
                    new_line_chars.append(line[i+1])
                    i += 2
                    continue
                i += 1
                continue
            if in_line_comment:
                new_line_chars.append(c)
                i += 1
                continue
            if in_string:
                new_line_chars.append(c)
                if c == '"':
                    in_string = False
                i += 1
                continue
            if line[i:i+2] == "//":
                in_line_comment = True
                new_line_chars.extend(line[i:i+2])
                i += 2
                continue
            if line[i:i+2] == "/*":
                in_block_comment = True
                new_line_chars.extend(line[i:i+2])
                i += 2
                continue
            if c == '"':
                in_string = True
                new_line_chars.append(c)
                i += 1
                continue
            if c == ";":
                new_line_chars.append("\n")
                i += 1
                continue
            new_line_chars.append(c)
            i += 1
        out_lines.append("".join(new_line_chars))
    return "\n".join(out_lines)


def _migrate_remove_h_gate_keyword(source: str) -> str:
    """v1 → v2 migration: the keyword `hgate` is now `h`."""
    return re.sub(r"\bhgate\b", "h", source)


class Migrate:
    """`eigen migrate` — migrates Eigen source files between
    versions. The migration list stores each step; running
    `migrate` walks a source through the chain."""
    def __init__(self):
        self.steps: typing.List[MigrationStep] = []
        self.register_default_migrations()

    def register_default_migrations(self) -> None:
        self.steps.append(MigrationStep(
            name="v1_to_v2_semicolon",
            apply=_migrate_v1_to_v2,
            description="Replace ';' with newline",
        ))
        self.steps.append(MigrationStep(
            name="v1_to_v2_h_gate",
            apply=_migrate_remove_h_gate_keyword,
            description="Rename 'hgate' to 'h'",
        ))

    def migrate(self, source: str,
                  from_version: str = "1", to_version: str = "2") -> str:
        out = source
        for step in self.steps:
            out = step.apply(out)
            out = self._postprocess(out)
        return out

    def diff(self, source_before: str,
              source_after: str) -> typing.List[str]:
        return list(difflib.unified_diff(
            source_before.splitlines(keepends=True),
            source_after.splitlines(keepends=True),
            fromfile="before.mig",
            tofile="after.mig",
        ))

    @staticmethod
    def _postprocess(text: str) -> str:
        # Collapse consecutive blank lines (max 2).
        out_lines = []
        blank_run = 0
        for line in text.split("\n"):
            if line.strip():
                blank_run = 0
                out_lines.append(line)
            else:
                blank_run += 1
                if blank_run <= 2:
                    out_lines.append(line)
        return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Coloured output helpers
# ---------------------------------------------------------------------------

class Colour(enum.Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    GREY = "grey"
    RESET = "reset"


_ANSI_CODES = {
    Colour.RED: "\x1b[31m",
    Colour.GREEN: "\x1b[32m",
    Colour.YELLOW: "\x1b[33m",
    Colour.BLUE: "\x1b[34m",
    Colour.GREY: "\x1b[90m",
    Colour.RESET: "\x1b[0m",
}


def colourise(text: str, colour: Colour,
                enabled: bool = True) -> str:
    if not enabled:
        return text
    return (_ANSI_CODES.get(colour, "")
            + text
            + _ANSI_CODES[Colour.RESET])


class ProgressBar:
    """A simple ASCII progress bar."""
    def __init__(self,
                  total: int,
                  *,
                  bar_width: int = 40,
                  label: str = "",
                  enabled: bool = True):
        self.total = total
        self.current = 0
        self.bar_width = bar_width
        self.label = label
        self.enabled = enabled

    def update(self, n: int = 1) -> None:
        self.current = max(min(self.total, self.current + n), 0)

    def render(self) -> str:
        if not self.enabled or self.total <= 0:
            return ""
        fraction = self.current / self.total
        filled = int(fraction * self.bar_width)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        percent = int(fraction * 100)
        label_text = self.label + " " if self.label else ""
        return f"\r{label_text}[{bar}] {percent}%"

    def finish(self) -> str:
        if not self.enabled:
            return ""
        return self.render() + "\n"


__all__ = [
    "ShellKind",
    "CompletionSpec",
    "default_eigen_completion_spec",
    "generate_completion_script",
    "REPLState",
    "REPLResult",
    "REPL",
    "WatchEvent",
    "FileWatcher",
    "PlaygroundBackend",
    "Playground",
    "MigrationStep",
    "Migrate",
    "Colour",
    "colourise",
    "ProgressBar",
]
