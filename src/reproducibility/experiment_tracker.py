"""
P3 §12.3 — Reproducibility (surface-level).

Roadmap (`sol.md` "12.3 Reproducibility"):
    - [ ] **Experiment tracking** — логирование параметров и результатов
    - [ ] **Seed management** — глобальный + per-component seeds
    - [ ] **Result caching** — кэширование результатов экспериментов
    - [ ] **Export to papers** — LaTeX/JSON export результатов

This module provides:

  * `ExperimentRun` — frozen dataclass bundling a `name`, optional
    `program_hash` (linking back to the `runtime_audit.AuditTrail` of
    §6.4), a deterministic `parameters` mapping, the simulator
    configuration, the deterministic flag, and free-form tags.
  * `ExperimentTracker(workdir, *, clock=time.time)` — append-only
    ledger of ExperimentRun records and their stored results. Supports:
      - `record(run, result, *, overwrite=False)` — persist a
        (run, result) pair, returning the assigned `run_id`.
      - `lookup(program_hash=None, parameters=None, tags=...)` —
        filter cached records and return their results (or empty
        list if no match). Acts as the "result caching" checkbox.
      - `iter_runs(*, with_result=False)` — generator over every
        stored run, optionally paired with its result.
      - `export_json(path)` — write the full ledger as a JSON file
        (one entry per run; human-readable, deterministic ordering
        by run_id).
      - `export_latex(path, *, title=None, columns=None)` — write a
        LaTeX `longtable` summarizing the recorded runs. Columns
        default to: run_id, name, program_hash (truncated), tags,
        parameters (truncated), deterministic, recorded_at (ns
        timestamp).
      - `clear()` — empty cache + ledger (for test/doctor).

Surfaces intentionally NOT covered by this envelope:
  * Global seed management is already handled at the `EigenVM(...)
    deterministic=True, seed=...` level (see §6.1 / §0 VM hardening).
    We don't add a second layer here.
  * Program-hash lookup is delegated to `runtime_audit.AuditTrail`
    via the recorded `program_hash` field. The tracker just stores
    the hash; the user is responsible for joining to the audit
    trail if they want the full source text.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import time
import typing


@dataclasses.dataclass(frozen=True)
class ExperimentRun:
    """Identifier + metadata bundle for one experiment invocation.

    `parameters` is a JSON-serializable mapping (e.g., `{"theta": 1.57,
    "shots": 1000}`). `simulator_config` is the QuantumSimulator /
    EigenVM constructor kwargs as a JSON-serializable dict; for
    the surface-level tracker we don't validate the contents — the
    user is responsible for keeping them reproducible. The
    `deterministic` flag is the value passed to `EigenVM(...,
    deterministic=...)` at execution time and is the canonical
    reproducibility bit.

    `tags` is a tuple of strings for filter / grouping queries.
    `run_id` is an optional field assigned by `ExperimentTracker.record()`
    — applications that build an `ExperimentRun` themselves leave it
    empty; the canonical ID is allocated lazily on the first
    `record()` call.
    """
    name: str
    program_hash: typing.Optional[str] = None
    parameters: typing.Mapping[str, typing.Any] = dataclasses.field(
        default_factory=dict)
    simulator_config: typing.Mapping[str, typing.Any] = dataclasses.field(
        default_factory=dict)
    deterministic: bool = False
    tags: typing.Tuple[str, ...] = ()
    recorded_at: int = 0  # nanoseconds since epoch; 0 == unrecorded
    run_id: str = ""  # set by ExperimentTracker.record()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "program_hash": self.program_hash or "",
            "parameters": dict(self.parameters),
            "simulator_config": dict(self.simulator_config),
            "deterministic": self.deterministic,
            "tags": list(self.tags),
            "recorded_at": self.recorded_at,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRun":
        return cls(
            name=d["name"],
            program_hash=d.get("program_hash") or None,
            parameters=dict(d.get("parameters", {}) or {}),
            simulator_config=dict(d.get("simulator_config", {}) or {}),
            deterministic=bool(d.get("deterministic", False)),
            tags=tuple(d.get("tags", []) or []),
            recorded_at=int(d.get("recorded_at", 0) or 0),
            run_id=d.get("run_id", ""),
        )


@dataclasses.dataclass
class _LedgerEntry:
    run_id: str
    run: ExperimentRun
    result: typing.Any  # JSON-serializable

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run": self.run.to_dict(),
            "result": self.result,
        }


def _params_key(parameters) -> str:
    """Stable stringified key for the lookup index. Uses JSON with
    sort_keys=True so two equivalent mappings (regardless of
    insertion order) map to the same key.
    """
    return json.dumps(parameters, sort_keys=True, default=str)


class ExperimentTracker:
    """Append-only experiment ledger + result cache.

    Persists to a single JSON file `experiments.json` under `workdir`
    on every `record()`/`clear()`. Lookups are in-memory (the file is
    loaded once on construction and rebuilt on `clear()`).
    """

    LEDGER_FILENAME = "experiments.json"

    def __init__(self, workdir: str, *, clock=time.time):
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.ledger_path = os.path.join(self.workdir, self.LEDGER_FILENAME)
        self._clock = clock
        self._counter = itertools.count(1)
        self._entries: typing.List[_LedgerEntry] = self._load()

    # ------------------------------------------------------- I/O

    def _load(self) -> typing.List[_LedgerEntry]:
        if not os.path.isfile(self.ledger_path):
            return []
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return []
        out = []
        for entry in raw.get("entries", []):
            run = ExperimentRun.from_dict(entry.get("run", {}))
            out.append(_LedgerEntry(
                run_id=entry["run_id"], run=run, result=entry.get("result"),
            ))
        # Restart counter after the largest existing run_id, so the
        # next record() returns monotonically-increasing IDs even
        # across clear()/reload cycles. This matches the roadmap's
        # "result caching reproducibility" checkbox.
        max_n = 0
        for entry in out:
            try:
                # Format: "run-NNN" — extract NNN as int.
                if entry.run_id.startswith("run-"):
                    n = int(entry.run_id[4:])
                    max_n = max(max_n, n)
            except ValueError:
                continue
        self._counter = itertools.count(max_n + 1)
        return out

    def _save(self) -> None:
        payload = {"entries": [e.to_dict() for e in self._entries]}
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    # ---------------------------------------------------- writes

    def record(self, run: ExperimentRun, result: typing.Any, *,
               overwrite: bool = False) -> str:
        """Persist `result` indexed by `run`. Returns the assigned
        `run_id`. By default, re-recording the same (program_hash +
        parameters) pair appends a new entry; pass `overwrite=True`
        to replace any existing cached result for this (hash, params)
        key (keeps the new run_id). Records are append-only otherwise
        — historical results MUST stay queryable for reproducibility.
        """
        # Stamp the run with the recorded_at timestamp + assigned id.
        stamp_ns = int(self._clock() * 1e9)
        if overwrite:
            params_key = _params_key(run.parameters)
            for i, entry in enumerate(self._entries):
                if (entry.run.program_hash == run.program_hash
                        and _params_key(entry.run.parameters) == params_key):
                    # Replace in place but keep the original run_id so
                    # downstream lookups don't have to recompute.
                    new_run = dataclasses.replace(
                        run, recorded_at=stamp_ns,
                        run_id=entry.run_id)
                    self._entries[i] = _LedgerEntry(
                        run_id=entry.run_id, run=new_run, result=result)
                    self._save()
                    return entry.run_id
        run_id = f"run-{next(self._counter):04d}"
        run = dataclasses.replace(run, recorded_at=stamp_ns, run_id=run_id)
        self._entries.append(_LedgerEntry(run_id=run_id, run=run, result=result))
        self._save()
        return run_id

    def clear(self) -> None:
        """Drop every record (test/doctor). NOT recoverable."""
        self._entries = []
        self._counter = itertools.count(1)
        if os.path.isfile(self.ledger_path):
            try:
                os.unlink(self.ledger_path)
            except OSError:
                pass

    # ----------------------------------------------------- reads

    def lookup(self, *,
               program_hash: typing.Optional[str] = None,
               parameters: typing.Optional[typing.Mapping] = None,
               tags: typing.Optional[typing.Iterable[str]] = None,
               name: typing.Optional[str] = None,
               ) -> typing.List[typing.Tuple[ExperimentRun, typing.Any]]:
        """Return `(run, result)` pairs matching ALL non-None filters.
        Each filter is matched exactly (string equality for hash and
        name; json-key equality for parameters; subset for tags).
        """
        out = []
        tag_set = set(tags) if tags is not None else None
        for entry in self._entries:
            r = entry.run
            if program_hash is not None and r.program_hash != program_hash:
                continue
            if parameters is not None and _params_key(r.parameters) != _params_key(parameters):
                continue
            if name is not None and r.name != name:
                continue
            if tag_set is not None and not tag_set.issubset(set(r.tags)):
                continue
            out.append((r, entry.result))
        return out

    def iter_runs(self, *, with_result: bool = False,
                  ) -> typing.Iterator:
        """Generator over every recorded run. When `with_result=True`
        yields `(run, result)` tuples; otherwise yields just `run`.
        """
        for entry in self._entries:
            if with_result:
                yield entry.run, entry.result
            else:
                yield entry.run

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, run_id: str) -> bool:
        return any(e.run_id == run_id for e in self._entries)

    # --------------------------------------------------- exports

    def export_json(self, path: str) -> None:
        """Write the full ledger to `path` as JSON. Same shape as the
        on-disk `experiments.json` ledger file but at an
        arbitrary location — useful for bundling results with a
        publication/upload.
        """
        payload = {"entries": [e.to_dict() for e in self._entries]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)

    def export_latex(self, path: str, *,
                    title: typing.Optional[str] = "Eigen Experiment Log",
                    columns: typing.Optional[typing.List[str]] = None,
                    ) -> None:
        """Write a LaTeX `longtable` summarizing the recorded runs.

        By default the table has columns:
            run_id | name | program_hash (truncated) | tags |
            parameters (json, truncated) | deterministic | recorded_at

        `columns` may override the column list (subset of the above
        names).         The generated `.tex` is intentionally minimal: no
        external package beyond `longtable`, so it can be
        ``\\input{}``-ed straight into a paper template.
        """
        if columns is None:
            columns = ["run_id", "name", "program_hash", "tags",
                       "parameters", "deterministic", "recorded_at"]
        # Column display labels.
        labels = {
            "run_id": "Run ID",
            "name": "Name",
            "program_hash": "Program Hash",
            "tags": "Tags",
            "parameters": "Parameters",
            "deterministic": "Det.",
            "recorded_at": "Recorded At (ns)",
        }
        lines = []
        if title:
            lines.append(f"% Auto-generated by Eigen ExperimentTracker")
            lines.append(f"% {title}")
            lines.append("")
        lines.append(r"\begin{longtable}{" + "l" * len(columns) + "}")
        if title:
            lines.append(r"\caption{" + _tex_escape(title) + r"} \\")
        lines.append(r"\hline")
        lines.append(" & ".join(labels[c] for c in columns) + r" \\ \hline")
        lines.append(r"\endhead")
        lines.append(r"\hline \multicolumn{" + str(len(columns)) +
                     r"}{r}{\emph{continued}} \\ \hline")
        lines.append(r"\endfoot")
        lines.append(r"\hline")
        lines.append(r"\endlastfoot")
        for entry in self._entries:
            row = []
            for c in columns:
                row.append(_cell_value(entry, c))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\end{longtable}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _tex_escape(s: str) -> str:
    """Minimal TeX escape for safe text embedding."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _truncate(s: str, n: int = 12) -> str:
    return s if len(s) <= n else s[:n] + "..."


def _cell_value(entry: _LedgerEntry, column: str) -> str:
    """Render the per-row cell value for `column`, with light TeX
    escaping (and truncation for long hash/params fields so the
    paper-template table stays readable on one page).
    """
    r = entry.run
    if column == "run_id":
        return _tex_escape(entry.run_id)
    if column == "name":
        return _tex_escape(r.name)
    if column == "program_hash":
        return _tex_escape(_truncate(r.program_hash or "", n=16))
    if column == "tags":
        return _tex_escape(", ".join(r.tags))
    if column == "parameters":
        s = json.dumps(r.parameters, sort_keys=True, default=str)
        return _tex_escape(_truncate(s, n=32))
    if column == "deterministic":
        return "yes" if r.deterministic else "no"
    if column == "recorded_at":
        return str(r.recorded_at)
    return ""
