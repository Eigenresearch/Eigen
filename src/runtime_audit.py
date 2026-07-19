"""
§6.4 — Audit Trail for runtime executions.

The roadmap (`sol.md` "Воспроизводимость" section 6.4) lists:
    - Детерминированный seed management             (handled separately
                                                      via `--deterministic`
                                                      and `seed=` VM ctor)
    - Lockfile для зависимостей                      (in packager)
    - Версионирование результатов                   (handled via `program_hash`
                                                      field on entries)
    - Audit trail для всех выполнений               ← THIS module

The AuditTrail:
  * Captures one `AuditEntry` per VM/runtime execution — the program
    hash (so we can reproduce from source), the runtime parameters
    (seed, sim_type, noise profile, deterministic flag), the wall-clock
    duration in nanoseconds, an `outcome` summary (success / failure
    with type and message), and a structured `result_fingerprint` that
    lets downstream tools diff consecutive runs of the same program.
  * Persists into a per-host JSONL file (`eigen_audit.jsonl` by default)
    with one line per entry. JSONL is intentional — it's append-only
    (you don't have to load-then-save the whole trail to add an entry)
    and it streams well across networked auditing tools.
  * Supports a module-level singleton (`get_default_trail()`) so call
    sites don't have to thread the trail through everywhere; consumers
    that prefer explicit dependency injection can construct their own
    `AuditTrail` instance and pass it in.
  * Provides an opt-in integration point: the `EigenVM.execute()` and
    `EigenRuntime.execute()` methods accept an optional `audit` kwarg
    and call `trail.record(...)` around the inner _execute_locked call.
    We keep this opt-in (off by default) to avoid changing test counts
    that didn't ask for auditing.

The `program_hash` is computed from the canonical source text via
SHA-256 (matches the convention used by the test_cache and compiler_db);
callers that already have a hash from elsewhere can pass it in to
avoid recomputing.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


DEFAULT_AUDIT_FILE = "eigen_audit.jsonl"


def hash_program(source: str | bytes) -> str:
    """SHA-256 of canonical program source. Returns a hex digest.

    Accepts ``str`` or ``bytes``; ``str`` is encoded as UTF-8. The hash
    is intentionally canonical (no whitespace trimming or comment
    stripping) — we want the same source to always produce the same
    hash AND different-but-equivalent sources to produce different
    hashes (so users can correlate runs back to exact checked-in code).
    """
    if isinstance(source, str):
        payload = source.encode("utf-8")
    else:
        payload = bytes(source)
    return hashlib.sha256(payload).hexdigest()


@dataclass
class AuditEntry:
    """One persistent record of a single runtime execution."""
    program_hash: str
    started_at_ns: int                # monotonic ns since boot, for duration
    wall_clock_ns: int                # wall-clock duration of execution
    outcome: str                      # 'success' | 'failure'
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    seed: Optional[int] = None
    sim_type: Optional[str] = None
    gpu_platform: Optional[str] = None
    deterministic: bool = False
    noise_type: Optional[str] = None
    noise_prob: Optional[float] = None
    num_instructions: Optional[int] = None
    result_fingerprint: Optional[str] = None
    hostname: str = field(default_factory=socket.gethostname)
    python_version: str = field(default_factory=lambda: sys.version)
    # An opaque caller-supplied dict (e.g. extra CLI args, batch ID,
    # CI build link, etc.). We JSON-serialise it on write so downstream
    # consumers can grep on stable keys.
    extra: dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Return a single-line JSON string suitable for `print >> .jsonl`."""
        # `default=str` lets us handle enum/Path/non-JSON primitives in
        # `extra` without raising — they're surfaced as their str() repr.
        return json.dumps(dataclasses.asdict(self),
                          default=str, sort_keys=True)


class AuditTrail:
    """Append-only audit log for VM/runtime executions.

    Construction:
        ``AuditTrail(path=None, enabled=True)`` — when ``path`` is
        ``None``, the trail stays in-memory only (useful for tests and
        for short-lived CLIs that just want the most-recent-N entries).
        When ``path`` is provided, every ``record`` call appends a
        line to the file in JSONL format (one self-contained JSON
        document per line).

    The trail is intentionally not in-memory-cached beyond ``max_buffer``
    entries; long-running processes should rely on the persisted JSONL
    file rather than holding the whole history in RAM.
    """

    def __init__(self,
                 path: Optional[str | os.PathLike] = None,
                 enabled: bool = True,
                 max_buffer: int = 1024):
        self.path = Path(path) if path is not None else None
        self.enabled = enabled
        self.max_buffer = max_buffer
        self._buffer: list[AuditEntry] = []
        self._failed_writes = 0

    def record(self,
               program_hash: str,
               *,
               seed: Optional[int] = None,
               sim_type: Optional[str] = None,
               gpu_platform: Optional[str] = None,
               deterministic: bool = False,
               noise_type: Optional[str] = None,
               noise_prob: Optional[float] = None,
               num_instructions: Optional[int] = None,
               outcome: str = "success",
               error: Optional[BaseException] = None,
               started_at_ns: Optional[int] = None,
               wall_clock_ns: Optional[int] = None,
               result_fingerprint: Optional[str] = None,
               extra: Optional[dict] = None) -> AuditEntry:
        """Build an ``AuditEntry`` from the supplied fields, append it
        to the in-memory buffer (up to ``max_buffer`` slots), flush it
        to the on-disk JSONL file when configured, and return the
        entry to the caller. Mutating the returned object does NOT
        retroactively edit the persisted entry — the call is
        point-in-time.

        ``error``: when provided, ``outcome`` becomes ``"failure"``
        unless the caller explicitly set it otherwise; ``error_type``
        and ``error_message`` are populated from the exception.

        ``started_at_ns`` / ``wall_clock_ns``: callers that already
        measured the duration (e.g. by wrapping ``execute()`` in
        their own timer) pass them through; otherwise leave them
        ``None`` and ``record`` uses the current monotonic clock
        (which means ``wall_clock_ns == 0`` — you have to have your
        own measurement when you want a meaningful duration).
        """
        if not self.enabled:
            # Still build the entry so callers can inspect/reuse it —
            # we just don't persist it anywhere.
            return AuditEntry(
                program_hash=program_hash,
                started_at_ns=started_at_ns or time.monotonic_ns(),
                wall_clock_ns=wall_clock_ns or 0,
                outcome=("failure" if error is not None
                         else outcome),
                error_type=type(error).__name__ if error else None,
                error_message=str(error) if error else None,
                seed=seed,
                sim_type=sim_type,
                gpu_platform=gpu_platform,
                deterministic=deterministic,
                noise_type=noise_type,
                noise_prob=noise_prob,
                num_instructions=num_instructions,
                result_fingerprint=result_fingerprint,
                extra=dict(extra or {}),
            )

        if started_at_ns is None:
            started_at_ns = time.monotonic_ns()
        if wall_clock_ns is None:
            wall_clock_ns = 0

        if error is not None and outcome == "success":
            outcome = "failure"

        entry = AuditEntry(
            program_hash=program_hash,
            started_at_ns=started_at_ns,
            wall_clock_ns=wall_clock_ns,
            outcome=outcome,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            seed=seed,
            sim_type=sim_type,
            gpu_platform=gpu_platform,
            deterministic=deterministic,
            noise_type=noise_type,
            noise_prob=noise_prob,
            num_instructions=num_instructions,
            result_fingerprint=result_fingerprint,
            extra=dict(extra or {}),
        )

        self._buffer.append(entry)
        if len(self._buffer) > self.max_buffer:
            # Drop the oldest entry — we don't bother flushing to disk
            # since the file is the durable record anyway.
            self._buffer.pop(0)

        self._write_to_disk(entry)
        return entry

    def _write_to_disk(self, entry: AuditEntry) -> None:
        if self.path is None:
            return
        try:
            # Append mode: don't read+rewrite the whole file just to
            # add a line. 'a' creates the file if it doesn't exist.
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(entry.to_jsonl() + "\n")
        except OSError:
            # Don't fail the audited operation just because we couldn't
            # write the audit trail — increment a counter and move on.
            # Tests assert this behaviour explicitly.
            self._failed_writes += 1

    def entries(self) -> list[AuditEntry]:
        """Return the in-memory buffered entries in insertion order.

        This is NOT the full on-disk trail — when persistence is
        configured, use ``read_jsonl()`` to see everything (including
        rotated entries beyond ``max_buffer``).
        """
        return list(self._buffer)

    def read_jsonl(self) -> list[dict]:
        """Read the persisted JSONL file (if any) into a list of dicts.

        Returns an empty list when no persistence path is configured
        or the file does not exist yet. Lines that do not parse as
        JSON are skipped (with ``_failed_writes`` incremented per
        skip) so a single malformed line doesn't poison the whole
        read.
        """
        if self.path is None:
            return []
        if not self.path.exists():
            return []
        out: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    self._failed_writes += 1
        return out

    def clear(self) -> None:
        """Drop the in-memory buffer AND any persisted JSONL file.

        Use with care — this is intended for tests that need a clean
        slate. Production callers should rotate / archive the file
        via external tooling instead of overwriting it.
        """
        self._buffer.clear()
        if self.path is not None and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                self._failed_writes += 1


# -------- module-level singleton --------------------------------------

_default_trail: Optional[AuditTrail] = None


def get_default_trail() -> AuditTrail:
    """Return a process-lifetime `AuditTrail` singleton.

    Defaults to in-memory (no path, enabled=True). Callers wanting
    persistence should call ``set_default_trail(AuditTrail(path=...))``
    early in process startup.
    """
    global _default_trail
    if _default_trail is None:
        _default_trail = AuditTrail(path=None, enabled=True)
    return _default_trail


def set_default_trail(trail: AuditTrail) -> None:
    """Replace the module-level singleton (used by tests)."""
    global _default_trail
    _default_trail = trail


def reset_default_trail() -> None:
    """Forget the singleton — next call to ``get_default_trail()``
    will create a fresh in-memory one. Tests use this in setUp to
    ensure isolation."""
    global _default_trail
    _default_trail = None


def fingerprint_result(result: Any) -> str:
    """Hash a serializable Python object (typically ``vm.globals`` or
    a representative subset of the run's results) into a stable
    fingerprint the audit log can compare across runs."""
    payload = json.dumps(result, default=str, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
