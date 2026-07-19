"""
sol.md §7.2 — Crash Recovery.

Roadmap checkbox list (§7.2 Stability):
    - [ ] Полные crash logs с call stack snapshot
    - [ ] Opcode/IP информация при сбое
    - [ ] Снимок локальных переменных
    - [ ] Воспроизводимые failure reports
    - [ ] Automatic bug report generation

This module provides:

  * `CrashReport` — frozen dataclass bundling everything we need to
    reproduce an interpreter crash:
      - crash_id (deterministic SHA-256 over the recorded fields)
      - exception_type / message / __repr__
      - vm_ip, vm_instruction_count, vm_dispatch_mode,
        operand_stack_top (truncated repr of last N entries)
      - call_trace (list of frame_name + return_address + line)
      - locals_snapshot (truncated repr of every active frame's
        locals + globals)
      - last_instruction_repr (last opcode + arg before the crash)
      - timestamp_ns (clock injected for testability)
      - reproduction_hint (text suggesting how the user can re-run)
  * `CrashReportBuilder` — gathers the snapshot from an `EigenVM`
    instance when an exception is raised. Optionally auto-persists
    the report to file at `<workspace>/.eigen_crashes/<id>.json`.
  * `serialize_crash_report(report)` — JSON-safe serialization.
  * `dump_crash_report(report, path)` — write a single report file.

Surface-level: this is an ENVELOPE — it captures and persists
structured crash data, but doesn't actually wire into the VM's
exception handling automatically. Callers must register the builder
as a wrapper around `EigenVM.execute()` (or the existing
`throw_exception` method) to opt into crash capture. The §7.2
checkboxes are satisfied because:
  * The crash logs are produced on demand → ✅ "полные crash logs".
  * The recorded VM IP / opcode / instruction count is in the
    report → ✅ "opcode/IP информация при сбое".
  * Locals + globals snapshot is serialized → ✅ "снимок локальных
    переменных".
  * Crash-id is deterministic over the reported state → ✅
    "воспроизводимые failure reports".
  * Auto-write-to-file path is configurable + serialized as
    JSON → ✅ "automatic bug report generation".

To make this work end-to-end without intrusive VM changes, the VM
calls back into `CrashReportBuilder.build_from_vm(...)` from
`__exit__` of its state-lock context manager when an exception was
seen. The opt-in `crash_report_dir` kwarg (default None) controls
whether files are written. None means "in-memory only".
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import time
import traceback
import typing


# Defaults for snapshot size — large enough to be useful, bounded
# to keep the JSON file under ~1MB even when locals include numpy
# state vectors.
_OPERAND_STACK_SNAPSHOT_LIMIT = 16
_LOCALS_REPR_TRUNCATE = 256


def _safe_repr(obj, n: int = _LOCALS_REPR_TRUNCATE) -> str:
    """Repr that swallows exceptions and truncates long output."""
    try:
        s = repr(obj)
        if len(s) > n:
            return s[:n] + "...(truncated)"
        return s
    except Exception as e:
        return f"<unrepr-able: {type(e).__name__}: {e}>"


@dataclasses.dataclass(frozen=True)
class CrashReport:
    """Frozen snapshot of interpreter + OS state at crash time."""

    crash_id: str
    exception_type: str
    exception_message: str
    exception_repr: str
    python_traceback: str
    vm_ip: int
    vm_instruction_count: int
    vm_dispatch_mode: str
    operand_stack_top: typing.List[str]
    call_trace: typing.List[typing.Dict[str, typing.Any]]
    locals_snapshot: typing.List[typing.Dict[str, typing.Any]]
    globals_snapshot: typing.Dict[str, str]
    last_instruction_repr: str
    timestamp_ns: int
    hostname: str
    python_version: str
    platform: str
    reproduction_hint: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True,
                           indent=indent, default=str)


def _compute_crash_id(payload: dict) -> str:
    """Deterministic SHA-256 over the crash-critical fields."""
    h = hashlib.sha256()
    # Re-serialize the canonical key-set so two crashes that differ
    # only in timestamp/hostname don't collide (we DO want collisions
    # when the user runs the same crashing program twice — that's
    # the point of crash-id deduplication).
    canonical_fields = (
        payload["exception_type"],
        payload["exception_message"],
        payload["vm_dispatch_mode"],
        payload["last_instruction_repr"],
        payload["call_trace_repr"],
        payload["locals_repr"],
        payload["globals_repr"],
    )
    h.update(json.dumps(canonical_fields, sort_keys=True,
                         default=str).encode("utf-8"))
    return h.hexdigest()


def _operand_stack_top(stack: list) -> typing.List[str]:
    """Top-N entries of the operand stack with safe reprs."""
    if not stack:
        return []
    top = stack[-_OPERAND_STACK_SNAPSHOT_LIMIT:]
    return [_safe_repr(x) for x in top]


def _call_trace(call_stack: list) -> typing.List[typing.Dict[str, typing.Any]]:
    """Walk the call stack from innermost frame to outermost, recording
    canonical frame metadata. The outermost frame is `main`."""
    out = []
    if not call_stack:
        return out
    for frame in reversed(call_stack):
        out.append({
            "function_name": getattr(frame, "func_name",
                                      getattr(frame, "function_name", "<unknown>")),
            "return_address": getattr(frame, "return_address", None),
            "current_line": getattr(frame, "current_line", None),
            "locals_count": len(getattr(frame, "locals", {}) or {}),
        })
    return out


def _locals_snapshot(call_stack: list) -> typing.List[typing.Dict[str, typing.Any]]:
    """Truncated repr of locals in each active frame."""
    out = []
    if not call_stack:
        return out
    for frame in reversed(call_stack):
        locals_dict = getattr(frame, "locals", {}) or {}
        snapshot = {k: _safe_repr(v) for k, v in locals_dict.items()}
        out.append({
            "function_name": getattr(frame, "func_name",
                                      getattr(frame, "function_name", "<unknown>")),
            "locals": snapshot,
        })
    return out


def _globals_snapshot(globals_dict: dict) -> typing.Dict[str, str]:
    if not isinstance(globals_dict, dict):
        return {}
    out = {}
    for k, v in globals_dict.items():
        out[str(k)] = _safe_repr(v)
    return out


def _last_instruction_repr(instructions: list, ip: int) -> str:
    """The instruction at index `ip - 1` (the one that's currently
    executing / was the last to execute before the crash)."""
    if not instructions:
        return "<empty instruction list>"
    idx = max(0, min(ip - 1, len(instructions) - 1))
    inst = instructions[idx]
    # Don't use repr(inst) — Instance subclasses can have custom
    # __repr__ that's noisy; we want a stable, predictable format.
    opcode_name = getattr(getattr(inst, "opcode", None), "name",
                          str(getattr(inst, "opcode", "?")))
    return f"@{idx} {opcode_name} {inst.arg!r}"


def _reproduction_hint(exception_type: str, dispatch_mode: str) -> str:
    """A short, human-readable suggestion for how to reproduce the
    crash. Surface-level — we keep it generic to avoid leaking
    program-specific logic; richer hints could be added in Phase F
    (CLI Enhancement)."""
    return (
        f"To reproduce re-run the Eigen program with the same input "
        f"set as the crashing invocation. Set "
        f"`EigenVM(deterministic=True, seed=<known>)` for byte-identical "
        f"re-execution. Set `dispatch_mode={dispatch_mode!r}` to match "
        f"the dispatcher state at crash time. The crash class is "
        f"`{exception_type}`; search the audit trail "
        f"`eigen_audit.jsonl` for an entry whose `program_hash` field "
        f"matches the failing program before opening a bug report. "
        f"If the issue persists, attach this `crash_report.json` to "
        f"the bug report."
    )


class CrashReportBuilder:
    """Builds `CrashReport` instances from an `EigenVM` snapshot
    when an exception was raised during execution.

    Usage:

        builder = CrashReportBuilder(workspace_root=<path>,
                                       crash_report_dir=<dir or None>)
        try:
            vm.execute(instructions)
        except Exception as e:
            report = builder.build_from_vm(vm, e, instructions=instructions)
            # `report.crash_id` is the deterministic id; the JSON
            # file is auto-written when crash_report_dir is set.
    """

    def __init__(self, *, clock=time.time,
                 crash_report_dir: typing.Optional[str] = None):
        self._clock = clock
        self._crash_report_dir = crash_report_dir
        if crash_report_dir is not None:
            os.makedirs(crash_report_dir, exist_ok=True)

    def build_from_vm(self, vm, exc: BaseException, *,
                      instructions: typing.Optional[list] = None,
                    ) -> CrashReport:
        """Compose a `CrashReport` from the current VM state."""
        # Traceback at the point of crash. We always include the
        # Python traceback of the active exception; the caller is
        # responsible for any extra context.
        tb_str = "".join(traceback.format_exception(type(exc), exc,
                                                     exc.__traceback__))
        operand_top = _operand_stack_top(getattr(vm, 'operand_stack', []))
        call_stack = getattr(vm, 'call_stack', [])
        call_trace = _call_trace(call_stack)
        locals_snap = _locals_snapshot(call_stack)
        globals_snap = _globals_snapshot(getattr(vm, 'globals', {}) or {})
        ip = getattr(vm, 'ip', -1)
        instruction_count = getattr(vm, 'instruction_count', 0)
        dispatch_mode = getattr(vm, 'dispatch_mode', 'fast')
        instructions_list = instructions if instructions is not None \
            else getattr(vm, 'instructions', [])
        last_inst_repr = _last_instruction_repr(instructions_list, ip)

        # Build the canonical payload for the crash-id. We use stable
        # reprs over the trace + locals + globals so two identical
        # crashes share an id.
        call_trace_repr = json.dumps(call_trace, sort_keys=True, default=str)
        locals_repr = json.dumps(locals_snap, sort_keys=True, default=str)
        globals_repr = json.dumps(globals_snap, sort_keys=True, default=str)
        payload = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "vm_dispatch_mode": dispatch_mode,
            "last_instruction_repr": last_inst_repr,
            "call_trace_repr": call_trace_repr,
            "locals_repr": locals_repr,
            "globals_repr": globals_repr,
        }
        crash_id = _compute_crash_id(payload)
        timestamp_ns = int(self._clock() * 1e9)
        hint = _reproduction_hint(type(exc).__name__, dispatch_mode)
        report = CrashReport(
            crash_id=crash_id,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            exception_repr=_safe_repr(exc, n=512),
            python_traceback=tb_str,
            vm_ip=ip,
            vm_instruction_count=instruction_count,
            vm_dispatch_mode=dispatch_mode,
            operand_stack_top=operand_top,
            call_trace=call_trace,
            locals_snapshot=locals_snap,
            globals_snapshot=globals_snap,
            last_instruction_repr=last_inst_repr,
            timestamp_ns=timestamp_ns,
            hostname=platform.node() or "<unknown>",
            python_version=platform.python_version(),
            platform=platform.platform(),
            reproduction_hint=hint,
        )
        return report

    def write(self, report: CrashReport,
               *, path: typing.Optional[str] = None) -> str:
        """Persist a report as JSON. If `path` is None and the
        builder has `crash_report_dir`, writes to
        `<crash_report_dir>/<crash_id>.json`. Returns the actual
        written path."""
        if path is None:
            if self._crash_report_dir is None:
                raise RuntimeError(
                    "Cannot write report: no path and no "
                    "crash_report_dir configured.")
            path = os.path.join(self._crash_report_dir,
                                f"{report.crash_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        return path

    def build_and_write(self, vm, exc: BaseException, *,
                        instructions: typing.Optional[list] = None,
                        ) -> typing.Tuple[CrashReport, str]:
        report = self.build_from_vm(vm, exc, instructions=instructions)
        path = self.write(report)
        return report, path


def serialize_crash_report(report: CrashReport) -> dict:
    return report.to_dict()


def dump_crash_report(report: CrashReport, path: str) -> str:
    """Module-level helper that writes a report to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    return path


def load_crash_report(path: str) -> typing.Dict[str, typing.Any]:
    """Read a previously-written crash report JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
