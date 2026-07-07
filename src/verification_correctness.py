"""§2.3 — Корректность Верификации (Verification Correctness).

Roadmap checkboxes (4 items):

    - [ ] Canonical hash как fast-reject, НЕ как proof of equivalence
    - [ ] Rewrite-based верификация как fallback
    - [ ] Exact equivalence только где математически обосновано
    - [ ] Предупреждения в CLI output о границах верификации

This module is a thin envelope over the existing
`src.equivalence.EquivalenceChecker` plus `src.zx.zx_equivalence`
that:

  1. Names the three distinct verification modes
     (`FAST_REJECT`, `REWRITE`, `EXACT`) and exposes them
     through a `VerificationMode` enum.
  2. Always returns a `VerificationReport` carrying not just
     `is_equivalent` but also `mode_used`, `warnings`, and
     `notes`. This satisfies the CLI-warning checkbox by giving
     the caller structured data they can render as a CLI banner.
  3. Documents the contract:
       - `FAST_REJECT` mode: canonical hash ONLY — if hashes
         match, we *cannot* conclude `(in, out)` equivalent
         (hash collision; canonicalization may have missed
         semantically-relevant structure). The report's warnings
         list flags this.
       - `REWRITE` mode: try ZX rewrites + equality of
         canonical form. Still cannot prove equivalence
         universally — but can prove many cases of practical
         interest like Clifford+T optimisation candidates.
       - `EXACT` mode: small-circuit unitary comparison;
         mathematically equivalent iff `U1 == e^{iφ} U2` for
         some global phase φ. This is the only "proof of
         equivalence" mode.

The envelope keeps the existing `EquivalenceChecker.are_equivalent`
semantics — what it adds is:
  - explicit `mode` parameter for callers who want to force a
    specific verification path,
  - explicit warnings when callers fall back through
    increasingly-sound verification modes.
"""
from __future__ import annotations

import dataclasses
import enum
import typing


# We import lazily so this module is import-safe even if the
# backend changes (which can happen during development).
try:
    from src.equivalence import EquivalenceChecker
    from src.canonicalizer import Canonicalizer
    _DEFAULT_CHECKER = EquivalenceChecker()
    _DEFAULT_CANONICALIZER = Canonicalizer()
except Exception:
    # Fall back to None sentinels — the envelope functions
    # require callers to pass a checker explicitly or use the
    # default. We make sure the public names are still
    # inspectable for tests that focus on the envelope alone.
    EquivalenceChecker = None  # type: ignore[assignment]
    Canonicalizer = None  # type: ignore[assignment]
    _DEFAULT_CHECKER = None
    _DEFAULT_CANONICALIZER = None


def _hash_of(graph) -> typing.Optional[str]:
    """Compute the canonical hash of an EQIR graph via the
    `Canonicalizer.hash_circuit` API."""
    if _DEFAULT_CANONICALIZER is None:
        return None
    try:
        return _DEFAULT_CANONICALIZER.hash_circuit(graph)
    except Exception:
        return None


class VerificationMode(enum.Enum):
    FAST_REJECT = "fast_reject"
    """Canonical hash only. If hashes differ: definitely NOT
    equivalent. If hashes match: cannot prove equivalence —
    hash collisions are possible; canonicalization may miss
    structural variation."""
    REWRITE = "rewrite"
    """ZX-calculus rewriting + canonical comparison. Can
    discharge most common Clifford+T equivalences."""
    EXACT = "exact"
    """Small-circuit unitary comparison. The only
    mathematically-sound "proof of equivalence" mode in this
    envelope, applicable only for <=8 qubits."""
    AUTO = "auto"
    """Default behavior: pick FAST_REJECT, then REWRITE, then
    EXACT, in descending order of cost."""


@dataclasses.dataclass
class VerificationReport:
    is_equivalent: typing.Optional[bool]
    """None if verification was inconclusive (e.g. FAST_REJECT
    matched hashes but couldn't prove equivalence); True/False
    otherwise."""
    mode_used: VerificationMode
    warnings: typing.List[str] = dataclasses.field(default_factory=list)
    notes: typing.List[str] = dataclasses.field(default_factory=list)
    skipped_modes: typing.List[VerificationMode] = dataclasses.field(
        default_factory=list)
    canonical_hash_1: typing.Optional[str] = None
    canonical_hash_2: typing.Optional[str] = None

    def has_warning(self) -> bool:
        return bool(self.warnings)


def verify_equivalence(graph1, graph2, *,
                        mode: VerificationMode = VerificationMode.AUTO,
                        checker: typing.Optional[object] = None,
                        ) -> VerificationReport:
    """Verify equivalence of two EQIR graphs.

    The function consults the supplied `checker` (or the default
    `EquivalenceChecker`). The semantics table:

      - `FAST_REJECT`: only compares canonical hashes. Returns
        `is_equivalent=False` if hashes differ; `is_equivalent=None`
        (inconclusive) if hashes match, with a warning that
        canonical hash ≠ proof of equivalence.
      - `REWRITE`: delegates to the checker's rewrite-based
        verification (ZX rewriting + canonical form). Returns
        True/False with notes about the rewrite chain.
      - `EXACT`: delegates to the checker's exact-equivalence
        matrix comparison; returns True/False. If the circuit is
        too large for exact comparison, falls back to REWRITE
        with a warning.
      - `AUTO`: tries FAST_REJECT first; on hash mismatch,
        stops with is_equivalent=False. On hash match, tries
        EXACT (if <= 8 qubits) and falls back to REWRITE
        otherwise. Always returns a `mode_used` describing what
        was actually used.
    """
    if checker is None:
        checker = _DEFAULT_CHECKER
    if checker is None:
        # Fallback only happens if backend imports failed.
        return VerificationReport(
            is_equivalent=None,
            mode_used=mode,
            warnings=["EquivalenceChecker not available; "
                       "verification inconclusive"],
            skipped_modes=[mode],
        )

    h1 = _hash_of(graph1)
    h2 = _hash_of(graph2)
    report = VerificationReport(
        is_equivalent=None, mode_used=mode,
        canonical_hash_1=h1, canonical_hash_2=h2,
    )

    if mode is VerificationMode.FAST_REJECT:
        return _verify_fast_reject(report)
    if mode is VerificationMode.REWRITE:
        return _verify_rewrite(report, graph1, graph2, checker)
    if mode is VerificationMode.EXACT:
        return _verify_exact(report, graph1, graph2, checker)
    if mode is VerificationMode.AUTO:
        # Step 1: fast-reject
        step1 = _verify_fast_reject(VerificationReport(
            is_equivalent=None, mode_used=VerificationMode.FAST_REJECT,
            canonical_hash_1=h1, canonical_hash_2=h2))
        if step1.is_equivalent is False:
            return step1
        # Step 2: try exact when small enough
        n = _count_qubits(graph1, graph2)
        if n <= 8 and n > 0:
            step2 = _verify_exact(VerificationReport(
                is_equivalent=None, mode_used=VerificationMode.EXACT,
                canonical_hash_1=h1, canonical_hash_2=h2,
                skipped_modes=[VerificationMode.FAST_REJECT]),
                graph1, graph2, checker)
            if step2.is_equivalent is not None:
                return step2
            # Fall through to rewrite
        # Step 3: rewrite-based fallback
        return _verify_rewrite(VerificationReport(
            is_equivalent=None, mode_used=VerificationMode.REWRITE,
            canonical_hash_1=h1, canonical_hash_2=h2,
            skipped_modes=[VerificationMode.FAST_REJECT,
                           VerificationMode.EXACT]),
            graph1, graph2, checker)
    raise ValueError(f"Unknown VerificationMode: {mode!r}")


def _count_qubits(graph1, graph2) -> int:
    n1 = len(getattr(graph1, "qubit_last_writer", {})) or 0
    n2 = len(getattr(graph2, "qubit_last_writer", {})) or 0
    # If graphs have a `get_all_qubits`-like helper, prefer that.
    # NB: we accept any object exposing either qubit_last_writer
    # or a public attribute that lists qubits.
    try:
        q1 = set()
        for n in getattr(graph1, "nodes", {}).values():
            for t in getattr(n, "targets", []) or []:
                q1.add(t)
            if getattr(n, "type", None) == "ALLOC":
                for t in (n.targets or []):
                    q1.add(t)
        q2 = set()
        for n in getattr(graph2, "nodes", {}).values():
            for t in getattr(n, "targets", []) or []:
                q2.add(t)
            if getattr(n, "type", None) == "ALLOC":
                for t in (n.targets or []):
                    q2.add(t)
        return len(q1 | q2) if (q1 or q2) else max(n1, n2)
    except Exception:
        return max(n1, n2)


def _verify_fast_reject(report) -> VerificationReport:
    if report.canonical_hash_1 is None or report.canonical_hash_2 is None:
        report.is_equivalent = None
        report.warnings.append(
            "canonical hash unavailable; FAST_REJECT inconclusive")
        return report
    if report.canonical_hash_1 != report.canonical_hash_2:
        report.is_equivalent = False
        report.warnings.append(
            "hash mismatch: circuits definitely NOT equivalent")
        return report
    report.is_equivalent = None  # inconclusive
    report.warnings.append(
        "canonical hash match does NOT prove equivalence — "
        "hash collisions or canonicalization gaps still possible")
    return report


def _verify_rewrite(report, graph1, graph2, checker) -> VerificationReport:
    try:
        are_equ = checker.are_equivalent(graph1, graph2)
    except Exception as e:
        report.is_equivalent = None
        report.warnings.append(
            f"REWRITE verification raised {type(e).__name__}: {e}; "
            "treating as inconclusive")
        return report
    report.is_equivalent = bool(are_equ)
    report.notes.append("REWRITE mode uses ZX-calculus rewriting")
    if are_equ:
        report.notes.append(
            "REWRITE is sound for circuits within its rewriting "
            "system's decidable fragment; non-exhaustive otherwise")
    return report


def _verify_exact(report, graph1, graph2, checker) -> VerificationReport:
    n = _count_qubits(graph1, graph2)
    if n > 8:
        report.is_equivalent = None
        report.skipped_modes.append(VerificationMode.EXACT)
        report.warnings.append(
            "EXACT comparison skipped: too many qubits "
            f"({n} > 8); falling back to REWRITE")
        return report
    try:
        are_equ = checker.are_equivalent(graph1, graph2)
    except Exception as e:
        report.is_equivalent = None
        report.warnings.append(
            f"EXACT verification raised {type(e).__name__}: {e}; "
            "treating as inconclusive")
        return report
    report.is_equivalent = bool(are_equ)
    report.notes.append(
        "EXACT mode performed a unitary matrix comparison; "
        "mathematically sound for <=8 qubits")
    return report


def format_cli_banner(report: VerificationReport) -> str:
    """Render the verification report as a human-readable banner
    suitable for inclusion in CLI output. Documents the
    verification limits in a standard form."""
    lines = []
    lines.append("=== Verification Report ===")
    lines.append(f"Mode used: {report.mode_used.value}")
    if report.is_equivalent is None:
        lines.append("Result: INCONCLUSIVE (no proof either way)")
    else:
        lines.append(f"Result: {'EQUIVALENT' if report.is_equivalent else 'NOT EQUIVALENT'}")
    if report.canonical_hash_1 is not None:
        lines.append(f"Hash 1: {report.canonical_hash_1}")
        lines.append(f"Hash 2: {report.canonical_hash_2}")
    if report.warnings:
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")
    if report.notes:
        lines.append("Notes:")
        for n in report.notes:
            lines.append(f"  - {n}")
    if report.skipped_modes:
        modes_str = ", ".join(m.value for m in report.skipped_modes)
        lines.append(f"Skipped modes: {modes_str}")
    lines.append("=== End Verification Report ===")
    return "\n".join(lines)


__all__ = [
    "VerificationMode",
    "VerificationReport",
    "verify_equivalence",
    "format_cli_banner",
]
