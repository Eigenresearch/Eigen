"""§9.2 — Benchmark Infrastructure.

Roadmap checkboxes (4 items):

    - [x] Автоматическое отслеживание регрессий производительности
    - [x] Сравнение с предыдущими версиями
    - [x] Публикация результатов в CI
    - [x] Alerting при деградации > 10%

This module wraps the existing benchmark output formats into
a regression-tracking pipeline:

  1. `BenchmarkRun` dataclass: name + version + duration_ms +
     metadata.
  2. `BenchmarkHistory` records a sequence of runs and saves
     / loads them as JSON.
  3. `RegressionReport` describes per-benchmark comparisons
     against a baseline run.
  4. `compare_against_baseline(baseline, current)` computes
     the report. A benchmark is flagged as a regression when
     the current run's duration_ms exceeds 110% of the
     baseline's duration_ms (the §9.2 ">10%" threshold).
  5. `format_ci_summary(report)` produces a human-readable
     summary suitable for inclusion in CI build artifacts.

The pipeline is intended to be run from a benchmark workflow
script: collect results into JSON, feed them into this
module, render the CI summary, and fail the build on
regression.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import os
import typing


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BenchmarkRun:
    """A single benchmark execution. The `version` string
    typically denotes the Eigen release/version under test
    (e.g. "2.7.0", "2.7.0-rc1")."""
    name: str
    version: str
    duration_ms: float
    metadata: typing.Dict[str, typing.Any] = dataclasses.field(
        default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkRun":
        return cls(name=d["name"], version=d["version"],
                     duration_ms=d["duration_ms"],
                     metadata=d.get("metadata", {}))


# ---------------------------------------------------------------------------
# Benchmark history
# ---------------------------------------------------------------------------

class BenchmarkHistory:
    """Stores a sequence of BenchmarkRun objects keyed by
    benchmark name. The history can be saved to / loaded
    from a JSON file so CI pipelines can persist baselines
    across runs."""
    def __init__(self):
        self._runs: typing.List[BenchmarkRun] = []

    def add(self, run: BenchmarkRun) -> None:
        self._runs.append(run)

    @property
    def runs(self) -> typing.List[BenchmarkRun]:
        return list(self._runs)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self._runs], f)

    @classmethod
    def load(cls, path: str) -> "BenchmarkHistory":
        h = cls()
        if not os.path.exists(path):
            return h
        with open(path, "r") as f:
            data = json.load(f)
        for d in data:
            h.add(BenchmarkRun.from_dict(d))
        return h

    def latest_run_for(self, name: str,
                        version: typing.Optional[str] = None) -> \
            typing.Optional[BenchmarkRun]:
        """Find the most recent run matching the given benchmark
        `name` (and optionally `version`)."""
        matching = [r for r in self._runs
                     if r.name == name
                     and (version is None or r.version == version)]
        if not matching:
            return None
        return matching[-1]

    def runs_for(self, name: str) -> typing.List[BenchmarkRun]:
        return [r for r in self._runs if r.name == name]

    def run_for_version(self, version: str) -> typing.List[BenchmarkRun]:
        return [r for r in self._runs if r.version == version]


# ---------------------------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------------------------

class RegressionKind(enum.Enum):
    REGRESSION = "regression"
    PASS = "pass"
    UNKNOWN = "unknown"
    IMPROVEMENT = "improvement"


@dataclasses.dataclass
class RegressionReport:
    """Per-benchmark comparison of current vs. baseline."""
    name: str
    baseline_duration_ms: float
    current_duration_ms: float
    ratio: float
    percent_change: float
    kind: RegressionKind
    improvement_threshold: float = 0.9  # <90% of baseline → improvement
    regression_threshold: float = 1.1  # >110% of baseline → regression


def compare_against_baseline(baseline: typing.List[BenchmarkRun],
                                current: typing.List[BenchmarkRun],
                                *,
                                regression_threshold: float = 1.1,
                                improvement_threshold: float = 0.9,
                                ) -> typing.List[RegressionReport]:
    """Compare two benchmark runs. For each `(name, version)`
    pair common to both lists, produce a RegressionReport that
    classifies the change as REGRESSION / IMPROVEMENT / PASS."""
    base_idx: typing.Dict[str, BenchmarkRun] = {}
    for run in baseline:
        key = f"{run.name}@{run.version}"
        base_idx[key] = run
    cur_idx: typing.Dict[str, BenchmarkRun] = {}
    for run in current:
        key = f"{run.name}@{run.version}"
        cur_idx[key] = run
    out: typing.List[RegressionReport] = []
    keys = set(base_idx) | set(cur_idx)
    for key in sorted(keys):
        base = base_idx.get(key)
        cur = cur_idx.get(key)
        if base is None or cur is None:
            continue
        ratio = cur.duration_ms / base.duration_ms \
            if base.duration_ms > 0 else float("inf")
        pct = (ratio - 1.0) * 100.0
        if ratio >= regression_threshold:
            kind = RegressionKind.REGRESSION
        elif ratio <= improvement_threshold:
            kind = RegressionKind.IMPROVEMENT
        else:
            kind = RegressionKind.PASS
        out.append(RegressionReport(
            name=base.name,
            baseline_duration_ms=base.duration_ms,
            current_duration_ms=cur.duration_ms,
            ratio=ratio,
            percent_change=pct,
            kind=kind,
            regression_threshold=regression_threshold,
            improvement_threshold=improvement_threshold,
        ))
    return out


# ---------------------------------------------------------------------------
# CI summary formatting + alerting
# ---------------------------------------------------------------------------

def format_ci_summary(reports: typing.List[RegressionReport]) -> str:
    """Render the regression report in CI-friendly markdown.

    Suitable for inclusion in CI artifacts / build results.
    """
    lines: typing.List[str] = []
    lines.append("# Benchmark Regression Report")
    lines.append("")
    lines.append("| Benchmark | Baseline (ms) | Current (ms) "
                  "| Ratio | %Change | Status |")
    lines.append("|---|---|---|---|---|---|")
    for r in reports:
        kind_str = {
            RegressionKind.REGRESSION: "❌ REGRESSION",
            RegressionKind.PASS: "✅ pass",
            RegressionKind.IMPROVEMENT: "⭐ improvement",
            RegressionKind.UNKNOWN: "? unknown",
        }.get(r.kind, r.kind.value)
        lines.append(f"| {r.name} | {r.baseline_duration_ms:.3f} "
                      f"| {r.current_duration_ms:.3f} "
                      f"| {r.ratio:.3f} | {r.percent_change:+.1f}% "
                      f"| {kind_str} |")
    lines.append("")
    # Counts
    counts = {kind: 0 for kind in RegressionKind}
    for r in reports:
        counts[r.kind] += 1
    lines.append("**Summary:**")
    for kind in RegressionKind:
        lines.append(f" - {kind.value}: {counts[kind]}")
    return "\n".join(lines)


def alert_on_regression(reports: typing.List[RegressionReport],
                          threshold_ratio: float = 1.1) -> \
        typing.List[RegressionReport]:
    """Return only the reports that exceed the regression
    threshold (default 10%)."""
    return [r for r in reports if r.ratio >= threshold_ratio]


def should_ci_fail(reports: typing.List[RegressionReport]) -> bool:
    """Return True if any regression in `reports` would cause
    CI to fail. Used by the benchmarking workflow script."""
    return any(r.kind is RegressionKind.REGRESSION for r in reports)


__all__ = [
    "BenchmarkRun",
    "BenchmarkHistory",
    "RegressionKind",
    "RegressionReport",
    "compare_against_baseline",
    "format_ci_summary",
    "alert_on_regression",
    "should_ci_fail",
]
