"""§5.3 — Runtime Adaptation.

Roadmap checkboxes (3 items):

    - [x] Memory-aware execution — switching strategies when
          approaching memory limits.
    - [x] Graceful degradation — lower precision instead of crash
          when resources are scarce.
    - [x] Auto-scaling — automatic scaling of parallelism.

This module is an envelope over the existing simulation pipeline
that:

  1. `MemoryBudget` dataclass: defines a hard memory cap (in
     bytes) plus thresholds (warn / danger / hard).
  2. `PrecisionLevel` enum: full (double) / half (32-bit-
     compatible single precision) / reduced (16-bit float /
     log-domain) / symbolic.
  3. `AdaptationDecision` dataclass: captures a decision to
     adjust precision or parallelism, with a reason string.
  4. `RuntimeAdapter` orchestrator: at runtime, given the
     current memory usage and a "progress" tick, decides
     whether to:
        - keep strategy
        - bump precision down
        - reduce parallelism
        - emit a warning
  5. `AutoScaler` reads a "queue depth" or pending shots and
     adjusts parallelism (workers count) up/down within a
     configured range, respecting min/max bounds.

The envelope is non-intrusive: callers pass their existing
parameters and the adapter returns `AdaptationDecision`s that
callers may apply at their leisure.
"""
from __future__ import annotations

import dataclasses
import enum
import threading
import typing


# ---------------------------------------------------------------------------
# Precision level
# ---------------------------------------------------------------------------

class PrecisionLevel(enum.Enum):
    DOUBLE = "double"      # full 64-bit float
    SINGLE = "single"      # 32-bit float
    HALF = "half"          # 16-bit float (or log-domain surrogate)
    SYMBOLIC = "symbolic"  # no numerical evaluation — symbolic result


@dataclasses.dataclass(frozen=True)
class PrecisionLevelOrder:
    """Helper for comparing precision levels."""
    @staticmethod
    def ordinal(level: PrecisionLevel) -> int:
        return {PrecisionLevel.DOUBLE: 0,
                 PrecisionLevel.SINGLE: 1,
                 PrecisionLevel.HALF: 2,
                 PrecisionLevel.SYMBOLIC: 3}[level]


# ---------------------------------------------------------------------------
# Memory budget
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MemoryBudget:
    """Memory budget in bytes."""
    byte_limit: int
    warn_threshold: float = 0.7   # 70% of byte_limit
    danger_threshold: float = 0.85
    hard_threshold: float = 0.95

    def __post_init__(self):
        if not (0.0 < self.warn_threshold < self.danger_threshold
                < self.hard_threshold <= 1.0):
            raise ValueError(
                "MemoryBudget thresholds must satisfy "
                "0 < warn < danger < hard <= 1")

    def fraction_used(self, bytes_used: int) -> float:
        if self.byte_limit <= 0:
            return 1.0
        return bytes_used / self.byte_limit

    def zone(self, bytes_used: int) -> str:
        """Returns 'ok', 'warn', 'danger', 'hard'."""
        f = self.fraction_used(bytes_used)
        if f >= self.hard_threshold:
            return "hard"
        if f >= self.danger_threshold:
            return "danger"
        if f >= self.warn_threshold:
            return "warn"
        return "ok"


# ---------------------------------------------------------------------------
# Decision envelope
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AdaptationDecision:
    """A decision returned by the `RuntimeAdapter`."""
    new_precision: typing.Optional[PrecisionLevel] = None
    new_worker_count: typing.Optional[int] = None
    reason: str = ""
    severity: str = "info"  # "info", "warn", "danger"

    def is_noop(self) -> bool:
        return (self.new_precision is None
                and self.new_worker_count is None)


# ---------------------------------------------------------------------------
# Runtime adapter
# ---------------------------------------------------------------------------

class RuntimeAdapter:
    """Adapts runtime parameters based on memory usage.

    Decisions:
      - "ok"     → noop
      - "warn"   → no change, severity=warn
      - "danger" → drop precision by one level (e.g. DOUBLE → SINGLE)
      - "hard"   → drop precision one level AND halve worker count,
                   severity=danger

    The adapter keeps the current state (`precision` and
    `worker_count`) up-to-date as decisions are made and
    `apply_decision` is called.
    """
    def __init__(self, budget: MemoryBudget, *,
                 initial_precision: PrecisionLevel = PrecisionLevel.DOUBLE,
                 initial_worker_count: int = 1,
                 min_workers: int = 1,
                 max_workers: int = 16):
        self.budget = budget
        self.precision = initial_precision
        self.worker_count = initial_worker_count
        self.min_workers = min_workers
        self.max_workers = max_workers
        if not (1 <= min_workers <= max_workers):
            raise ValueError("min_workers must satisfy "
                              "1 <= min_workers <= max_workers")
        if not (min_workers <= initial_worker_count <= max_workers):
            raise ValueError("initial_worker_count must fall within "
                              "[min_workers, max_workers]")
        self._lock = threading.Lock()
        self.events: typing.List[AdaptationDecision] = []

    def decide(self, bytes_used: int,
                pending_units: int = 0) -> AdaptationDecision:
        """Inspect the current memory usage and decide whether
        to adapt. The decision is NOT applied automatically —
        callers should `apply_decision` if they want to mutate
        state (and most do)."""
        zone = self.budget.zone(bytes_used)
        if zone == "ok":
            return AdaptationDecision(reason="memory OK", severity="info")
        if zone == "warn":
            return AdaptationDecision(reason="memory warn",
                                         severity="warn")
        if zone == "danger":
            np = self._lower_precision(self.precision)
            if np is None:
                # Already at the lowest precision; emit a
                # warning but no change.
                return AdaptationDecision(reason="already at lowest "
                                              "precision; further "
                                              "degradation impossible",
                                              severity="danger")
            return AdaptationDecision(new_precision=np,
                                         reason="memory danger: "
                                              "lowering precision",
                                         severity="danger")
        # hard
        target_workers = max(self.min_workers,
                              self.worker_count // 2)
        np = self._lower_precision(self.precision)
        if np is None:
            return AdaptationDecision(new_worker_count=target_workers,
                                         reason="memory hard: already at "
                                              "lowest precision; halving "
                                              "workers",
                                         severity="danger")
        return AdaptationDecision(new_precision=np,
                                     new_worker_count=target_workers,
                                     reason="memory hard: lowering "
                                          "precision + halving workers",
                                     severity="danger")

    def apply_decision(self, d: AdaptationDecision) -> None:
        """Apply `d` to the adapter's internal state. Records the
        event for later inspection."""
        with self._lock:
            if d.new_precision is not None:
                self.precision = d.new_precision
            if d.new_worker_count is not None:
                self.worker_count = d.new_worker_count
            self.events.append(d)

    def step(self, bytes_used: int,
              pending_units: int = 0) -> AdaptationDecision:
        """Decide + apply in one call."""
        d = self.decide(bytes_used, pending_units)
        self.apply_decision(d)
        return d

    def reset_to(self, *,
                  precision: PrecisionLevel = PrecisionLevel.DOUBLE,
                  worker_count: int = 1) -> None:
        with self._lock:
            self.precision = precision
            self.worker_count = worker_count
            self.events.clear()

    @staticmethod
    def _lower_precision(level: PrecisionLevel):
        try:
            ord_ = PrecisionLevelOrder.ordinal(level)
        except KeyError:
            return None
        if level is PrecisionLevel.SYMBOLIC:
            return None
        return list(PrecisionLevel)[ord_ + 1]


# ---------------------------------------------------------------------------
# Auto-scaler (parallelism)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AutoScalerConfig:
    min_workers: int = 1
    max_workers: int = 16
    # When the queue depth exceeds `scale_up_threshold` * worker_count,
    # we add workers (up to max_workers). When the queue depth drops
    # below `scale_down_threshold` * worker_count, we remove workers
    # (down to min_workers).
    scale_up_threshold: float = 2.0
    scale_down_threshold: float = 0.5
    # Step sizes for scaling up / down.
    scale_up_step: int = 1
    scale_down_step: int = 1


class AutoScaler:
    """Scales worker count based on queue depth. The
    `pending_units` integer represents a queue depth (number of
    shots / batches / tasks not yet started)."""
    def __init__(self, config: AutoScalerConfig, *,
                  initial_workers: int = 1):
        if not (config.min_workers <= initial_workers
                <= config.max_workers):
            raise ValueError("initial_workers out of bounds")
        self.config = config
        self.worker_count = initial_workers
        self._lock = threading.Lock()
        self.events: typing.List[AdaptationDecision] = []

    def decide(self, pending_units: int) -> AdaptationDecision:
        """Decide whether to scale up/down/idle."""
        depth_per_worker = pending_units / max(self.worker_count, 1)
        if depth_per_worker > self.config.scale_up_threshold:
            new_count = min(self.worker_count + self.config.scale_up_step,
                              self.config.max_workers)
            return AdaptationDecision(
                new_worker_count=new_count,
                reason=f"queue depth {pending_units} above scale-up "
                      f"threshold; adding workers", severity="info",
            )
        if depth_per_worker < self.config.scale_down_threshold:
            new_count = max(self.worker_count - self.config.scale_down_step,
                              self.config.min_workers)
            return AdaptationDecision(
                new_worker_count=new_count,
                reason=f"queue depth {pending_units} below scale-down "
                      f"threshold; removing workers", severity="info",
            )
        return AdaptationDecision(reason="queue depth acceptable",
                                     severity="info")

    def apply_decision(self, d: AdaptationDecision) -> None:
        with self._lock:
            if d.new_worker_count is not None:
                self.worker_count = d.new_worker_count
            self.events.append(d)

    def step(self, pending_units: int) -> AdaptationDecision:
        d = self.decide(pending_units)
        self.apply_decision(d)
        return d


# ---------------------------------------------------------------------------
# Combined adapter: memory + parallelism
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CombinedRuntimeState:
    bytes_used: int
    pending_units: int
    workers: int
    precision: PrecisionLevel


class CombinedRuntimeAdapter:
    """A wrapper that combines `RuntimeAdapter` (memory-aware
    precision) and `AutoScaler` (queue-aware parallelism)."""
    def __init__(self, budget: MemoryBudget,
                  *,
                  initial_precision: PrecisionLevel = PrecisionLevel.DOUBLE,
                  initial_workers: int = 1,
                  scaler_config: typing.Optional[AutoScalerConfig] = None,
                  min_workers: int = 1,
                  max_workers: int = 16):
        self.memory_adapter = RuntimeAdapter(
            budget, initial_precision=initial_precision,
            initial_worker_count=initial_workers,
            min_workers=min_workers, max_workers=max_workers)
        if scaler_config is None:
            scaler_config = AutoScalerConfig(min_workers=min_workers,
                                              max_workers=max_workers)
        self.scaler = AutoScaler(scaler_config,
                                   initial_workers=initial_workers)
        self._lock = threading.Lock()

    def step(self, bytes_used: int, pending_units: int) -> AdaptationDecision:
        """Combine memory and queue decisions into a single
        AdaptationDecision. The two decisions are merged: memory
        wins on precision, queue wins on worker count (but a
        memory-hard decision may pre-empt the queue's choice)."""
        with self._lock:
            mem_d = self.memory_adapter.decide(bytes_used, pending_units)
            # Apply memory decision first (memory is the more
            # critical resource).
            self.memory_adapter.apply_decision(mem_d)
            # If memory said "hard", it may already have set a
            # worker_count. In that case we skip the scaler.
            if mem_d.new_worker_count is not None:
                # The memory adapter's hard-dropped worker count
                # becomes the new scaler baseline.
                # We update the scaler's worker count to match.
                self.scaler.worker_count = mem_d.new_worker_count
                return mem_d
            # Otherwise, run the scaler.
            scale_d = self.scaler.decide(pending_units)
            self.scaler.apply_decision(scale_d)
            # Merge into the memory decision (no overlap on
            # fields).
            return AdaptationDecision(
                new_precision=mem_d.new_precision,
                new_worker_count=scale_d.new_worker_count,
                reason=(mem_d.reason + "; " + scale_d.reason)
                       if scale_d.new_worker_count is not None
                       else mem_d.reason,
                severity=max(mem_d.severity, scale_d.severity),
            )

    def state(self) -> CombinedRuntimeState:
        with self._lock:
            return CombinedRuntimeState(
                bytes_used=0, pending_units=0,
                workers=self.scaler.worker_count,
                precision=self.memory_adapter.precision,
            )


__all__ = [
    "PrecisionLevel",
    "MemoryBudget",
    "AdaptationDecision",
    "RuntimeAdapter",
    "AutoScalerConfig",
    "AutoScaler",
    "CombinedRuntimeAdapter",
    "CombinedRuntimeState",
]
