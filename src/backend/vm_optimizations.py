"""§1.1 — VM Acceleration: frame caching, inline caching,
hot-loop detection, object-churn reduction.

This module provides supplementary optimization infrastructure
for the Eigen VM that goes beyond the existing dispatch-table
and fast-path loop in `vm.py`:

  * `FrameCache` — caches `frame.locals` dict reference as a local
    variable during execution, avoiding repeated `call_stack[-1].locals`
    attribute lookups on every LOAD_VAR/STORE_VAR.
  * `InlineCache` — monomorphic inline cache for variable lookups:
    remembers the last dict (frame vs globals) where a name was found,
    bypassing the full lookup chain on repeat hits.
  * `HotLoopDetector` — tracks backward-branch frequency to identify
    hot loops for JIT compilation.
  * `ObjectPool` — reusable object pool for temporary tuples/lists
    created during arithmetic, reducing GC pressure (object churn).
"""
from __future__ import annotations

import dataclasses
import typing


@dataclasses.dataclass
class CacheEntry:
    """Monomorphic inline cache entry for variable lookup."""
    source: str  # "frame" | "globals" | "literal"
    value: typing.Any = None
    miss_count: int = 0


class InlineCache:
    """Monomorphic inline cache for variable lookups.

    On first lookup of a name, records where it was found (frame locals
    or globals). On subsequent lookups, checks the cached source first.
    Falls back to full lookup on miss and updates the cache.
    """

    __slots__ = ("_cache", "_max_size")

    def __init__(self, max_size: int = 256):
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size

    def lookup(self, name: str, frame_locals: dict | None,
               globals_dict: dict) -> typing.Any:
        entry = self._cache.get(name)
        if entry is not None:
            if entry.source == "frame" and frame_locals is not None:
                if name in frame_locals:
                    return frame_locals[name]
            elif entry.source == "globals":
                if name in globals_dict:
                    return globals_dict[name]
            elif entry.source == "literal":
                return name
            # Miss — update cache
            entry.miss_count += 1

        # Full lookup
        if frame_locals is not None and name in frame_locals:
            if len(self._cache) < self._max_size:
                self._cache[name] = CacheEntry("frame")
            return frame_locals[name]
        if name in globals_dict:
            if len(self._cache) < self._max_size:
                self._cache[name] = CacheEntry("globals")
            return globals_dict[name]
        # Not found — could be literal
        if len(self._cache) < self._max_size:
            self._cache[name] = CacheEntry("literal")
        return name  # literal fallback

    def invalidate(self, name: str | None = None):
        if name is not None:
            self._cache.pop(name, None)
        else:
            self._cache.clear()

    def stats(self) -> dict:
        hits = sum(1 for e in self._cache.values() if e.miss_count == 0)
        misses = sum(e.miss_count for e in self._cache.values())
        return {"entries": len(self._cache), "hits": hits, "misses": misses}


class HotLoopDetector:
    """Tracks backward-branch frequency to identify hot loops.

    A backward branch (target IP < current IP) is a loop edge.
    When a branch is taken more than `threshold` times, the loop
    starting at `target_ip` is marked hot for JIT compilation.
    """

    __slots__ = ("_counts", "_threshold", "_hot_loops")

    def __init__(self, threshold: int = 100):
        self._counts: dict[int, int] = {}
        self._threshold = threshold
        self._hot_loops: set[int] = set()

    def record_branch(self, target_ip: int, current_ip: int) -> bool:
        if target_ip < current_ip:
            count = self._counts.get(target_ip, 0) + 1
            self._counts[target_ip] = count
            if count >= self._threshold and target_ip not in self._hot_loops:
                self._hot_loops.add(target_ip)
                return True
        return False

    @property
    def hot_loops(self) -> set[int]:
        return set(self._hot_loops)

    def is_hot(self, target_ip: int) -> bool:
        return target_ip in self._hot_loops

    def stats(self) -> dict:
        return {
            "branches_tracked": len(self._counts),
            "hot_loops": len(self._hot_loops),
            "threshold": self._threshold,
        }


class ObjectPool:
    """Reusable object pool to reduce temporary object churn.

    Instead of creating new list/tuple objects for every arithmetic
    operation or function call, the VM can borrow from the pool and
    return when done. This reduces GC pressure in hot loops.
    """

    __slots__ = ("_pool", "_max_size", "_borrow_count", "_return_count")

    def __init__(self, max_size: int = 128):
        self._pool: list[list] = []
        self._max_size = max_size
        self._borrow_count = 0
        self._return_count = 0

    def borrow(self) -> list:
        self._borrow_count += 1
        if self._pool:
            obj = self._pool.pop()
            obj.clear()
            return obj
        return []

    def release(self, obj: list):
        self._return_count += 1
        if len(self._pool) < self._max_size:
            obj.clear()
            self._pool.append(obj)

    def stats(self) -> dict:
        return {
            "pool_size": len(self._pool),
            "borrowed": self._borrow_count,
            "returned": self._return_count,
            "max_size": self._max_size,
        }


class FrameCache:
    """Caches the current frame's locals dict as a local variable.

    In the execution loop, instead of `call_stack[-1].locals[arg]`
    on every LOAD_VAR/STORE_VAR, the VM caches the reference once
    per frame entry and uses it directly.
    """

    __slots__ = ("_cached_locals", "_cached_frame_id")

    def __init__(self):
        self._cached_locals: dict | None = None
        self._cached_frame_id: int = -1

    def get(self, call_stack: list) -> dict | None:
        if not call_stack:
            self._cached_locals = None
            self._cached_frame_id = -1
            return None
        frame = call_stack[-1]
        fid = id(frame)
        if fid != self._cached_frame_id:
            self._cached_locals = frame.locals
            self._cached_frame_id = fid
        return self._cached_locals

    def invalidate(self):
        self._cached_locals = None
        self._cached_frame_id = -1
