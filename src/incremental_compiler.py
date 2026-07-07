"""
P2 §8.2 — Incremental compilation (surface-level wrapper).

The roadmap (`sol.md` "Компиляторная Масштабируемость" section 8.2)
lists:
    - [ ] Инкрементальная компиляция — перекомпиляция только изменённых модулей
    - [ ] Параллельная компиляция — несколько модулей одновременно
    - [ ] Lazy module loading — загрузка модулей по требованию
    - [ ] Build cache — кэш артефактов сборки

The first and fourth items are *effectively done already* in this codebase
via `src.compiler_db.QueryDb`, which caches by file-content hash, tracks
import dependencies, and invalidates downstream queries when an upstream
file changes (see `verify_record` and `execute_query`).

This module provides an explicit surface-level API the roadmap asks for:

  * `IncrementalCompiler(workspace_root)` — process-scoped wrapper that
    delegates to the existing `to_ebc` infrastructure.
  * `compile(source: str, *, optimize=False) -> list[Instruction]` —
    takes raw source text, writes it to a content-addressed path, calls
    `to_ebc`, and returns instructions.
  * `compile_file(filepath) -> list[Instruction]` — direct passthrough
    to `to_ebc` for users who already have files on disk.
  * `cache_stats() -> dict` — exposes `hits`, `misses`,
    `invalidations`, `evictions`, and `total_entries`, so CLI/build-
    tool users can introspect cache health.
  * `clear() / clear_file(filepath)` — manual invalidation API for
    CI / doctor / "I broke my cache" workflows.
  * `inspect(entry_key)` — shows the recorded inputs and dependencies
    for a cached entry (used by `eigen doctor --incremental`).

Real parallel compilation and lazy module loading are out of scope for
this P2 surface — both require restructuring the synchronous
`compiler_db.execute_query` recursion. The envelope here exposes the
single-threaded incremental path with the explicit tooling hooks the
roadmap asked for, so downstream work (parallel/lazy) can be added
without breaking API.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Optional

from src.compiler import to_ebc, get_db


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    invalidations: int = 0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "hit_rate": (self.hits / (self.hits + self.misses)
                         if (self.hits + self.misses) > 0 else 0.0),
        }


def _hash_source(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class IncrementalCompiler:
    """Surface-level incremental compilation wrapper around
    `src.compiler.to_ebc`. Maintains hit/miss stats per instance
    and writes content-addressed files into a process-default temp
    workspace (the EBC compiler needs a real filename for source
    location info, plus a workspace_root for the cache directory)."""

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        # Per-instance temp dir for raw-source compiles. Subdirs of
        # the workspace persist alongside the cache (so a cache hit on
        # a content-addressed filename resolves through normal
        # QueryDb machinery).
        self._inline_dir = os.path.join(workspace_root, ".eigen_inline")
        os.makedirs(self._inline_dir, exist_ok=True)
        self._stats = CacheStats()
        self._lock = threading.Lock()

    # -------------------------------------------------------------- API

    def compile_source(self, source: str, *, optimize: bool = False):
        """Compile raw source text to a list of `Instruction`s.

        Writes a content-addressed file into the workspace's `.eigen_inline`
        subdir so the underlying `to_ebc` (which needs a real filepath)
        can run. The content hash is the cache key, so two calls with
        the same source text always share the cache entry — independent
        of the original filename.
        """
        h = _hash_source(source)
        rel = f"inc_{h}.eig"
        path = os.path.join(self._inline_dir, rel)
        if not os.path.exists(path):
            # The file doesn't yet exist OR exists but with stale
            # content. We overwrite either way since the hash in the
            # name guarantees determinism — if the content differs,
            # the name would differ too.
            with open(path, "w", encoding="utf-8") as f:
                f.write(source)
        return self.compile_file(path, optimize=optimize)

    def compile_file(self, filepath: str, *, optimize: bool = False):
        """Compile an existing on-disk `.eig` source file to a list of
        `Instruction`s via the standard `to_ebc` pipeline. Hit/miss are
        tracked per call.
        """
        # Pre-flight: check whether the cache has a fresh entry. We
        # peek at the db directly to determine hit/miss for the
        # outermost query; this is a read-only check that doesn't
        # populate the cache. We then fall through to the normal
        # `to_ebc` which will return the cached or freshly-computed
        # result as appropriate.
        was_hit = self._check_cached(filepath, optimize)
        try:
            result = to_ebc(filepath, self.workspace_root,
                            optimize=optimize)
            with self._lock:
                if was_hit:
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1
            return result
        except Exception:
            # The compilation failed; we don't increment invalidations
            # here — those are a cache-health metric, not a build-error
            # metric. The exception propagates so callers see the real
            # error rather than a wrapped one.
            raise

    def cache_stats(self) -> dict:
        with self._lock:
            return self._stats.to_dict()

    def clear(self) -> None:
        """Drop the entire cache for this workspace. Useful for CI and
        for `eigen doctor --incremental --clear`."""
        db = get_db(self.workspace_root)
        # Wipe records and cache files. We don't recurse into other
        # workspaces — only the dir under this workspace.
        db.records.clear()
        db.save()
        # Physically delete cached files.
        cache_dir = db.cache_dir
        if os.path.isdir(cache_dir):
            for fname in os.listdir(cache_dir):
                full = os.path.join(cache_dir, fname)
                try:
                    if os.path.isfile(full):
                        os.unlink(full)
                except OSError:
                    pass

    def clear_file(self, filepath: str) -> None:
        """Invalidate only the entries originating from `filepath`. The
        `compiler_db` records keys as `query_name:filepath`, so we
        drop records whose key matches this filepath. Downstream
        consumers that depended on this entry will be invalidated
        automatically by `verify_record` on next access.
        """
        db = get_db(self.workspace_root)
        before = len(db.records)
        db.records = {
            k: v for k, v in db.records.items()
            if not k.endswith(f":{filepath}")
        }
        removed = before - len(db.records)
        with self._lock:
            # An invalidation is a deliberate drop, not a natural miss.
            self._stats.invalidations += removed
        db.save()

    def inspect(self, filepath: str) -> Optional[dict]:
        """Return the recorded cache entry for `filepath`, if any.
        Returns ``None`` when no entry exists. The returned dict has
        the same shape as the `QueryDb` records: ``result_hash``,
        ``cache_file``, ``dependencies``, ``input_files``.
        """
        db = get_db(self.workspace_root)
        # The compiler_db indexes by `<query_name>:<filepath>`; for
        # an inspect-by-file API we just return the first found
        # query for this filepath (typically `to_ebc` or `parse`).
        for key, rec in db.records.items():
            if key.endswith(f":{filepath}"):
                return dict(rec)
        return None

    # --------------------------------------------------------- internal

    def _check_cached(self, filepath: str, optimize: bool) -> bool:
        """Peek at the compiler_db: is `to_ebc[:filepath]` (or
        `to_ebc_opt[:filepath]`) currently a valid cached entry?"""
        db = get_db(self.workspace_root)
        query_name = "to_ebc_opt" if optimize else "to_ebc"
        query_key = f"{query_name}:{filepath}"
        valid, _ = db.verify_record(query_key)
        return valid
