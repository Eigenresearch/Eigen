"""
P2 §8.2 — Incremental compilation wrapper tests.

See `src/incremental_compiler.py` for docs. The wrapper is a thin
shell around the existing `compiler_db.QueryDb`-backed `to_ebc`
compile pipeline; tests pin its observable behaviours:

  * Repeated compiles of the same file path hit the cache after the
    first call (no disk I/O beyond the cache metadata check).
  * Compiles via `compile_source(source)` with the same source text
    are content-addressed — two different paths with identical
    content resolve to the same cached artifact.
  * `clear_file(path)` invalidates only the records for that path,
    leaving unrelated entries intact.
  * `clear()` wipes everything for the workspace.
  * `cache_stats()` reports hits, misses, invalidations.
  * `inspect(path)` exposes the recorded dependencies + input files.
"""
import os
import shutil
import tempfile
import unittest

from src.incremental_compiler import IncrementalCompiler


_OK_SRC_A = """eigen 1.0
func compute() -> int {
    return 1 + 2
}
let result: int = compute()
"""


_OK_SRC_B = """eigen 1.0
func different() -> int {
    return 5 * 5
}
let result: int = different()
"""


class TestIncrementalCompiler(unittest.TestCase):

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="eigen_inc_test_")
        # Write a real .eig file for the path-based API.
        self.file_a = os.path.join(self.workspace, "compute.eig")
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write(_OK_SRC_A)
        self.compiler = IncrementalCompiler(self.workspace)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        shutil.rmtree(self.workspace, ignore_errors=True)

    def test_compile_file_returns_instructions(self):
        from src.backend.bytecode import Instruction
        out = self.compiler.compile_file(self.file_a)
        self.assertIsInstance(out, list)
        if out:
            self.assertIsInstance(out[0], Instruction)

    def test_repeated_compile_file_hits_cache(self):
        self.compiler.compile_file(self.file_a)
        self.compiler.compile_file(self.file_a)
        stats = self.compiler.cache_stats()
        self.assertEqual(stats["hits"], 1)

    def test_compile_source_content_addressed(self):
        # Two identical source strings via compile_source must share
        # the cache (both call paths should result in 1 miss + 1 hit).
        self.compiler.compile_source(_OK_SRC_A)
        self.compiler.compile_source(_OK_SRC_A)
        stats = self.compiler.cache_stats()
        # First call misses; second hits.
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_compile_source_different_strings_dont_hit(self):
        self.compiler.compile_source(_OK_SRC_A)
        self.compiler.compile_source(_OK_SRC_B)
        stats = self.compiler.cache_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 2)

    def test_clear_file_invalidates_only_target(self):
        # Compile two distinct files.
        file_b = os.path.join(self.workspace, "different.eig")
        with open(file_b, "w", encoding="utf-8") as f:
            f.write(_OK_SRC_B)
        self.compiler.compile_file(self.file_a)
        self.compiler.compile_file(file_b)
        # Invalidate just A.
        self.compiler.clear_file(self.file_a)
        # Recompile A — should miss (since we wiped its records).
        self.compiler.compile_file(self.file_a)
        # Recompile B — should still hit (we only cleared A).
        self.compiler.compile_file(file_b)
        stats = self.compiler.cache_stats()
        self.assertGreaterEqual(stats["invalidations"], 1)
        # Initial two + A's recompile-after-clear = 3 misses; B's
        # recompile hits cleanly.
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 3)

    def test_clear_wipes_all_entries(self):
        # Compile two distinct files.
        file_b = os.path.join(self.workspace, "different.eig")
        with open(file_b, "w", encoding="utf-8") as f:
            f.write(_OK_SRC_B)
        self.compiler.compile_file(self.file_a)
        self.compiler.compile_file(file_b)
        self.compiler.clear()
        # Recompile both: both should miss (records wiped).
        self.compiler.compile_file(self.file_a)
        self.compiler.compile_file(file_b)
        stats = self.compiler.cache_stats()
        self.assertEqual(stats["hits"], 0)
        # Initial two + recompiles after clear = 4 misses total.
        self.assertEqual(stats["misses"], 4)

    def test_cache_stats_initial_state(self):
        stats = self.compiler.cache_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["hit_rate"], 0.0)

    def test_cache_stats_hit_rate_after_one_hit_one_miss(self):
        self.compiler.compile_file(self.file_a)
        self.compiler.compile_file(self.file_a)
        stats = self.compiler.cache_stats()
        self.assertEqual(stats["hits"] + stats["misses"], 2)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_inspect_returns_record_or_none(self):
        # Before compiling: no record.
        self.assertIsNone(self.compiler.inspect(self.file_a))
        self.compiler.compile_file(self.file_a)
        rec = self.compiler.inspect(self.file_a)
        self.assertIsNotNone(rec)
        # The shape is the QueryDb record dict.
        self.assertIn("result_hash", rec)
        self.assertIn("cache_file", rec)
        self.assertIn("dependencies", rec)
        self.assertIn("input_files", rec)

    def test_inline_dir_is_under_workspace(self):
        # The content-addressed inline directory lives under the
        # workspace, so cache hits don't require copying source across
        # workspaces.
        self.assertTrue(
            os.path.isdir(os.path.join(self.workspace, ".eigen_inline")))


if __name__ == "__main__":
    unittest.main()
