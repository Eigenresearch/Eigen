"""
P3 §12.3 — Reproducibility / ExperimentTracker tests.

Covers:
  * `ExperimentRun` round-trip via `to_dict`/`from_dict`.
  * `ExperimentTracker.record` assigns monotonic run_ids.
  * `ExperimentTracker.lookup` filters by (program_hash, parameters,
    tags, name).
  * Persistence: drop the tracker, build a new one over the same
    workdir, see the ledger reload.
  * `overwrite=True` replaces the in-place record but preserves the
    run_id; default appends.
  * `export_json` writes the same shape as the in-memory ledger.
  * `export_latex` writes a valid longtable block; TeX-escapes user
    text content.
  * `iter_runs` exposes every stored run.
  * `clear()` wipes everything.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from src.reproducibility.experiment_tracker import (
    ExperimentRun,
    ExperimentTracker,
)


class TestExperimentTracker(unittest.TestCase):

    def setUp(self):
        # Use a deterministic clock — tests want repeatable timestamps.
        clock_iter = iter([1_000_000.0 + i for i in range(100)])
        self.clock = lambda: next(clock_iter)
        self.workdir = tempfile.mkdtemp(prefix="eigen_repro_test_")
        self.tracker = ExperimentTracker(self.workdir, clock=self.clock)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _seed(self, count: int):
        for i in range(count):
            run = ExperimentRun(
                name=f"exp{i}",
                program_hash=f"hash{i:08}",
                parameters={"theta": i * 0.1, "shots": 100},
                simulator_config={"sim_type": "dense", "seed": 42},
                deterministic=True,
                tags=("bell",) if i % 2 == 0 else ("rb",),
            )
            self.tracker.record(run, {"survival": 1.0 - i * 0.05})

    # --------------------------------------------------- run data

    def test_run_to_dict_from_dict(self):
        run = ExperimentRun(name="exp1", program_hash="abc",
                            parameters={"theta": 1.5},
                            simulator_config={"seed": 7},
                            deterministic=True, tags=("A", "B"),
                            recorded_at=12345)
        d = run.to_dict()
        self.assertEqual(d["name"], "exp1")
        out = ExperimentRun.from_dict(d)
        self.assertEqual(out.name, "exp1")
        self.assertEqual(out.parameters, run.parameters)
        self.assertEqual(out.tags, run.tags)
        self.assertEqual(out.recorded_at, run.recorded_at)

    # ------------------------------------------------ record+lookup

    def test_record_assigns_run_ids(self):
        self._seed(3)
        ids = [e.run_id for e in self.tracker.iter_runs()]
        self.assertEqual(ids, ["run-0001", "run-0002", "run-0003"])

    def test_lookup_by_program_hash(self):
        self._seed(3)
        hits = self.tracker.lookup(program_hash="hash00000001")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0].name, "exp1")
        self.assertEqual(hits[0][1], {"survival": 0.95})

    def test_lookup_by_parameters(self):
        self._seed(3)
        hits = self.tracker.lookup(parameters={"theta": 0.2, "shots": 100})
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0].name, "exp2")

    def test_lookup_by_tag(self):
        self._seed(4)
        hits = self.tracker.lookup(tags=("bell",))
        self.assertEqual(len(hits), 2)
        self.assertEqual({h[0].name for h in hits}, {"exp0", "exp2"})

    def test_lookup_by_name(self):
        self._seed(3)
        hits = self.tracker.lookup(name="exp0")
        self.assertEqual(len(hits), 1)

    def test_lookup_no_filter_returns_all(self):
        self._seed(3)
        self.assertEqual(len(self.tracker.lookup()), 3)

    def test_lookup_no_match(self):
        self._seed(2)
        self.assertEqual(
            self.tracker.lookup(program_hash="nonexistent"), [])

    # ---- result caching: overwrite replaces, default appends

    def test_record_default_appends_new_entry(self):
        run = ExperimentRun(name="e1", program_hash="h1",
                            parameters={"a": 1})
        rid1 = self.tracker.record(run, {"r": "first"})
        rid2 = self.tracker.record(run, {"r": "second"})
        self.assertNotEqual(rid1, rid2)
        # Both records present.
        hits = self.tracker.lookup(program_hash="h1")
        self.assertEqual(len(hits), 2)

    def test_record_overwrite_replaces_in_place(self):
        run = ExperimentRun(name="e1", program_hash="h1",
                            parameters={"a": 1})
        rid1 = self.tracker.record(run, {"r": "first"})
        rid2 = self.tracker.record(run, {"r": "second"}, overwrite=True)
        self.assertEqual(rid1, rid2)
        hits = self.tracker.lookup(program_hash="h1")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1], {"r": "second"})

    # ------------------------------------------------------- persist

    def test_persistence_reload(self):
        self._seed(3)
        # Build a fresh tracker over the same workdir — the ledger
        # should reload from disk.
        clock_iter2 = iter([9_000_000.0 + i for i in range(100)])
        clock2 = lambda: next(clock_iter2)
        tracker2 = ExperimentTracker(self.workdir, clock=clock2)
        self.assertEqual(len(tracker2), 3)
        # The next record should get run-0004 (counter continues).
        new = ExperimentRun(name="e3", program_hash="h3",
                            parameters={"x": 0})
        rid = tracker2.record(new, {"r": "third"})
        self.assertEqual(rid, "run-0004")

    # ---------------------------------------------------- exports

    def test_export_json_matches_in_memory(self):
        self._seed(2)
        out_path = os.path.join(self.workdir, "exports.json")
        self.tracker.export_json(out_path)
        self.assertTrue(os.path.isfile(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.assertIn("entries", raw)
        self.assertEqual(len(raw["entries"]), 2)
        self.assertEqual(raw["entries"][0]["run"]["name"], "exp0")

    def test_export_latex_writes_longtable_block(self):
        self._seed(2)
        # Names with TeX-special chars to verify escaping.
        run = ExperimentRun(name="exp_with_underscores",
                            program_hash="hash",
                            parameters={"theta": 1.5})
        self.tracker.record(run, {"r": 1})
        out_path = os.path.join(self.workdir, "exports.tex")
        self.tracker.export_latex(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(r"\begin{longtable}", content)
        self.assertIn(r"\end{longtable}", content)
        self.assertIn("Run ID", content)
        # Underscore escaped:
        self.assertIn(r"\_", content)
        # At least one row contains a run_id pattern.
        self.assertIn("run-0001", content)
        # Some hash prefixed text appears.
        self.assertIn("hash", content)

    def test_export_latex_custom_columns(self):
        self._seed(1)
        out_path = os.path.join(self.workdir, "exports.tex")
        self.tracker.export_latex(out_path,
                                  columns=["run_id", "name", "deterministic"])
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Run ID", content)
        self.assertIn("Det.", content)
        self.assertIn("yes", content)

    def test_iter_runs_with_result(self):
        self._seed(2)
        observed = list(self.tracker.iter_runs(with_result=True))
        self.assertEqual(len(observed), 2)
        self.assertIsInstance(observed[0], tuple)
        self.assertEqual(observed[0][1].get("survival"), 1.0)

    def test_clear_wipes_ledger(self):
        self._seed(2)
        self.tracker.clear()
        self.assertEqual(len(self.tracker), 0)
        self.assertFalse(os.path.isfile(self.tracker.ledger_path))
        # Next record uses run-0001.
        run = ExperimentRun(name="post", program_hash="post",
                            parameters={})
        rid = self.tracker.record(run, {"r": 1})
        self.assertEqual(rid, "run-0001")

    def test_contains_run_id(self):
        run = ExperimentRun(name="e1", program_hash="h1",
                            parameters={"a": 1})
        rid = self.tracker.record(run, {"r": 1})
        self.assertIn(rid, self.tracker)
        self.assertNotIn("run-9999", self.tracker)


if __name__ == "__main__":
    unittest.main()
