"""§9.2 — Benchmark Infrastructure tests, organised by the four
roadmap checkboxes."""
import json
import os
import tempfile
import unittest

from src.benchmark_infrastructure import (
    BenchmarkRun,
    BenchmarkHistory,
    RegressionKind,
    RegressionReport,
    compare_against_baseline,
    format_ci_summary,
    alert_on_regression,
    should_ci_fail,
)


class TestBenchmarkRun(unittest.TestCase):
    def test_to_dict_includes_metadata(self):
        r = BenchmarkRun(name="bench1", version="2.7.0",
                           duration_ms=10.0, metadata={"x": 1})
        d = r.to_dict()
        self.assertEqual(d["name"], "bench1")
        self.assertEqual(d["metadata"], {"x": 1})

    def test_from_dict_round_trip(self):
        d = {"name": "a", "version": "v1", "duration_ms": 5.0,
              "metadata": {"n": 10}}
        r = BenchmarkRun.from_dict(d)
        self.assertEqual(r.name, "a")
        self.assertEqual(r.duration_ms, 5.0)
        self.assertEqual(r.metadata, {"n": 10})

    def test_from_dict_with_missing_metadata(self):
        d = {"name": "a", "version": "v1", "duration_ms": 5.0}
        r = BenchmarkRun.from_dict(d)
        self.assertEqual(r.metadata, {})


class TestBenchmarkHistory(unittest.TestCase):
    def test_add_one_run(self):
        h = BenchmarkHistory()
        h.add(BenchmarkRun(name="b", version="v1", duration_ms=1.0))
        self.assertEqual(len(h.runs), 1)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hist.json")
            h1 = BenchmarkHistory()
            h1.add(BenchmarkRun(name="b1", version="v1",
                                  duration_ms=2.5,
                                  metadata={"i": 1}))
            h1.add(BenchmarkRun(name="b2", version="v1",
                                  duration_ms=3.5))
            h1.save(path)
            h2 = BenchmarkHistory.load(path)
            self.assertEqual(len(h2.runs), 2)
            self.assertEqual(h2.runs[0].name, "b1")

    def test_load_missing_path_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nope.json")
            h = BenchmarkHistory.load(path)
            self.assertEqual(h.runs, [])

    def test_latest_run_for_last_added(self):
        h = BenchmarkHistory()
        h.add(BenchmarkRun(name="b", version="v1", duration_ms=1.0))
        h.add(BenchmarkRun(name="b", version="v1", duration_ms=2.0))
        latest = h.latest_run_for("b")
        self.assertEqual(latest.duration_ms, 2.0)

    def test_latest_run_for_version_filter(self):
        h = BenchmarkHistory()
        h.add(BenchmarkRun(name="b", version="v1", duration_ms=1.0))
        h.add(BenchmarkRun(name="b", version="v2", duration_ms=2.0))
        latest_v1 = h.latest_run_for("b", version="v1")
        self.assertEqual(latest_v1.duration_ms, 1.0)

    def test_latest_run_for_missing_returns_none(self):
        h = BenchmarkHistory()
        self.assertIsNone(h.latest_run_for("nope"))


class TestCompareAgainstBaseline(unittest.TestCase):
    def setUp(self):
        self.baseline = [
            BenchmarkRun(name="a", version="v", duration_ms=100.0),
            BenchmarkRun(name="b", version="v", duration_ms=50.0),
            BenchmarkRun(name="c", version="v", duration_ms=10.0),
        ]

    def test_no_change_is_pass(self):
        current = [
            BenchmarkRun(name="a", version="v", duration_ms=100.0),
        ]
        reports = compare_against_baseline(self.baseline, current)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].kind, RegressionKind.PASS)

    def test_regression_detected_at_110_percent(self):
        current = [
            BenchmarkRun(name="a", version="v", duration_ms=110.001),
            # 110.001 / 100 = 1.10001 ≥ 1.1 → regression
        ]
        reports = compare_against_baseline(self.baseline, current)
        self.assertEqual(reports[0].kind, RegressionKind.REGRESSION)

    def test_improvement_detected_at_90_percent(self):
        current = [
            BenchmarkRun(name="b", version="v", duration_ms=45.0),
            # 45 / 50 = 0.9 ≤ 0.9 → improvement
        ]
        reports = compare_against_baseline(self.baseline, current)
        self.assertEqual(reports[0].kind, RegressionKind.IMPROVEMENT)

    def test_pass_is_between_90_and_110_percent(self):
        current = [
            BenchmarkRun(name="c", version="v", duration_ms=11.0),
            # 11 / 10 = 1.1 → not <1.1 (regression)
        ]
        reports = compare_against_baseline(self.baseline, current)
        # 1.1 ratio exactly hit the threshold; we use `>=`, so this
        # is a regression.
        self.assertEqual(reports[0].kind, RegressionKind.REGRESSION)

    def test_baseline_zero_means_infinite_regression(self):
        baseline = [BenchmarkRun(name="a", version="v", duration_ms=0.0)]
        current = [BenchmarkRun(name="a", version="v", duration_ms=1.0)]
        reports = compare_against_baseline(baseline, current)
        self.assertEqual(reports[0].kind, RegressionKind.REGRESSION)
        self.assertTrue(reports[0].ratio > 1.0)

    def test_missing_baseline_or_current_returns_none(self):
        # Missing baseline + missing current → return empty.
        self.assertEqual(compare_against_baseline([], []), [])

    def test_custom_thresholds(self):
        custom = [
            BenchmarkRun(name="a", version="v", duration_ms=200.0),
        ]
        reports = compare_against_baseline(self.baseline, custom,
                                            regression_threshold=1.5,
                                            improvement_threshold=0.5)
        # 200/100 = 2.0 > 1.5 → regression.
        self.assertEqual(reports[0].kind, RegressionKind.REGRESSION)
        self.assertEqual(reports[0].regression_threshold, 1.5)
        self.assertEqual(reports[0].improvement_threshold, 0.5)


class TestFormatCiSummary(unittest.TestCase):
    def test_renders_markdown_table(self):
        reports = [
            RegressionReport(name="a",  # pass
                                baseline_duration_ms=100.0,
                                current_duration_ms=100.0,
                                ratio=1.0, percent_change=0.0,
                                kind=RegressionKind.PASS),
            RegressionReport(name="b",  # regression
                                baseline_duration_ms=50.0,
                                current_duration_ms=75.0,
                                ratio=1.5, percent_change=50.0,
                                kind=RegressionKind.REGRESSION),
        ]
        out = format_ci_summary(reports)
        self.assertIn("# Benchmark Regression Report", out)
        self.assertIn("| Benchmark |", out)
        self.assertIn("a", out)
        self.assertIn("b", out)
        self.assertIn("REGRESSION", out)

    def test_includes_summary_counts(self):
        reports = [
            RegressionReport(name="a",
                                baseline_duration_ms=100.0,
                                current_duration_ms=100.0,
                                ratio=1.0, percent_change=0.0,
                                kind=RegressionKind.PASS),
        ]
        out = format_ci_summary(reports)
        self.assertIn("Summary", out)
        self.assertIn("regression: 0", out)
        self.assertIn("pass: 1", out)


class TestAlerting(unittest.TestCase):
    def test_alert_on_regression_returns_regresions_only(self):
        reports = [
            RegressionReport(name="a",
                                baseline_duration_ms=100.0,
                                current_duration_ms=100.0,
                                ratio=1.0, percent_change=0.0,
                                kind=RegressionKind.PASS),
            RegressionReport(name="b",
                                baseline_duration_ms=100.0,
                                current_duration_ms=150.0,
                                ratio=1.5, percent_change=50.0,
                                kind=RegressionKind.REGRESSION),
        ]
        alerts = alert_on_regression(reports)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].name, "b")

    def test_should_ci_fail_true_when_any_regression(self):
        reports = [
            RegressionReport(name="a",
                                baseline_duration_ms=100.0,
                                current_duration_ms=100.0,
                                ratio=1.0, percent_change=0.0,
                                kind=RegressionKind.PASS),
            RegressionReport(name="b",
                                baseline_duration_ms=100.0,
                                current_duration_ms=150.0,
                                ratio=1.5, percent_change=50.0,
                                kind=RegressionKind.REGRESSION),
        ]
        self.assertTrue(should_ci_fail(reports))

    def test_should_ci_fail_false_when_all_pass(self):
        reports = [
            RegressionReport(name="a",
                                baseline_duration_ms=100.0,
                                current_duration_ms=100.0,
                                ratio=1.0, percent_change=0.0,
                                kind=RegressionKind.PASS),
        ]
        self.assertFalse(should_ci_fail(reports))


if __name__ == "__main__":
    unittest.main()
