"""§5.3 — Runtime Adaptation tests, organised by the three
roadmap checkboxes."""
import unittest

from src.runtime_adaptation import (
    PrecisionLevel,
    MemoryBudget,
    RuntimeAdapter,
    AutoScalerConfig,
    AutoScaler,
    CombinedRuntimeAdapter,
    CombinedRuntimeState,
)


# ----- MemoryBudget tests --------------------------------------------

class TestMemoryBudget(unittest.TestCase):
    def test_zone_ok(self):
        b = MemoryBudget(byte_limit=1000)
        self.assertEqual(b.zone(500), "ok")

    def test_zone_warn(self):
        b = MemoryBudget(byte_limit=1000)
        self.assertEqual(b.zone(750), "warn")  # 75% > 70% warn

    def test_zone_danger(self):
        b = MemoryBudget(byte_limit=1000)
        self.assertEqual(b.zone(870), "danger")  # 87% > 85% danger

    def test_zone_hard(self):
        b = MemoryBudget(byte_limit=1000)
        self.assertEqual(b.zone(950), "hard")  # 95% > 95% hard

    def test_thresholds_must_be_ordered(self):
        with self.assertRaises(ValueError):
            MemoryBudget(byte_limit=1000, warn_threshold=0.85,
                          danger_threshold=0.7)

    def test_thresholds_must_be_in_range(self):
        with self.assertRaises(ValueError):
            MemoryBudget(byte_limit=1000, warn_threshold=0.0)

    def test_fraction_used_returns_one_when_limit_is_zero(self):
        zero = MemoryBudget(byte_limit=0)
        self.assertEqual(zero.fraction_used(123), 1.0)


# ----- RuntimeAdapter tests ------------------------------------------

class TestRuntimeAdapter(unittest.TestCase):
    def setUp(self):
        self.budget = MemoryBudget(byte_limit=1000)
        self.adapter = RuntimeAdapter(self.budget)

    def test_initial_state(self):
        self.assertEqual(self.adapter.precision, PrecisionLevel.DOUBLE)
        self.assertEqual(self.adapter.worker_count, 1)

    def test_decide_ok_returns_noop(self):
        d = self.adapter.decide(500)  # 50%
        self.assertTrue(d.is_noop())
        self.assertEqual(d.severity, "info")

    def test_decide_warn_does_not_change_precision(self):
        d = self.adapter.decide(750)
        self.assertIsNone(d.new_precision)
        self.assertEqual(d.severity, "warn")

    def test_decide_danger_drops_precision(self):
        d = self.adapter.decide(870)
        self.assertEqual(d.new_precision, PrecisionLevel.SINGLE)
        self.assertEqual(d.severity, "danger")

    def test_decide_hard_drops_precision_and_workers(self):
        a = RuntimeAdapter(self.budget, initial_worker_count=4)
        d = a.decide(970)
        self.assertEqual(d.new_precision, PrecisionLevel.SINGLE)
        self.assertEqual(d.new_worker_count, 2)  # 4 // 2
        self.assertEqual(d.severity, "danger")

    def test_decide_hard_at_lowest_precision_emits_warning(self):
        a = RuntimeAdapter(self.budget,
                            initial_precision=PrecisionLevel.SYMBOLIC)
        d = a.decide(970)
        self.assertIsNone(d.new_precision)
        self.assertEqual(d.severity, "danger")

    def test_step_applies_decision(self):
        a = RuntimeAdapter(self.budget)
        a.step(870)  # drop SINGLE
        self.assertEqual(a.precision, PrecisionLevel.SINGLE)
        # drop again
        a.step(970)  # drop to HALF
        self.assertEqual(a.precision, PrecisionLevel.HALF)
        # drop again — HALF → SYMBOLIC
        a.step(970)
        self.assertEqual(a.precision, PrecisionLevel.SYMBOLIC)
        # drop cannot go further; should warn but not crash.
        a.step(970)
        self.assertEqual(a.precision, PrecisionLevel.SYMBOLIC)

    def test_records_events(self):
        a = RuntimeAdapter(self.budget, initial_worker_count=4)
        a.step(970)  # danger+hard → drop precision and workers
        self.assertEqual(len(a.events), 1)
        self.assertEqual(a.events[0].new_precision, PrecisionLevel.SINGLE)

    def test_worker_count_respects_min_workers(self):
        a = RuntimeAdapter(self.budget, initial_worker_count=2,
                            min_workers=2, max_workers=8)
        # 1 worker min, so dropping to half shouldn't go below 2.
        a.step(970)
        self.assertEqual(a.worker_count, 2)

    def test_bad_workers_args_raise(self):
        with self.assertRaises(ValueError):
            RuntimeAdapter(self.budget, initial_worker_count=0)
        with self.assertRaises(ValueError):
            RuntimeAdapter(self.budget, initial_worker_count=10,
                            min_workers=2, max_workers=4)

    def test_reset_to_clears_state(self):
        a = RuntimeAdapter(self.budget, initial_worker_count=4)
        a.step(970)
        self.assertEqual(a.precision, PrecisionLevel.SINGLE)
        a.reset_to(precision=PrecisionLevel.DOUBLE, worker_count=4)
        self.assertEqual(a.precision, PrecisionLevel.DOUBLE)
        self.assertEqual(a.worker_count, 4)
        self.assertEqual(len(a.events), 0)


# ----- AutoScaler tests -----------------------------------------------

class TestAutoScaler(unittest.TestCase):
    def test_decide_scale_up(self):
        scaler = AutoScaler(AutoScalerConfig(min_workers=1, max_workers=4),
                              initial_workers=1)
        d = scaler.decide(pending_units=10)  # depth per worker = 10
        self.assertEqual(d.new_worker_count, 2)

    def test_decide_scale_down(self):
        scaler = AutoScaler(AutoScalerConfig(min_workers=1, max_workers=4),
                              initial_workers=4)
        d = scaler.decide(pending_units=1)  # depth per worker = 0.25
        self.assertEqual(d.new_worker_count, 3)

    def test_decide_noop(self):
        scaler = AutoScaler(AutoScalerConfig(min_workers=1, max_workers=4),
                              initial_workers=2)
        d = scaler.decide(pending_units=2)  # 2/2 = 1.0 → noop
        self.assertTrue(d.is_noop())

    def test_scale_up_step_size(self):
        scaler = AutoScaler(AutoScalerConfig(min_workers=1, max_workers=10,
                                               scale_up_step=3),
                              initial_workers=1)
        d = scaler.decide(pending_units=10)
        self.assertEqual(d.new_worker_count, 4)

    def test_scale_down_step_size(self):
        scaler = AutoScaler(AutoScalerConfig(min_workers=1, max_workers=10,
                                               scale_down_step=2),
                              initial_workers=8)
        d = scaler.decide(pending_units=0)
        self.assertEqual(d.new_worker_count, 6)

    def test_scale_up_respects_max(self):
        scaler = AutoScaler(AutoScalerConfig(min_workers=1, max_workers=2),
                              initial_workers=2)
        d = scaler.decide(pending_units=100)
        self.assertEqual(d.new_worker_count, 2)  # at max

    def test_scale_down_respects_min(self):
        scaler = AutoScaler(AutoScalerConfig(min_workers=2, max_workers=8),
                              initial_workers=2)
        d = scaler.decide(pending_units=0)
        self.assertEqual(d.new_worker_count, 2)  # at min

    def test_initial_workers_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            AutoScaler(AutoScalerConfig(min_workers=1, max_workers=4),
                         initial_workers=10)


# ----- Combined adapter tests -----------------------------------------

class TestCombinedRuntimeAdapter(unittest.TestCase):
    def setUp(self):
        self.budget = MemoryBudget(byte_limit=1000)
        self.adapter = CombinedRuntimeAdapter(
            self.budget,
            initial_precision=PrecisionLevel.DOUBLE,
            initial_workers=4,
            scaler_config=AutoScalerConfig(min_workers=1, max_workers=8),
            min_workers=1, max_workers=8,
        )

    def test_state_reflects_initial(self):
        s = self.adapter.state()
        self.assertEqual(s.workers, 4)
        self.assertEqual(s.precision, PrecisionLevel.DOUBLE)

    def test_memory_hard_decision_wins_workers(self):
        # 97% memory usage → hard zone + halve workers
        d = self.adapter.step(970, pending_units=0)
        self.assertEqual(d.new_precision, PrecisionLevel.SINGLE)
        self.assertEqual(d.new_worker_count, 2)
        # The scaler's worker_count should be updated too.
        self.assertEqual(self.adapter.scaler.worker_count, 2)

    def test_queue_triggers_scale_up_when_memory_ok(self):
        d = self.adapter.step(200, pending_units=40)
        # Memory OK; queue depth large per worker (40/4=10 > 2).
        self.assertEqual(d.new_worker_count, 5)

    def test_queue_triggers_scale_down_when_memory_ok(self):
        d = self.adapter.step(200, pending_units=0)
        self.assertEqual(d.new_worker_count, 3)


class TestPrecisionLevelOrder(unittest.TestCase):
    def test_double_is_lowest(self):
        # Importing indirectly to verify the helper still works.
        from src.runtime_adaptation import PrecisionLevelOrder as O
        self.assertEqual(O.ordinal(PrecisionLevel.DOUBLE), 0)
        self.assertEqual(O.ordinal(PrecisionLevel.SINGLE), 1)
        self.assertEqual(O.ordinal(PrecisionLevel.HALF), 2)
        self.assertEqual(O.ordinal(PrecisionLevel.SYMBOLIC), 3)


# ----- Combined state dataclass ---------------------------------------

class TestCombinedRuntimeState(unittest.TestCase):
    def test_is_constructible(self):
        s = CombinedRuntimeState(bytes_used=100, pending_units=10,
                                    workers=4,
                                    precision=PrecisionLevel.DOUBLE)
        self.assertEqual(s.workers, 4)


if __name__ == "__main__":
    unittest.main()
