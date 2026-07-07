"""§6.3 — Hybrid Execution Orchestration tests, organised by the
four roadmap checkboxes."""
import os
import tempfile
import threading
import time
import unittest

from src.hybrid_orchestration import (
    TaskType,
    Task,
    TaskResult,
    ProviderKind,
    ProviderSession,
    SessionManager,
    CheckpointEntry,
    CheckpointManager,
    HybridPlan,
    HybridOrchestrator,
    OrchestrationError,
    CrossBackendSyncPoint,
    CrossBackendSynchronizer,
)


# ---------------------------------------------------------------------------
# Tasks and TaskResult
# ---------------------------------------------------------------------------

class TestTaskType(unittest.TestCase):
    def test_four_kinds(self):
        self.assertEqual({t.value for t in TaskType},
                          {"classical", "quantum", "barrier",
                           "condition"})


class TestTask(unittest.TestCase):
    def test_default_task_type_is_classical(self):
        t = Task(name="x", fn=lambda: 1)
        self.assertEqual(t.task_type, TaskType.CLASSICAL)
        self.assertEqual(t.depends_on, [])

    def test_explicit_depends_on(self):
        t = Task(name="x", fn=lambda: 1, depends_on=["y"])
        self.assertEqual(t.depends_on, ["y"])


class TestTaskResult(unittest.TestCase):
    def test_default_fields(self):
        r = TaskResult(name="x", task_type=TaskType.CLASSICAL,
                         succeeded=True)
        self.assertEqual(r.value, None)
        self.assertIsNone(r.error)
        self.assertEqual(r.checkpoint_data, {})
        self.assertEqual(r.wall_clock_ns, 0)


# ---------------------------------------------------------------------------
# ProviderSession & SessionManager
# ---------------------------------------------------------------------------

class TestProviderSession(unittest.TestCase):
    def test_default_session_id_is_uuid(self):
        s = ProviderSession(provider=ProviderKind.IBM, device="ibm_brisbane")
        self.assertTrue(s.session_id)
        self.assertEqual(s.state, "open")

    def test_close_sets_state(self):
        s = ProviderSession(provider=ProviderKind.IBM, device="ibm_brisbane")
        s.close()
        self.assertEqual(s.state, "closed")

    def test_is_expired(self):
        s = ProviderSession(provider=ProviderKind.IONQ,
                              device="ionq_simulator",
                              expires_at=100.0)
        self.assertTrue(s.is_expired(now=200.0))
        self.assertFalse(s.is_expired(now=50.0))

    def test_no_expiry_returns_not_expired(self):
        s = ProviderSession(provider=ProviderKind.BRAKET,
                              device="sv1")
        self.assertFalse(s.is_expired(now=1.0))


class TestSessionManager(unittest.TestCase):
    def test_open_and_get(self):
        sm = SessionManager()
        s = ProviderSession(provider=ProviderKind.IBM,
                              device="ibm_brisbane")
        sid = sm.open(s)
        self.assertEqual(sm.get(sid).device, "ibm_brisbane")

    def test_get_missing_returns_none(self):
        sm = SessionManager()
        self.assertIsNone(sm.get("nope"))

    def test_close_session(self):
        sm = SessionManager()
        s = ProviderSession(provider=ProviderKind.IBM, device="d")
        sid = sm.open(s)
        self.assertTrue(sm.close(sid))
        self.assertEqual(sm.get(sid).state, "closed")

    def test_close_missing_returns_false(self):
        sm = SessionManager()
        self.assertFalse(sm.close("nope"))

    def test_list_all(self):
        sm = SessionManager()
        sm.open(ProviderSession(provider=ProviderKind.IBM, device="d1"))
        sm.open(ProviderSession(provider=ProviderKind.IONQ, device="d2"))
        self.assertEqual(len(sm.all_sessions()), 2)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sessions.pkl")
            sm1 = SessionManager()
            sm1.open(ProviderSession(provider=ProviderKind.IBM,
                                       device="ibm_brisbane"))
            sm1.save(path)
            sm2 = SessionManager()
            sm2.load(path)
            self.assertEqual(len(sm2.all_sessions()), 1)
            self.assertEqual(sm2.all_sessions()[0].device, "ibm_brisbane")


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManager(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(dir_path=tmpdir)
            entry = CheckpointEntry(
                job_id="job1",
                completed_results={"t1": TaskResult(
                    name="t1", task_type=TaskType.CLASSICAL,
                    succeeded=True, value=42)},
                pending_task_names=["t2"],
                timestamp_ns=0)
            path = cm.save(entry)
            self.assertTrue(os.path.exists(path))
            loaded = cm.load("job1")
            self.assertEqual(loaded.job_id, "job1")
            self.assertIn("t1", loaded.completed_results)
            self.assertEqual(loaded.completed_results["t1"].value, 42)
            self.assertEqual(loaded.pending_task_names, ["t2"])

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(dir_path=tmpdir)
            self.assertIsNone(cm.load("nope"))

    def test_path_for_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(dir_path=tmpdir)
            p1 = cm.path_for("job_xyz")
            p2 = cm.path_for("job_xyz")
            self.assertEqual(p1, p2)
            self.assertIn("job_xyz", p1)


# ---------------------------------------------------------------------------
# HybridPlan
# ---------------------------------------------------------------------------

class TestHybridPlan(unittest.TestCase):
    def test_add_classical(self):
        plan = HybridPlan().add_classical("c1", lambda: 1)
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].task_type, TaskType.CLASSICAL)

    def test_add_quantum(self):
        plan = HybridPlan().add_quantum("q1", lambda: 2)
        self.assertEqual(plan.tasks[0].task_type, TaskType.QUANTUM)

    def test_add_barrier(self):
        plan = HybridPlan().add_barrier("b1")
        self.assertEqual(plan.tasks[0].task_type, TaskType.BARRIER)

    def test_add_condition(self):
        plan = HybridPlan().add_condition("cond1",
                                            lambda r: True)
        self.assertEqual(plan.tasks[0].task_type, TaskType.CONDITION)
        self.assertIsNotNone(plan.tasks[0].predicate)


# ---------------------------------------------------------------------------
# HybridOrchestrator
# ---------------------------------------------------------------------------

class TestHybridOrchestrator(unittest.TestCase):
    def test_simple_classical_run(self):
        plan = HybridPlan().add_classical("c1", lambda: 42)
        orch = HybridOrchestrator()
        results = orch.execute(plan)
        self.assertTrue(results["c1"].succeeded)
        self.assertEqual(results["c1"].value, 42)

    def test_classical_and_quantum(self):
        plan = (HybridPlan()
                .add_classical("c1", lambda: 1)
                .add_quantum("q1", lambda: 2)
                .add_classical("c2", lambda: 3))
        orch = HybridOrchestrator()
        results = orch.execute(plan)
        self.assertEqual(results["c1"].value, 1)
        self.assertEqual(results["q1"].task_type, TaskType.QUANTUM)
        self.assertEqual(results["q1"].value, 2)
        self.assertEqual(results["c2"].value, 3)

    def test_task_failure_recorded(self):
        def boom():
            raise ValueError("kaboom")
        plan = HybridPlan().add_classical("c1", boom)
        orch = HybridOrchestrator()
        results = orch.execute(plan)
        self.assertFalse(results["c1"].succeeded)
        self.assertIn("kaboom", results["c1"].error)

    def test_barrier_emits_event(self):
        plan = HybridPlan().add_barrier("b1")
        orch = HybridOrchestrator()
        orch.execute(plan)
        self.assertTrue(any("barrier" in e for e in orch.events))

    def test_missing_dep_records_error(self):
        plan = HybridPlan().add_classical("c2", lambda: 1,
                                            depends_on=["c1"])
        orch = HybridOrchestrator()
        results = orch.execute(plan)
        self.assertFalse(results["c2"].succeeded)
        self.assertIn("Missing dependency", results["c2"].error)

    def test_existing_dep_runs(self):
        plan = (HybridPlan()
                .add_classical("c1", lambda: 1)
                .add_classical("c2", lambda: 2, depends_on=["c1"]))
        orch = HybridOrchestrator()
        results = orch.execute(plan)
        self.assertTrue(results["c1"].succeeded)
        self.assertTrue(results["c2"].succeeded)

    def test_condition_true_runs_next(self):
        plan = (HybridPlan()
                .add_classical("c1", lambda: 1)
                .add_condition("cond1",
                                lambda r: r["c1"].value == 1)
                .add_classical("c2", lambda: 2))
        orch = HybridOrchestrator()
        results = orch.execute(plan)
        self.assertTrue(results["cond1"].succeeded)
        self.assertTrue(results["cond1"].value)
        self.assertTrue(results["c2"].succeeded)

    def test_condition_false_skips_next(self):
        plan = (HybridPlan()
                .add_classical("c1", lambda: 1)
                .add_condition("cond1",
                                lambda r: r["c1"].value == 99)
                .add_classical("c2", lambda: 2))
        orch = HybridOrchestrator()
        results = orch.execute(plan)
        self.assertTrue(results["cond1"].succeeded)
        self.assertFalse(results["cond1"].value)
        self.assertFalse(results["c2"].succeeded)
        self.assertIn("skipped", results["c2"].error)

    def test_skip_only_applies_to_next_non_barrier(self):
        plan = (HybridPlan()
                .add_classical("c1", lambda: 1)
                .add_condition("cond1", lambda r: False)
                .add_barrier("b")
                .add_classical("c2", lambda: 2))
        orch = HybridOrchestrator()
        results = orch.execute(plan)
        # Barrier runs even after a False condition; c2 should run too.
        self.assertTrue(results["b"].succeeded)
        self.assertTrue(results["c2"].succeeded)

    def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(dir_path=tmpdir)

            # First run: c1, c2.
            plan1 = (HybridPlan()
                     .add_classical("c1", lambda: 1)
                     .add_classical("c2", lambda: 2))
            orch1 = HybridOrchestrator(checkpoint_mgr=cm)
            orch1.execute(plan1, job_id="job_x")
            self.assertTrue(orch1.results["c1"].succeeded)
            self.assertTrue(orch1.results["c2"].succeeded)

            # Second run with a NEW plan that adds c3. We expect
            # c1, c2 results to be preserved from checkpoint,
            # and c3 to run.
            plan2 = (HybridPlan()
                     .add_classical("c1", lambda: 100)  # different value
                     .add_classical("c2", lambda: 200)
                     .add_classical("c3", lambda: 3))
            orch2 = HybridOrchestrator(checkpoint_mgr=cm)
            results = orch2.execute(plan2, job_id="job_x")
            # Resumed: c1/c2 values should match the checkpoint (1, 2).
            self.assertEqual(results["c1"].value, 1)
            self.assertEqual(results["c2"].value, 2)
            # c3 should be the fresh execution.
            self.assertEqual(results["c3"].value, 3)

    def test_events_list_recorded(self):
        plan = HybridPlan().add_classical("c1", lambda: 1)
        orch = HybridOrchestrator()
        orch.execute(plan)
        self.assertTrue(any("c1" in e for e in orch.events))


# ---------------------------------------------------------------------------
# Cross-backend synchronisation
# ---------------------------------------------------------------------------

class TestCrossBackendSyncPoint(unittest.TestCase):
    def test_arrive_and_wait_blocks_until_count_reached(self):
        sp = CrossBackendSyncPoint(name="x", expected_count=2)

        # First arrival should block; second should release both.
        first_done = threading.Event()
        second_done = threading.Event()

        def first():
            sp.arrive_and_wait()
            first_done.set()

        def second():
            sp.arrive_and_wait()
            second_done.set()

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        # Give first thread a chance to call arrive_and_wait.
        time.sleep(0.05)
        self.assertFalse(first_done.is_set())
        t2.start()
        t2.join(timeout=2.0)
        t1.join(timeout=2.0)
        self.assertTrue(first_done.is_set())
        self.assertTrue(second_done.is_set())

    def test_reset_clears_count(self):
        sp = CrossBackendSyncPoint(name="x", expected_count=2)
        sp.arrived = 1
        sp.reset()
        self.assertEqual(sp.arrived, 0)


class TestCrossBackendSynchronizer(unittest.TestCase):
    def test_register_and_get(self):
        s = CrossBackendSynchronizer()
        sp = s.register_sync_point("p1", expected_count=2)
        self.assertEqual(sp.expected_count, 2)
        self.assertEqual(s.get_sync_point("p1"), sp)

    def test_get_missing_returns_none(self):
        s = CrossBackendSynchronizer()
        self.assertIsNone(s.get_sync_point("nope"))


if __name__ == "__main__":
    unittest.main()
