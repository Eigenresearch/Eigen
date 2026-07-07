"""§12.3 part 2 — Seed management for reproducibility tests."""
import threading
import unittest

from src.reproducibility.seed_management import (
    SeedManager,
    GlobalSeedManager,
    set_global_seed,
    get_global_seed,
    set_component_seed,
    clear_component_seed,
    get_seed_for,
    reset_global_seed,
    get_global_registry,
)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestSeedManagerConstruction(unittest.TestCase):
    def test_basic_construction(self):
        sm = SeedManager(master_seed=42)
        self.assertEqual(sm.master_seed, 42)

    def test_zero_master_seed_allowed(self):
        sm = SeedManager(master_seed=0)
        self.assertEqual(sm.master_seed, 0)

    def test_large_master_seed_allowed(self):
        sm = SeedManager(master_seed=2 ** 63)
        self.assertEqual(sm.master_seed, 2 ** 63)

    def test_negative_master_raises_value_error(self):
        with self.assertRaises(ValueError):
            SeedManager(master_seed=-1)

    def test_non_int_master_raises_type_error(self):
        with self.assertRaises(TypeError):
            SeedManager(master_seed="42")

    def test_bool_master_raises_type_error(self):
        with self.assertRaises(TypeError):
            SeedManager(master_seed=True)

    def test_float_master_raises_type_error(self):
        with self.assertRaises(TypeError):
            SeedManager(master_seed=42.5)


# ---------------------------------------------------------------------------
# Seed derivation — determinism + uniqueness
# ---------------------------------------------------------------------------

class TestSeedManagerDerivation(unittest.TestCase):
    def test_same_master_same_component_yields_same_seed(self):
        sm = SeedManager(master_seed=100)
        self.assertEqual(sm.seed_for("alpha"), sm.seed_for("alpha"))

    def test_distinct_components_give_distinct_seeds(self):
        sm = SeedManager(master_seed=100)
        names = ("simulator", "noise", "scheduler", "readout",
                    "audit", "compiler", "aot")
        seeds = {n: sm.seed_for(n) for n in names}
        self.assertEqual(len(seeds.values()), len(set(seeds.values())))

    def test_distinct_masters_give_distinct_seeds_for_same_component(self):
        sm_a = SeedManager(master_seed=1)
        sm_b = SeedManager(master_seed=2)
        self.assertNotEqual(sm_a.seed_for("sim"),
                              sm_b.seed_for("sim"))

    def test_seed_is_in_64bit_range(self):
        sm = SeedManager(master_seed=42)
        for name in ("a", "b", "c"):
            seed = sm.seed_for(name)
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2 ** 64)

    def test_non_string_component_name_raises_type_error(self):
        sm = SeedManager(master_seed=0)
        with self.assertRaises(TypeError):
            sm.seed_for(42)
        with self.assertRaises(TypeError):
            sm.seed_for(None)

    def test_empty_component_name_returns_seed(self):
        sm = SeedManager(master_seed=0)
        # Empty string is allowed; the hash just hashes (master, "")
        self.assertIsInstance(sm.seed_for(""), int)

    def test_unicode_component_name_returns_seed(self):
        sm = SeedManager(master_seed=0)
        self.assertIsInstance(sm.seed_for("noise_α"), int)


# ---------------------------------------------------------------------------
# child_seeds helper
# ---------------------------------------------------------------------------

class TestSeedManagerChildSeeds(unittest.TestCase):
    def test_returns_dict(self):
        sm = SeedManager(master_seed=3)
        out = sm.child_seeds("a", "b", "c")
        self.assertIsInstance(out, dict)
        self.assertEqual(sorted(out.keys()), ["a", "b", "c"])
        self.assertNotEqual(out["a"], out["b"])
        self.assertNotEqual(out["a"], out["c"])
        self.assertNotEqual(out["b"], out["c"])

    def test_empty_call_returns_empty_dict(self):
        self.assertEqual(SeedManager(master_seed=0).child_seeds(), {})

    def test_consistent_with_individual_calls(self):
        sm = SeedManager(master_seed=42)
        bulk = sm.child_seeds("a", "b")
        self.assertEqual(bulk["a"], sm.seed_for("a"))
        self.assertEqual(bulk["b"], sm.seed_for("b"))


# ---------------------------------------------------------------------------
# Submanager derivation
# ---------------------------------------------------------------------------

class TestSeedManagerSubmanager(unittest.TestCase):
    def test_submanager_master_is_parent_child_seed(self):
        parent = SeedManager(master_seed=7)
        sub = parent.derive_submanager("child")
        self.assertEqual(sub.master_seed, parent.seed_for("child"))

    def test_two_submanagers_for_same_name_are_equal_in_master(self):
        parent = SeedManager(master_seed=7)
        s1 = parent.derive_submanager("a")
        s2 = parent.derive_submanager("a")
        self.assertEqual(s1.master_seed, s2.master_seed)

    def test_submanager_seed_derivation_distinct_from_parent(self):
        parent = SeedManager(master_seed=7)
        sub = parent.derive_submanager("noise")
        # Sub's seed for any component name is generally not equal
        # to the parent's seed for the same component name (because
        # the underlying masters differ).
        self.assertNotEqual(sub.seed_for("sim"),
                              parent.seed_for("sim"))

    def test_submanager_distinct_from_other_submanagers(self):
        parent = SeedManager(master_seed=7)
        s1 = parent.derive_submanager("noise")
        s2 = parent.derive_submanager("scheduler")
        self.assertNotEqual(s1.master_seed, s2.master_seed)


# ---------------------------------------------------------------------------
# Thread-safety — module-level singleton
# ---------------------------------------------------------------------------

class TestSeedManagerThreadSafe(unittest.TestCase):
    def test_concurrent_set_get_seed(self):
        # Many threads alternate setting and getting; we just check
        # no exceptions and final seed is one we've seen.
        sm = SeedManager(master_seed=42)
        results = []
        errors = []

        def worker():
            try:
                s = sm.seed_for("thread-component")
                results.append(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        # All seeds must be identical (deterministic + stateless)
        self.assertEqual(len(set(results)), 1)


# ---------------------------------------------------------------------------
# GlobalSeedManager instance-level tests
# ---------------------------------------------------------------------------

class TestGlobalSeedManagerInstance(unittest.TestCase):
    def setUp(self):
        # Fresh instance for each test — bypasses module singleton.
        self.gs = GlobalSeedManager()

    def test_initial_state_uninitialized(self):
        self.assertFalse(self.gs.is_initialized())
        self.assertIsNone(self.gs.get_global_seed())

    def test_get_seed_for_raises_if_no_master(self):
        with self.assertRaises(RuntimeError):
            self.gs.get_seed_for("sim")

    def test_set_global_seed_initializes_master(self):
        self.gs.set_global_seed(99)
        self.assertEqual(self.gs.get_global_seed(), 99)
        self.assertTrue(self.gs.is_initialized())

    def test_set_global_seed_clears_overrides(self):
        self.gs.set_component_seed("sim", 5)
        self.gs.set_global_seed(99)
        # After reset, the override is gone — get_seed_for returns the
        # derivation from the master, NOT the override-5.
        self.assertNotEqual(self.gs.get_seed_for("sim"), 5)

    def test_set_component_seed_works_without_master(self):
        self.gs.set_component_seed("sim", 5)
        self.assertEqual(self.gs.get_seed_for("sim"), 5)
        self.assertTrue(self.gs.is_initialized())

    def test_set_component_seed_overrides_global_derivation(self):
        self.gs.set_global_seed(99)
        derived = self.gs.get_seed_for("sim")
        self.gs.set_component_seed("sim", 12345)
        self.assertEqual(self.gs.get_seed_for("sim"), 12345)
        # Different components still derived from master
        self.assertEqual(self.gs.get_seed_for("noise"),
                            SeedManager(99).seed_for("noise"))

    def test_clear_component_seed_removes_override(self):
        # The contract is: set_global_seed also WIPES overrides (so a
        # reset starts fresh). So set_global_seed first, then add the
        # override, then verify clearing puts us back to derivation.
        self.gs.set_global_seed(99)
        self.gs.set_component_seed("sim", 5)
        self.assertEqual(self.gs.get_seed_for("sim"), 5)
        self.gs.clear_component_seed("sim")
        # After clearing, fall back to global derivation
        self.assertEqual(self.gs.get_seed_for("sim"),
                            SeedManager(99).seed_for("sim"))

    def test_reset_clears_everything(self):
        self.gs.set_global_seed(99)
        self.gs.set_component_seed("sim", 5)
        self.gs.reset()
        self.assertIsNone(self.gs.get_global_seed())
        with self.assertRaises(RuntimeError):
            self.gs.get_seed_for("sim")
        self.assertFalse(self.gs.is_initialized())

    def test_set_global_seed_with_negative_raises(self):
        with self.assertRaises(ValueError):
            self.gs.set_global_seed(-1)

    def test_set_global_seed_with_non_int_raises(self):
        with self.assertRaises(TypeError):
            self.gs.set_global_seed("42")

    def test_set_global_seed_with_bool_raises(self):
        with self.assertRaises(TypeError):
            self.gs.set_global_seed(True)

    def test_set_component_seed_with_bool_raises(self):
        with self.assertRaises(TypeError):
            self.gs.set_component_seed("sim", True)


# ---------------------------------------------------------------------------
# Module-level shims (singleton access)
# ---------------------------------------------------------------------------

class TestModuleLevelSeedAccessors(unittest.TestCase):
    def setUp(self):
        # Reset before each test so the singleton is in a fresh state.
        reset_global_seed()

    def tearDown(self):
        reset_global_seed()

    def test_get_global_seed_initially_none(self):
        self.assertIsNone(get_global_seed())

    def test_get_seed_for_raises_when_uninitialized(self):
        with self.assertRaises(RuntimeError):
            get_seed_for("sim")

    def test_set_get_global_seed_round_trip(self):
        set_global_seed(12345)
        self.assertEqual(get_global_seed(), 12345)

    def test_get_seed_for_uses_global_master(self):
        set_global_seed(7)
        expected = SeedManager(7).seed_for("sim")
        self.assertEqual(get_seed_for("sim"), expected)

    def test_get_seed_for_distinct_components_distinct(self):
        set_global_seed(7)
        s_sim = get_seed_for("sim")
        s_noise = get_seed_for("noise")
        self.assertNotEqual(s_sim, s_noise)

    def test_set_component_seed_overrides_globally_visible(self):
        set_global_seed(7)
        set_component_seed("sim", 999)
        self.assertEqual(get_seed_for("sim"), 999)

    def test_clear_component_seed_globally_visible(self):
        set_global_seed(7)
        set_component_seed("sim", 999)
        clear_component_seed("sim")
        self.assertEqual(get_seed_for("sim"),
                            SeedManager(7).seed_for("sim"))

    def test_reset_global_seed_clears_master(self):
        set_global_seed(7)
        reset_global_seed()
        self.assertIsNone(get_global_seed())
        with self.assertRaises(RuntimeError):
            get_seed_for("sim")

    def test_reset_global_seed_clears_overrides(self):
        set_global_seed(7)
        set_component_seed("sim", 5)
        reset_global_seed()
        # After reset, the override is gone but so is the master,
        # so get_seed_for will raise.
        with self.assertRaises(RuntimeError):
            get_seed_for("sim")

    def test_get_global_registry_returns_singleton(self):
        r1 = get_global_registry()
        r2 = get_global_registry()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# Cross-instance independence
# ---------------------------------------------------------------------------

class TestSeedManagerInstanceIndependence(unittest.TestCase):
    def test_two_instances_with_same_master_produce_same_seeds(self):
        a = SeedManager(master_seed=123)
        b = SeedManager(master_seed=123)
        self.assertEqual(a.seed_for("foo"), b.seed_for("foo"))

    def test_two_global_managers_are_independent(self):
        # Each GlobalSeedManager is its own object with its own state.
        a = GlobalSeedManager()
        b = GlobalSeedManager()
        a.set_global_seed(1)
        b.set_global_seed(2)
        self.assertEqual(a.get_global_seed(), 1)
        self.assertEqual(b.get_global_seed(), 2)


if __name__ == "__main__":
    unittest.main()
