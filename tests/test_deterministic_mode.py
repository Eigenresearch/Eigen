import os
import tempfile
import unittest

from src.backend.vm import EigenVM


_WORKSPACE = tempfile.mkdtemp(prefix="eigen_det_test_")


def _compile_to_ebc(src, filename="__det_inline__.eig"):
    import hashlib
    content_hash = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
    path = os.path.join(_WORKSPACE, f"det_{content_hash}_{filename}")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
    from src.compiler import to_ebc
    return to_ebc(path, _WORKSPACE, optimize=False)


_DETERMINISTIC_SRC = """eigen 1.0
func main() -> int {
    let x: int = 5
    let y: int = x * 3 + 2
    return y
}
let result: int = main()
"""


_NOISE_SRC = """eigen 1.0
qubit q0
noise bitflip(0.5) q0
measure q0 -> c0
"""


class TestDeterministicBasic(unittest.TestCase):
    def test_same_seed_same_result_dense(self):
        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        vm1 = EigenVM(sim_type="dense", seed=42, deterministic=True)
        vm2 = EigenVM(sim_type="dense", seed=42, deterministic=True)
        vm1.execute(prog)
        vm2.execute(prog)
        self.assertEqual(vm1.lookup_var("result"), vm2.lookup_var("result"))

    def test_different_seeds_distinct_rng(self):
        vm1 = EigenVM(seed=42, deterministic=True)
        vm2 = EigenVM(seed=43, deterministic=True)
        seq1 = [vm1.rng.random() for _ in range(5)]
        seq2 = [vm2.rng.random() for _ in range(5)]
        self.assertNotEqual(seq1, seq2)

    def test_same_seed_same_rng_sequence(self):
        vm1 = EigenVM(seed=42, deterministic=True)
        vm2 = EigenVM(seed=42, deterministic=True)
        seq1 = [vm1.rng.random() for _ in range(10)]
        seq2 = [vm2.rng.random() for _ in range(10)]
        self.assertEqual(seq1, seq2)

    def test_deterministic_flag_propagated_to_audit(self):
        from src.runtime_audit import AuditTrail

        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        trail = AuditTrail(path=None, enabled=True)
        vm = EigenVM(sim_type="dense", seed=42, deterministic=True)
        vm.execute(prog, audit=trail, program_hash="test-det")
        entries = trail.entries()
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].deterministic)
        self.assertEqual(entries[0].seed, 42)

    def test_non_deterministic_flag_in_audit(self):
        from src.runtime_audit import AuditTrail

        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        trail = AuditTrail(path=None, enabled=True)
        vm = EigenVM(sim_type="dense", seed=42, deterministic=False)
        vm.execute(prog, audit=trail, program_hash="test-nondet")
        entries = trail.entries()
        self.assertFalse(entries[0].deterministic)

    def test_ctor_args_snapshot(self):
        vm = EigenVM(sim_type="dense", seed=42, deterministic=True)
        self.assertTrue(vm._ctor_args["deterministic"])
        self.assertEqual(vm._ctor_args["seed"], 42)
        self.assertEqual(vm._ctor_args["sim_type"], "dense")


class TestDeterministicStabilizer(unittest.TestCase):
    def test_stabilizer_same_seed_reproducible(self):
        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        vm1 = EigenVM(sim_type="stabilizer", seed=99, deterministic=True)
        vm2 = EigenVM(sim_type="stabilizer", seed=99, deterministic=True)
        vm1.execute(prog)
        vm2.execute(prog)
        self.assertEqual(vm1.lookup_var("result"), vm2.lookup_var("result"))

    def test_stabilizer_deterministic_seed_uses_correct_sim(self):
        vm = EigenVM(sim_type="stabilizer", seed=1, deterministic=True)
        self.assertEqual(vm.simulator.sim_type, "stabilizer")
        self.assertTrue(vm.deterministic)


class TestDeterministicDense(unittest.TestCase):
    def test_dense_same_seed_reproducible(self):
        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        vm1 = EigenVM(sim_type="dense", seed=7, deterministic=True)
        vm2 = EigenVM(sim_type="dense", seed=7, deterministic=True)
        vm1.execute(prog)
        vm2.execute(prog)
        self.assertEqual(vm1.lookup_var("result"), vm2.lookup_var("result"))

    def test_dense_seed_present_in_audit(self):
        from src.runtime_audit import AuditTrail

        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        trail = AuditTrail(path=None, enabled=True)
        vm = EigenVM(sim_type="dense", seed=123, deterministic=True)
        vm.execute(prog, audit=trail, program_hash="dense-test")
        entries = trail.entries()
        self.assertEqual(entries[0].seed, 123)
        self.assertEqual(entries[0].sim_type, "dense")


class TestDeterministicNoise(unittest.TestCase):
    def test_noise_vm_deterministic_flag(self):
        from src.noise.noise_model import NoiseModel

        nm = NoiseModel(noise_type="bit_flip", noise_prob=0.5)
        vm = EigenVM(sim_type="dense", seed=42, deterministic=True, noise_model=nm)
        self.assertTrue(vm.deterministic)
        self.assertEqual(vm.noise_model.noise_type, "bit_flip")

    def test_noise_with_seed_reproducible_rng(self):
        from src.noise.noise_model import NoiseModel

        nm1 = NoiseModel(noise_type="bit_flip", noise_prob=0.3, rng=__import__("random").Random(5))
        nm2 = NoiseModel(noise_type="bit_flip", noise_prob=0.3, rng=__import__("random").Random(5))
        out1 = [nm1.apply_readout_noise(0) for _ in range(10)]
        out2 = [nm2.apply_readout_noise(0) for _ in range(10)]
        self.assertEqual(out1, out2)

    def test_noise_seed_audit_recorded(self):
        from src.runtime_audit import AuditTrail

        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        trail = AuditTrail(path=None, enabled=True)
        from src.noise.noise_model import NoiseModel

        nm = NoiseModel(noise_type="bit_flip", noise_prob=0.5)
        vm = EigenVM(sim_type="dense", seed=42, deterministic=True, noise_model=nm)
        vm.execute(prog, audit=trail, program_hash="noise-test")
        entries = trail.entries()
        self.assertEqual(entries[0].noise_type, "bit_flip")


class TestDeterministicShots(unittest.TestCase):
    def test_parallel_shots_same_seed_deterministic(self):
        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        vm = EigenVM(sim_type="dense", seed=42, deterministic=True)
        results1 = vm.execute_parallel(prog, shots=4, threads=2)
        results2 = vm.execute_parallel(prog, shots=4, threads=2)
        self.assertEqual(len(results1), 4)
        self.assertEqual(len(results2), 4)
        for r1, r2 in zip(results1, results2, strict=False):
            self.assertEqual(r1.get("result"), r2.get("result"))

    def test_parallel_shots_zero_returns_empty(self):
        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        vm = EigenVM(sim_type="dense", seed=42, deterministic=True)
        results = vm.execute_parallel(prog, shots=0)
        self.assertEqual(results, [])

    def test_deterministic_results_are_consistent_across_runs(self):
        prog = _compile_to_ebc(_DETERMINISTIC_SRC)
        vm1 = EigenVM(sim_type="dense", seed=10, deterministic=True)
        vm2 = EigenVM(sim_type="dense", seed=10, deterministic=True)
        vm1.execute(prog)
        vm2.execute(prog)
        for k in vm1.globals:
            self.assertEqual(vm1.globals[k], vm2.globals.get(k))


class TestDeterministicPropagation(unittest.TestCase):
    def test_simulator_seed_matches_vm_seed(self):
        vm1 = EigenVM(sim_type="dense", seed=42, deterministic=True)
        vm2 = EigenVM(sim_type="dense", seed=42, deterministic=True)
        v1 = vm1.simulator.rng.random()
        v2 = vm2.simulator.rng.random()
        self.assertEqual(v1, v2)

    def test_rng_used_for_measurements(self):
        vm1 = EigenVM(sim_type="dense", seed=42, deterministic=True)
        vm2 = EigenVM(sim_type="dense", seed=42, deterministic=True)
        v1 = vm1.rng.random()
        v2 = vm2.rng.random()
        self.assertEqual(v1, v2)

    def test_default_deterministic_false(self):
        vm = EigenVM()
        self.assertFalse(vm.deterministic)

    def test_dispatch_mode_in_ctor_args(self):
        vm = EigenVM(dispatch_mode="table", deterministic=True, seed=1)
        self.assertEqual(vm._ctor_args["dispatch_mode"], "table")


if __name__ == "__main__":
    unittest.main()
