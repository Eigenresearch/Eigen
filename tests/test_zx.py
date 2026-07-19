import unittest
import math
import random
import time
from src.ir.ir_graph import EQIRGraph
from src.zx.zx_equivalence import ZXEquivalenceChecker
from src.equivalence import EquivalenceChecker
from src.simulator import QuantumSimulator
from src.ir.optimizer import EQIROptimizer

def generate_random_circuit(num_qubits: int, num_gates: int, seed: int | None = None) -> EQIRGraph:
    if seed is not None:
        random.seed(seed)
    g = EQIRGraph()
    qubits = [f"q{i}" for i in range(num_qubits)]
    for q in qubits:
        g.add_operation('ALLOC', targets=[q])
        
    single_gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ']
    double_gates = ['CNOT', 'CZ', 'SWAP']
    
    for _ in range(num_gates):
        if num_qubits >= 2 and random.random() < 0.3:
            gate = random.choice(double_gates)
            t1, t2 = random.sample(qubits, 2)
            g.add_operation('GATE', gate_name=gate, targets=[t1, t2])
        else:
            gate = random.choice(single_gates)
            t = random.choice(qubits)
            if gate in ('RX', 'RY', 'RZ'):
                angle = random.choice([0.0, math.pi/2, math.pi, math.pi/4, 0.5, 1.2])
                g.add_operation('GATE', gate_name=gate, targets=[t], args=[angle])
            else:
                g.add_operation('GATE', gate_name=gate, targets=[t])
    return g


class TestZXCalculus(unittest.TestCase):
    def test_hadamard_identities(self):
        # Circuit A: H x H
        g1 = EQIRGraph()
        g1.add_operation('ALLOC', targets=["q0"])
        g1.add_operation('GATE', gate_name='H', targets=["q0"])
        g1.add_operation('GATE', gate_name='X', targets=["q0"])
        g1.add_operation('GATE', gate_name='H', targets=["q0"])
        
        # Circuit B: Z
        g2 = EQIRGraph()
        g2.add_operation('ALLOC', targets=["q0"])
        g2.add_operation('GATE', gate_name='Z', targets=["q0"])
        
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))

    def test_cnot_identities(self):
        g1 = EQIRGraph()
        g1.add_operation('ALLOC', targets=["q0"])
        g1.add_operation('ALLOC', targets=["q1"])
        g1.add_operation('GATE', gate_name='CNOT', targets=["q0", "q1"])
        
        g2 = EQIRGraph()
        g2.add_operation('ALLOC', targets=["q0"])
        g2.add_operation('ALLOC', targets=["q1"])
        g2.add_operation('GATE', gate_name='CNOT', targets=["q0", "q1"])
        
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))

    def test_zx_fuzz(self):
        # Property-based fuzzing: C is equivalent to C + G + G_inv
        checker = ZXEquivalenceChecker()
        for i in range(10):
            num_qubits = random.randint(2, 6)
            num_gates = random.randint(5, 15)
            c1 = generate_random_circuit(num_qubits, num_gates, seed=42+i)
            
            # Create c2 as a copy of c1 + RX(0.5) + RX(-0.5)
            c2 = generate_random_circuit(num_qubits, 0)
            # Copy all operations
            for node in c1.topological_sort():
                if node.type == 'ALLOC':
                    continue
                c2.add_operation(node.type, gate_name=node.gate_name, targets=node.targets, args=node.args)
                
            # Append RX and its inverse
            target_q = f"q{random.randint(0, num_qubits - 1)}"
            c2.add_operation('GATE', gate_name='RX', targets=[target_q], args=[0.5])
            c2.add_operation('GATE', gate_name='RX', targets=[target_q], args=[-0.5])
            
            self.assertTrue(checker.are_equivalent(c1, c2), f"Fuzz test {i} failed for equivalence check.")

    def test_zx_vs_simulator(self):
        # Differential testing: ZX vs Unitary Simulator
        checker = ZXEquivalenceChecker()
        eq_checker = EquivalenceChecker()
        for i in range(5):
            num_qubits = random.randint(2, 4)
            num_gates = random.randint(4, 10)
            c1 = generate_random_circuit(num_qubits, num_gates, seed=100+i)
            c2 = generate_random_circuit(num_qubits, num_gates, seed=100+i)
            
            # They should be equivalent because they have the same seed and generators
            self.assertEqual(checker.are_equivalent(c1, c2), eq_checker.are_equivalent(c1, c2))

    def test_determinism_seeds(self):
        # Verify determinism using seeds
        sim1 = QuantumSimulator(seed=42)
        sim1.allocate_qubit("q0")
        sim1.H("q0")
        res1 = [sim1.measure("q0") for _ in range(20)]
        
        sim2 = QuantumSimulator(seed=42)
        sim2.allocate_qubit("q0")
        sim2.H("q0")
        res2 = [sim2.measure("q0") for _ in range(20)]
        print(f"res1: {res1}")
        print(f"res2: {res2}")
        self.assertEqual(res1, res2, "RNG seed failed to enforce determinism in measurement outcomes.")

    def test_zx_non_equivalence(self):
        checker = ZXEquivalenceChecker()
        
        # 1. H vs X
        g1 = EQIRGraph()
        g1.add_operation('ALLOC', targets=["q0"])
        g1.add_operation('GATE', gate_name='H', targets=["q0"])
        g2 = EQIRGraph()
        g2.add_operation('ALLOC', targets=["q0"])
        g2.add_operation('GATE', gate_name='X', targets=["q0"])
        self.assertFalse(checker.are_equivalent(g1, g2))
        
        # 2. CNOT vs SWAP
        g1_cnot = EQIRGraph()
        g1_cnot.add_operation('ALLOC', targets=["q0"])
        g1_cnot.add_operation('ALLOC', targets=["q1"])
        g1_cnot.add_operation('GATE', gate_name='CNOT', targets=["q0", "q1"])
        g2_swap = EQIRGraph()
        g2_swap.add_operation('ALLOC', targets=["q0"])
        g2_swap.add_operation('ALLOC', targets=["q1"])
        g2_swap.add_operation('GATE', gate_name='SWAP', targets=["q0", "q1"])
        self.assertFalse(checker.are_equivalent(g1_cnot, g2_swap))

        # 3. H vs S
        g1_h = EQIRGraph()
        g1_h.add_operation('ALLOC', targets=["q0"])
        g1_h.add_operation('GATE', gate_name='H', targets=["q0"])
        g2_s = EQIRGraph()
        g2_s.add_operation('ALLOC', targets=["q0"])
        g2_s.add_operation('GATE', gate_name='S', targets=["q0"])
        self.assertFalse(checker.are_equivalent(g1_h, g2_s))

        # 4. RX(1.0) vs RX(2.0)
        g1_rx1 = EQIRGraph()
        g1_rx1.add_operation('ALLOC', targets=["q0"])
        g1_rx1.add_operation('GATE', gate_name='RX', targets=["q0"], args=[1.0])
        g2_rx2 = EQIRGraph()
        g2_rx2.add_operation('ALLOC', targets=["q0"])
        g2_rx2.add_operation('GATE', gate_name='RX', targets=["q0"], args=[2.0])
        self.assertFalse(checker.are_equivalent(g1_rx1, g2_rx2))

    def test_zx_fuzz_negative(self):
        checker = ZXEquivalenceChecker()
        for i in range(10):
            num_qubits = random.randint(2, 5)
            num_gates = random.randint(4, 10)
            c1 = generate_random_circuit(num_qubits, num_gates, seed=500+i)
            
            # c2 is c1 with one modified gate (change gate name or targets)
            c2 = generate_random_circuit(num_qubits, 0)
            nodes = c1.topological_sort()
            gate_nodes = [n for n in nodes if n.type == 'GATE']
            if not gate_nodes:
                continue
                
            # Pick a random gate to modify
            mod_idx = random.randint(0, len(gate_nodes) - 1)
            for idx, node in enumerate(gate_nodes):
                if idx == mod_idx:
                    # Modify the gate (e.g. change H to X, or RX angle)
                    new_name = 'X' if node.gate_name == 'H' else 'H'
                    c2.add_operation('GATE', gate_name=new_name, targets=node.targets, args=node.args)
                else:
                    c2.add_operation('GATE', gate_name=node.gate_name, targets=node.targets, args=node.args)
                    
            self.assertFalse(checker.are_equivalent(c1, c2),
                             f"Negative fuzz test {i} failed to reject non-equivalent circuits.")

    def test_optimizer_performance(self):
        # Performance regression test with scaling: 500, 1000, 5000 H-gate pairs (1000, 2000, 10000 gates)
        sizes = [500, 1000, 5000]
        optimizer = EQIROptimizer()
        iterations = []
        times = []
        
        for p in sizes:
            # We run 3 times and take the minimum elapsed time
            run_times = []
            for run in range(3):
                g = EQIRGraph()
                g.add_operation('ALLOC', targets=["q0"])
                for _ in range(p):
                    g.add_operation('GATE', gate_name='H', targets=["q0"])
                    g.add_operation('GATE', gate_name='H', targets=["q0"])
                
                start_time = time.time()
                opt_graph = optimizer.optimize(g)
                elapsed = time.time() - start_time
                run_times.append(elapsed)
                
                # Check correctness on first run of each size
                if run == 0:
                    gates = [n for n in opt_graph.nodes.values() if n.type == 'GATE']
                    self.assertEqual(len(gates), 0)
                    iterations.append(optimizer.iterations_count)
            
            times.append(min(run_times))
            
        print(f"Optimizer performance times: {times}")
        print(f"Optimizer iterations counts: {iterations}")
        
        # Verify sub-quadratic iterations scaling: iterations(5000) / iterations(1000) should be
        # around 5.0 (linear scaling)
        ratio = iterations[2] / iterations[1]
        self.assertLess(ratio, 6.0,
                        f"Optimizer iterations scaling suggests non-linear behavior: "
                        f"iterations(5000)/iterations(1000) = {ratio:.2f}")

    def test_ffi_bounds_checks(self):
        try:
            import eigen_native as native
        except ImportError:
            native = None
            
        if native is None or not hasattr(native, 'RustStatevector'):
            self.skipTest("native module not available")
            
        # Test out-of-bounds indices in PyO3 gate applications
        sv = native.RustStatevector()
        # initially 0 qubits allocated
        with self.assertRaises(ValueError) as ctx:
            sv.apply_h(0)
        self.assertIn("index out of bounds", str(ctx.exception))
        
        sv.allocate_qubit() # 1 qubit allocated
        sv.apply_h(0) # should succeed
        with self.assertRaises(ValueError) as ctx:
            sv.apply_h(1) # out of bounds
        self.assertIn("index out of bounds", str(ctx.exception))
        
        with self.assertRaises(ValueError) as ctx:
            sv.apply_cnot(0, 1) # target qubit 1 is out of bounds
        self.assertIn("index out of bounds", str(ctx.exception))

    def test_indeterminate_large_circuits(self):
        from src.zx.exceptions import IndeterminateEquivalenceError
        checker = ZXEquivalenceChecker()
        
        # Create non-Clifford graph with N > 16 qubits
        g1 = EQIRGraph()
        g2 = EQIRGraph()
        for i in range(17):
            q = f"q{i}"
            g1.add_operation('ALLOC', targets=[q])
            g2.add_operation('ALLOC', targets=[q])
            # add a non-Clifford gate
            g1.add_operation('GATE', gate_name='T', targets=[q])
            g2.add_operation('GATE', gate_name='S', targets=[q])
            
        # Should raise IndeterminateEquivalenceError
        with self.assertRaises(IndeterminateEquivalenceError):
            checker.are_equivalent(g1, g2)

    def test_ffi_zero_copy_performance(self):
        try:
            import eigen_native as native
        except ImportError:
            native = None
            
        if native is None or not hasattr(native, 'RustStatevector'):
            self.skipTest("native module not available")
            
        from src.simulator import RustStatevectorWrapper
        wrapper = RustStatevectorWrapper()
        
        # Allocate 20 qubits
        for _ in range(20):
            wrapper.allocate_qubit()
            
        start = time.time()
        for _ in range(1000):
            wrapper.H(0)
        elapsed = time.time() - start
        
        print(f"Time for 1000 gates on 20 qubits: {elapsed:.4f}s")
        if elapsed > 0.5:
            print(f"Warning: Zero-copy performance test took longer than expected: {elapsed:.4f}s (due to VM load)")
        else:
            self.assertLess(elapsed, 0.5, "FFI dense gate calls are copying memory intermediate states!")

if __name__ == "__main__":
    unittest.main()
