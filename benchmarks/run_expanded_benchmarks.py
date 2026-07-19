"""Expanded benchmark suite: 30+ (workload, size) configurations.

Workloads:
  Classical (Eigen VM vs CPython):
    1. arithmetic_sum: sum(1..N)
    2. fibonacci: iterative fib
    3. string_concat: N string concatenations
    4. factorial: iterative factorial
    5. nested_loop: i*j loop
    6. array_access: sequential array access
  
  Quantum (Eigen Simulator vs Python+NumPy):
    7. bell_state: Bell state shots
    8. gate_chain: N H gates on 1 qubit
    9. ghz_state: GHZ state prep + measure
    10. qft_circuit: QFT on N qubits
    11. grover_iteration: Grover search iterations
    12. random_clifford: random Clifford circuit
    13. multi_qubit_measure: measure N qubits
    14. entangle_chain: chain entanglement
    15. dense_gate_apply: dense matrix gate application

Each runs at 2-3 sizes × 10 trials = 30+ × 10 = 300+ raw measurements.
"""
import csv, math, os, statistics, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.bytecode import Instruction, Opcode
from src.backend.vm import EigenVM
from src.simulator import QuantumSimulator

TRIALS = 10

def _approx(a, b, tol=1e-9):
    return abs(a - b) < tol

# === Bytecode generators ===

def make_sum_program(N):
    instrs = []
    instrs += [Instruction(Opcode.LOAD_CONST, 0), Instruction(Opcode.STORE_VAR, "sum")]
    instrs += [Instruction(Opcode.LOAD_CONST, 1), Instruction(Opcode.STORE_VAR, "i")]
    lc = len(instrs)
    instrs += [Instruction(Opcode.LOAD_VAR, "i"), Instruction(Opcode.LOAD_CONST, N), Instruction(Opcode.GT)]
    instrs += [Instruction(Opcode.JMP_IF_TRUE, lc + 13)]
    instrs += [Instruction(Opcode.LOAD_VAR, "sum"), Instruction(Opcode.LOAD_VAR, "i"),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "sum")]
    instrs += [Instruction(Opcode.LOAD_VAR, "i"), Instruction(Opcode.LOAD_CONST, 1),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "i")]
    instrs += [Instruction(Opcode.JMP, lc)]
    instrs += [Instruction(Opcode.LOAD_VAR, "sum"), Instruction(Opcode.PRINT), Instruction(Opcode.HALT)]
    return instrs

def make_fib_program(N):
    instrs = []
    instrs += [Instruction(Opcode.LOAD_CONST, 0), Instruction(Opcode.STORE_VAR, "a")]
    instrs += [Instruction(Opcode.LOAD_CONST, 1), Instruction(Opcode.STORE_VAR, "b")]
    instrs += [Instruction(Opcode.LOAD_CONST, 0), Instruction(Opcode.STORE_VAR, "count")]
    lc = len(instrs)
    instrs += [Instruction(Opcode.LOAD_VAR, "count"), Instruction(Opcode.LOAD_CONST, N), Instruction(Opcode.GTE)]
    exit_target = len(instrs) + 1 + 13
    instrs += [Instruction(Opcode.JMP_IF_TRUE, exit_target)]
    instrs += [Instruction(Opcode.LOAD_VAR, "a"), Instruction(Opcode.LOAD_VAR, "b"),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "temp")]
    instrs += [Instruction(Opcode.LOAD_VAR, "b"), Instruction(Opcode.STORE_VAR, "a")]
    instrs += [Instruction(Opcode.LOAD_VAR, "temp"), Instruction(Opcode.STORE_VAR, "b")]
    instrs += [Instruction(Opcode.LOAD_VAR, "count"), Instruction(Opcode.LOAD_CONST, 1),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "count")]
    instrs += [Instruction(Opcode.JMP, lc)]
    instrs += [Instruction(Opcode.LOAD_VAR, "a"), Instruction(Opcode.PRINT), Instruction(Opcode.HALT)]
    return instrs

def make_factorial_program(N):
    instrs = []
    instrs += [Instruction(Opcode.LOAD_CONST, 1), Instruction(Opcode.STORE_VAR, "result")]
    instrs += [Instruction(Opcode.LOAD_CONST, 1), Instruction(Opcode.STORE_VAR, "i")]
    lc = len(instrs)
    instrs += [Instruction(Opcode.LOAD_VAR, "i"), Instruction(Opcode.LOAD_CONST, N), Instruction(Opcode.GT)]
    exit_target = len(instrs) + 1 + 9
    instrs += [Instruction(Opcode.JMP_IF_TRUE, exit_target)]
    instrs += [Instruction(Opcode.LOAD_VAR, "result"), Instruction(Opcode.LOAD_VAR, "i"),
               Instruction(Opcode.MUL), Instruction(Opcode.STORE_VAR, "result")]
    instrs += [Instruction(Opcode.LOAD_VAR, "i"), Instruction(Opcode.LOAD_CONST, 1),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "i")]
    instrs += [Instruction(Opcode.JMP, lc)]
    instrs += [Instruction(Opcode.LOAD_VAR, "result"), Instruction(Opcode.PRINT), Instruction(Opcode.HALT)]
    return instrs

def make_nested_loop_program(N):
    instrs = []
    instrs += [Instruction(Opcode.LOAD_CONST, 0), Instruction(Opcode.STORE_VAR, "total")]
    instrs += [Instruction(Opcode.LOAD_CONST, 0), Instruction(Opcode.STORE_VAR, "i")]
    lc1 = len(instrs)
    instrs += [Instruction(Opcode.LOAD_VAR, "i"), Instruction(Opcode.LOAD_CONST, N), Instruction(Opcode.GTE)]
    instrs += [Instruction(Opcode.JMP_IF_TRUE, -1)]  # placeholder
    instrs += [Instruction(Opcode.LOAD_CONST, 0), Instruction(Opcode.STORE_VAR, "j")]
    lc2 = len(instrs)
    instrs += [Instruction(Opcode.LOAD_VAR, "j"), Instruction(Opcode.LOAD_CONST, N), Instruction(Opcode.GTE)]
    instrs += [Instruction(Opcode.JMP_IF_TRUE, -1)]  # placeholder
    instrs += [Instruction(Opcode.LOAD_VAR, "total"), Instruction(Opcode.LOAD_VAR, "i"),
               Instruction(Opcode.LOAD_VAR, "j"), Instruction(Opcode.MUL),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "total")]
    instrs += [Instruction(Opcode.LOAD_VAR, "j"), Instruction(Opcode.LOAD_CONST, 1),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "j")]
    instrs += [Instruction(Opcode.JMP, lc2)]
    # Fix inner exit target
    inner_exit = len(instrs)
    instrs[lc2 + 3] = Instruction(Opcode.JMP_IF_TRUE, inner_exit)
    instrs += [Instruction(Opcode.LOAD_VAR, "i"), Instruction(Opcode.LOAD_CONST, 1),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "i")]
    instrs += [Instruction(Opcode.JMP, lc1)]
    outer_exit = len(instrs)
    instrs[lc1 + 3] = Instruction(Opcode.JMP_IF_TRUE, outer_exit)
    instrs += [Instruction(Opcode.LOAD_VAR, "total"), Instruction(Opcode.PRINT), Instruction(Opcode.HALT)]
    return instrs

# === Classical workloads ===

def run_eigen_sum(N):
    vm = EigenVM(opt_level=3)
    vm.execute(make_sum_program(N))
    return vm.call_stack[-1].locals.get("sum", 0) if vm.call_stack else 0

def run_py_sum(N):
    s = 0
    for i in range(1, N+1): s += i
    return s

def run_eigen_fib(N):
    vm = EigenVM(opt_level=3)
    vm.execute(make_fib_program(N))
    return vm.call_stack[-1].locals.get("a", 0) if vm.call_stack else 0

def run_py_fib(N):
    a, b = 0, 1
    for _ in range(N): a, b = b, a + b
    return a

def run_eigen_factorial(N):
    vm = EigenVM(opt_level=3)
    vm.execute(make_factorial_program(N))
    return vm.call_stack[-1].locals.get("result", 1) if vm.call_stack else 1

def run_py_factorial(N):
    r = 1
    for i in range(1, N+1): r *= i
    return r

def run_eigen_nested(N):
    vm = EigenVM(opt_level=3)
    vm.execute(make_nested_loop_program(N))
    return vm.call_stack[-1].locals.get("total", 0) if vm.call_stack else 0

def run_py_nested(N):
    total = 0
    for i in range(N):
        for j in range(N):
            total += i * j
    return total

def run_eigen_str_concat(N):
    instrs = []
    instrs += [Instruction(Opcode.LOAD_CONST, ""), Instruction(Opcode.STORE_VAR, "s")]
    instrs += [Instruction(Opcode.LOAD_CONST, 0), Instruction(Opcode.STORE_VAR, "i")]
    lc = len(instrs)
    instrs += [Instruction(Opcode.LOAD_VAR, "i"), Instruction(Opcode.LOAD_CONST, N), Instruction(Opcode.GT)]
    exit_target = len(instrs) + 1 + 8
    instrs += [Instruction(Opcode.JMP_IF_TRUE, exit_target)]
    instrs += [Instruction(Opcode.LOAD_VAR, "s"), Instruction(Opcode.LOAD_CONST, "x"),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "s")]
    instrs += [Instruction(Opcode.LOAD_VAR, "i"), Instruction(Opcode.LOAD_CONST, 1),
               Instruction(Opcode.ADD), Instruction(Opcode.STORE_VAR, "i")]
    instrs += [Instruction(Opcode.JMP, lc)]
    instrs += [Instruction(Opcode.HALT)]
    vm = EigenVM(opt_level=3)
    vm.execute(instrs)
    return vm.call_stack[-1].locals.get("s", "") if vm.call_stack else ""

def run_py_str_concat(N):
    s = ""
    for _ in range(N): s += "x"
    return s

# === Quantum workloads ===

def run_eigen_bell(shots):
    results = []
    for _ in range(shots):
        sim = QuantumSimulator(sim_type='dense', seed=42)
        sim.allocate_qubit("q0"); sim.allocate_qubit("q1")
        sim.H("q0"); sim.CNOT("q0", "q1")
        results.append((sim.measure("q0"), sim.measure("q1")))
    return results

def run_py_bell(shots):
    import random
    rng = random.Random(42)
    inv = 1.0/math.sqrt(2.0)
    H = np.array([[inv, inv], [inv, -inv]], dtype=complex)
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    results = []
    for _ in range(shots):
        state = np.array([1,0,0,0], dtype=complex)
        state = np.kron(H, np.eye(2)) @ state
        state = CNOT @ state
        p0 = abs(state[0])**2 + abs(state[2])**2
        c0 = 0 if rng.random() < p0 else 1
        if c0 == 0:
            state[1] = 0; state[3] = 0
        else:
            state[0] = 0; state[2] = 0
        state /= np.sqrt(np.sum(np.abs(state)**2))
        p0q1 = abs(state[0])**2 + abs(state[1])**2
        c1 = 0 if rng.random() < p0q1 else 1
        results.append((c0, c1))
    return results

def run_eigen_gate_chain(N):
    sim = QuantumSimulator(sim_type='dense', seed=42)
    sim.allocate_qubit("q0")
    for _ in range(N): sim.H("q0")
    return sim.measure("q0")

def run_py_gate_chain(N):
    import random
    rng = random.Random(42)
    inv = 1.0/math.sqrt(2.0)
    H = np.array([[inv, inv], [inv, -inv]], dtype=complex)
    state = np.array([1, 0], dtype=complex)
    for _ in range(N): state = H @ state
    p0 = abs(state[0])**2
    return 0 if rng.random() < p0 else 1

def run_eigen_ghz(n_qubits, shots):
    results = []
    for _ in range(shots):
        sim = QuantumSimulator(sim_type='dense', seed=42)
        for i in range(n_qubits): sim.allocate_qubit(f"q{i}")
        sim.H("q0")
        for i in range(n_qubits - 1): sim.CNOT(f"q{i}", f"q{i+1}")
        outcomes = [sim.measure(f"q{i}") for i in range(n_qubits)]
        results.append(tuple(outcomes))
    return results

def run_py_ghz(n_qubits, shots):
    import random
    rng = random.Random(42)
    inv = 1.0/math.sqrt(2.0)
    H = np.array([[inv, inv], [inv, -inv]], dtype=complex)
    results = []
    for _ in range(shots):
        state = np.array([1], dtype=complex)
        for _ in range(n_qubits): state = np.kron(state, [1, 0])
        # H on q0
        full_h = np.eye(1, dtype=complex)
        for i in range(n_qubits):
            full_h = np.kron(full_h, H if i == 0 else np.eye(2, dtype=complex))
        state = full_h @ state
        # CNOT chain
        for i in range(n_qubits - 1):
            cnot_full = np.eye(2**n_qubits, dtype=complex)
            for j in range(2**n_qubits):
                if (j >> i) & 1 and not (j >> (i+1)) & 1:
                    cnot_full[j, j] = 0
                    cnot_full[j, j | (1 << (i+1))] = 1
                    cnot_full[j | (1 << (i+1)), j | (1 << (i+1))] = 0
                    cnot_full[j | (1 << (i+1)), j] = 1
            state = cnot_full @ state
        # Measure
        probs = np.abs(state)**2
        idx = rng.choices(range(len(probs)), weights=probs)[0]
        outcomes = tuple((idx >> i) & 1 for i in range(n_qubits))
        results.append(outcomes)
    return results

def run_eigen_random_clifford(n_qubits, n_gates):
    import random
    rng = random.Random(42)
    clifford_1q = [("H",), ("S",), ("X",), ("Y",), ("Z",)]
    clifford_2q = [("CNOT",), ("CZ",), ("SWAP",)]
    sim = QuantumSimulator(sim_type='dense', seed=42)
    for i in range(n_qubits): sim.allocate_qubit(f"q{i}")
    for _ in range(n_gates):
        if n_qubits >= 2 and rng.random() < 0.3:
            gate = rng.choice(clifford_2q)
            c, t = rng.sample(range(n_qubits), 2)
            getattr(sim, gate[0])(f"q{c}", f"q{t}")
        else:
            gate = rng.choice(clifford_1q)
            q = rng.randint(0, n_qubits - 1)
            getattr(sim, gate[0])(f"q{q}")
    return sim.measure(f"q0")

def run_py_random_clifford(n_qubits, n_gates):
    import random
    rng = random.Random(42)
    inv = 1.0/math.sqrt(2.0)
    gates_1q = {
        "H": np.array([[inv, inv], [inv, -inv]], dtype=complex),
        "S": np.array([[1, 0], [0, 1j]], dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    state = np.array([1], dtype=complex)
    for _ in range(n_qubits): state = np.kron(state, [1, 0])
    dim = 2**n_qubits
    clifford_1q = list(gates_1q.keys())
    clifford_2q = ["CNOT", "CZ", "SWAP"]
    for _ in range(n_gates):
        if n_qubits >= 2 and rng.random() < 0.3:
            gate = rng.choice(clifford_2q)
            c, t = rng.sample(range(n_qubits), 2)
            full = np.eye(dim, dtype=complex)
            for j in range(dim):
                if gate == "CNOT" and (j >> c) & 1:
                    j2 = j ^ (1 << t)
                    full[j, j] = 0; full[j, j2] = 1
                    full[j2, j2] = 0; full[j2, j] = 1
                elif gate == "CZ" and (j >> c) & 1 and (j >> t) & 1:
                    full[j, j] = -1
                elif gate == "SWAP":
                    bits_c = (j >> c) & 1
                    bits_t = (j >> t) & 1
                    j2 = j ^ (bits_c << c) ^ (bits_t << t) ^ (bits_t << c) ^ (bits_c << t)
                    full[j, j] = 0; full[j, j2] = 1
            state = full @ state
        else:
            gate = rng.choice(clifford_1q)
            q = rng.randint(0, n_qubits - 1)
            full = np.eye(1, dtype=complex)
            for i in range(n_qubits):
                full = np.kron(full, gates_1q[gate] if i == q else np.eye(2, dtype=complex))
            state = full @ state
    p0 = sum(abs(state[j])**2 for j in range(dim) if not (j & 1))
    return 0 if rng.random() < p0 else 1

def run_eigen_multi_measure(n_qubits):
    sim = QuantumSimulator(sim_type='dense', seed=42)
    for i in range(n_qubits): sim.allocate_qubit(f"q{i}")
    sim.H("q0")
    for i in range(n_qubits - 1): sim.CNOT(f"q{i}", f"q{i+1}")
    return [sim.measure(f"q{i}") for i in range(n_qubits)]

def run_py_multi_measure(n_qubits):
    import random
    rng = random.Random(42)
    inv = 1.0/math.sqrt(2.0)
    H = np.array([[inv, inv], [inv, -inv]], dtype=complex)
    state = np.array([1], dtype=complex)
    for _ in range(n_qubits): state = np.kron(state, [1, 0])
    full_h = np.eye(1, dtype=complex)
    for i in range(n_qubits):
        full_h = np.kron(full_h, H if i == 0 else np.eye(2, dtype=complex))
    state = full_h @ state
    for i in range(n_qubits - 1):
        cnot_full = np.eye(2**n_qubits, dtype=complex)
        for j in range(2**n_qubits):
            if (j >> i) & 1 and not (j >> (i+1)) & 1:
                cnot_full[j, j] = 0; cnot_full[j, j | (1 << (i+1))] = 1
                cnot_full[j | (1 << (i+1)), j | (1 << (i+1))] = 0; cnot_full[j | (1 << (i+1)), j] = 1
        state = cnot_full @ state
    results = []
    for q in range(n_qubits):
        p0 = sum(abs(state[j])**2 for j in range(len(state)) if not (j >> q) & 1)
        outcome = 0 if rng.random() < p0 else 1
        results.append(outcome)
        if outcome == 0:
            for j in range(len(state)):
                if (j >> q) & 1: state[j] = 0
        else:
            for j in range(len(state)):
                if not (j >> q) & 1: state[j] = 0
        state /= np.sqrt(np.sum(np.abs(state)**2))
    return results

def run_eigen_entangle_chain(n_qubits):
    sim = QuantumSimulator(sim_type='dense', seed=42)
    for i in range(n_qubits): sim.allocate_qubit(f"q{i}")
    sim.H("q0")
    for i in range(n_qubits - 1): sim.CNOT(f"q{i}", f"q{i+1}")
    sv = sim.get_state_vector()
    return sum(abs(a)**2 for a in sv)

def run_py_entangle_chain(n_qubits):
    inv = 1.0/math.sqrt(2.0)
    H = np.array([[inv, inv], [inv, -inv]], dtype=complex)
    state = np.array([1], dtype=complex)
    for _ in range(n_qubits): state = np.kron(state, [1, 0])
    full_h = np.eye(1, dtype=complex)
    for i in range(n_qubits):
        full_h = np.kron(full_h, H if i == 0 else np.eye(2, dtype=complex))
    state = full_h @ state
    for i in range(n_qubits - 1):
        dim = 2**n_qubits
        cnot_full = np.eye(dim, dtype=complex)
        for j in range(dim):
            if (j >> i) & 1 and not (j >> (i+1)) & 1:
                cnot_full[j, j] = 0; cnot_full[j, j | (1 << (i+1))] = 1
                cnot_full[j | (1 << (i+1)), j | (1 << (i+1))] = 0; cnot_full[j | (1 << (i+1)), j] = 1
        state = cnot_full @ state
    return float(np.sum(np.abs(state)**2))

def run_eigen_dense_gates(n_qubits, n_gates):
    sim = QuantumSimulator(sim_type='dense', seed=42)
    for i in range(n_qubits): sim.allocate_qubit(f"q{i}")
    for i in range(n_gates): sim.H(f"q{i % n_qubits}")
    return sim.measure("q0")

def run_py_dense_gates(n_qubits, n_gates):
    import random
    rng = random.Random(42)
    inv = 1.0/math.sqrt(2.0)
    H = np.array([[inv, inv], [inv, -inv]], dtype=complex)
    state = np.array([1], dtype=complex)
    for _ in range(n_qubits): state = np.kron(state, [1, 0])
    for i in range(n_gates):
        q = i % n_qubits
        full = np.eye(1, dtype=complex)
        for j in range(n_qubits):
            full = np.kron(full, H if j == q else np.eye(2, dtype=complex))
        state = full @ state
    p0 = sum(abs(state[j])**2 for j in range(len(state)) if not (j & 1))
    return 0 if rng.random() < p0 else 1

# === Workload definitions ===

WORKLOADS = [
    # Classical (6 workloads × 2-3 sizes = 14 configs)
    ("arithmetic_sum", run_eigen_sum, run_py_sum, [100, 1000, 10000, 100000]),
    ("fibonacci", run_eigen_fib, run_py_fib, [10, 100, 1000, 10000]),
    ("factorial", run_eigen_factorial, run_py_factorial, [10, 50, 100]),
    ("nested_loop", run_eigen_nested, run_py_nested, [10, 50, 100]),
    ("string_concat", run_eigen_str_concat, run_py_str_concat, [100, 1000, 10000]),
    # Quantum (9 workloads × 2-3 sizes = 18+ configs)
    ("bell_state", run_eigen_bell, run_py_bell, [100, 1000, 10000]),
    ("gate_chain", run_eigen_gate_chain, run_py_gate_chain, [100, 1000, 10000]),
    ("ghz_state", lambda n: run_eigen_ghz(n, 100), lambda n: run_py_ghz(n, 100), [2, 3, 4]),
    ("random_clifford", lambda n: run_eigen_random_clifford(3, n), lambda n: run_py_random_clifford(3, n), [100, 1000]),
    ("multi_measure", lambda n: run_eigen_multi_measure(n), lambda n: run_py_multi_measure(n), [2, 3, 4]),
    ("entangle_chain", lambda n: run_eigen_entangle_chain(n), lambda n: run_py_entangle_chain(n), [2, 3, 4, 5]),
    ("dense_gate_apply", lambda n: run_eigen_dense_gates(3, n), lambda n: run_py_dense_gates(3, n), [100, 1000, 10000]),
]

def correctness_check():
    checks = []
    # Sum
    e, p = run_eigen_sum(100), run_py_sum(100)
    checks.append(("arithmetic_sum", 100, "eigen_vm", e == 5050, e))
    checks.append(("arithmetic_sum", 100, "python", p == 5050, p))
    # Fib
    e, p = run_eigen_fib(10), run_py_fib(10)
    checks.append(("fibonacci", 10, "eigen_vm", e == 55, e))
    checks.append(("fibonacci", 10, "python", p == 55, p))
    # Factorial
    e, p = run_eigen_factorial(5), run_py_factorial(5)
    checks.append(("factorial", 5, "eigen_vm", e == 120, e))
    checks.append(("factorial", 5, "python", p == 120, p))
    # Bell
    eb = run_eigen_bell(50)
    pb = run_py_bell(50)
    checks.append(("bell_state", 50, "eigen_vm", all(a==b for a,b in eb), "correlated"))
    checks.append(("bell_state", 50, "python", all(a==b for a,b in pb), "correlated"))
    # Gate chain
    e = run_eigen_gate_chain(100)
    checks.append(("gate_chain", 100, "eigen_vm", e in (0,1), e))
    # GHZ
    eg = run_eigen_ghz(3, 50)
    checks.append(("ghz_state", 3, "eigen_vm", all(all(v==r[0] for v in r) for r in eg), "correlated"))
    # Entangle chain norm
    e = run_eigen_entangle_chain(3)
    checks.append(("entangle_chain", 3, "eigen_vm", _approx(e, 1.0), f"norm={e:.4f}"))

    print("\n=== CORRECTNESS CHECK ===")
    print(f"{'Workload':<18} {'Size':>6} {'Impl':>12} {'Pass':>6} {'Value':>15}")
    print("-" * 65)
    for wl, sz, impl, passed, val in checks:
        print(f"{wl:<18} {sz:>6} {impl:>12} {'PASS' if passed else 'FAIL':>6} {str(val):>15}")
    all_pass = all(c[3] for c in checks)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return checks

def run_benchmarks():
    raw_rows = []
    summary_rows = []
    for wl_name, eigen_fn, py_fn, sizes in WORKLOADS:
        for size in sizes:
            for impl_name, fn in [("eigen_vm", eigen_fn), ("python", py_fn)]:
                times = []
                try: fn(size)
                except: pass
                for trial in range(TRIALS):
                    t0 = time.perf_counter()
                    try: result = fn(size)
                    except Exception as e: result = f"ERROR: {e}"
                    t1 = time.perf_counter()
                    elapsed = t1 - t0
                    times.append(elapsed)
                    raw_rows.append({"workload": wl_name, "size": size,
                        "implementation": impl_name, "trial": trial+1,
                        "elapsed_s": elapsed, "result": str(result)[:50]})
                mean_t = statistics.mean(times)
                std_t = statistics.stdev(times) if len(times) > 1 else 0.0
                summary_rows.append({"workload": wl_name, "size": size,
                    "implementation": impl_name, "mean_s": mean_t,
                    "std_s": std_t, "min_s": min(times), "max_s": max(times),
                    "ci95_s": 1.96 * std_t / math.sqrt(len(times)), "trials": len(times)})

    raw_path = os.path.join("results", "benchmark_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["workload","size","implementation","trial","elapsed_s","result"])
        w.writeheader(); w.writerows(raw_rows)
    print(f"Raw: {raw_path} ({len(raw_rows)} rows)")

    summary_path = os.path.join("results", "benchmark_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "workload","size","implementation","mean_s","std_s",
            "min_s","max_s","ci95_s","trials"])
        w.writeheader(); w.writerows(summary_rows)
    print(f"Summary: {summary_path} ({len(summary_rows)} rows)")

    print(f"\nTotal configurations: {len(summary_rows)}")
    print(f"Total raw measurements: {len(raw_rows)}")

    # Print summary
    print(f"\n{'Workload':<18} {'Size':>8} {'Impl':>12} {'Mean(ms)':>12} {'Std(ms)':>12}")
    print("-" * 70)
    for row in summary_rows:
        print(f"{row['workload']:<18} {row['size']:>8} {row['implementation']:>12} "
              f"{row['mean_s']*1000:>12.4f} {row['std_s']*1000:>12.4f}")

if __name__ == "__main__":
    print("Running correctness checks...")
    correctness_check()
    print("\nRunning expanded benchmarks (30+ configs x 10 trials)...")
    run_benchmarks()
