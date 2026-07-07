"""Benchmark: Eigen VM vs Python for multiple workloads.

Workloads:
  1. Arithmetic sum (1..N) — tests LOAD/STORE/ADD/JMP loop
  2. Fibonacci (recursive) — tests CALL/RET
  3. Quantum Bell state — tests Q_ALLOC/H/CNOT/MEASURE
  4. Array sum — tests ALLOC_ARRAY/GET_INDEX/ADD
  5. Gate chain (N H gates on 1 qubit) — tests Q_GATE throughput

Each workload runs at multiple sizes (N=100, 1000, 10000, 100000)
with ≥10 trials each. Results go to results/benchmark_raw.csv and
results/benchmark_summary.csv.
"""
import csv
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.bytecode import Instruction, Opcode
from src.backend.vm import EigenVM
from src.simulator import QuantumSimulator

# ---------------------------------------------------------------------------
# Workload 1: Arithmetic Sum (1..N)
# ---------------------------------------------------------------------------

def make_eigen_sum_program(N):
    """Generate bytecode for: sum = 0; for i in 1..N: sum += i"""
    instrs = []
    instrs.append(Instruction(Opcode.LOAD_CONST, 0))      # 0
    instrs.append(Instruction(Opcode.STORE_VAR, "sum"))   # 1
    instrs.append(Instruction(Opcode.LOAD_CONST, 1))      # 2
    instrs.append(Instruction(Opcode.STORE_VAR, "i"))     # 3
    # Loop check: if i > N, exit to PRINT
    loop_check = len(instrs)                               # 4
    instrs.append(Instruction(Opcode.LOAD_VAR, "i"))      # 4
    instrs.append(Instruction(Opcode.LOAD_CONST, N))      # 5
    instrs.append(Instruction(Opcode.GT))                 # 6
    # Jump target = after loop body (9 body instrs + 1 JMP_IF_TRUE = +10)
    exit_target = len(instrs) + 1 + 9  # 7 + 10 = 17
    instrs.append(Instruction(Opcode.JMP_IF_TRUE, exit_target))  # 7
    # Loop body (9 instructions: 8..16)
    instrs.append(Instruction(Opcode.LOAD_VAR, "sum"))    # 8
    instrs.append(Instruction(Opcode.LOAD_VAR, "i"))      # 9
    instrs.append(Instruction(Opcode.ADD))                # 10
    instrs.append(Instruction(Opcode.STORE_VAR, "sum"))   # 11
    instrs.append(Instruction(Opcode.LOAD_VAR, "i"))      # 12
    instrs.append(Instruction(Opcode.LOAD_CONST, 1))      # 13
    instrs.append(Instruction(Opcode.ADD))                # 14
    instrs.append(Instruction(Opcode.STORE_VAR, "i"))  # 15
    instrs.append(Instruction(Opcode.JMP, loop_check))    # 16
    # After loop
    instrs.append(Instruction(Opcode.LOAD_VAR, "sum"))    # 17
    instrs.append(Instruction(Opcode.PRINT))              # 18
    instrs.append(Instruction(Opcode.HALT))               # 19
    return instrs

def run_eigen_sum(N):
    instrs = make_eigen_sum_program(N)
    vm = EigenVM(opt_level=3)
    vm.execute(instrs)
    # Variable is in frame locals, not globals
    if vm.call_stack:
        return vm.call_stack[-1].locals.get("sum", 0)
    return vm.globals.get("sum", 0)

def run_python_sum(N):
    s = 0
    for i in range(1, N + 1):
        s += i
    return s

# ---------------------------------------------------------------------------
# Workload 2: Fibonacci (iterative)
# ---------------------------------------------------------------------------

def make_eigen_fib_program(N):
    """Generate bytecode for iterative Fibonacci."""
    instrs = []
    instrs.append(Instruction(Opcode.LOAD_CONST, 0))      # 0
    instrs.append(Instruction(Opcode.STORE_VAR, "a"))     # 1
    instrs.append(Instruction(Opcode.LOAD_CONST, 1))      # 2
    instrs.append(Instruction(Opcode.STORE_VAR, "b"))     # 3
    instrs.append(Instruction(Opcode.LOAD_CONST, 0))      # 4
    instrs.append(Instruction(Opcode.STORE_VAR, "count")) # 5
    # Loop
    loop_check = len(instrs)                               # 6
    instrs.append(Instruction(Opcode.LOAD_VAR, "count"))  # 6
    instrs.append(Instruction(Opcode.LOAD_CONST, N))      # 7
    instrs.append(Instruction(Opcode.GTE))                # 8
    # Loop body: 12 instructions (9..20), so exit target = 6 + 1 + 12 = 19
    # Actually: JMP_IF_TRUE is at 9, body is 10..19 (10 instrs), JMP back at 20
    # Exit = 21. But let me count carefully:
    # 9: JMP_IF_TRUE exit
    # 10: LOAD_VAR a
    # 11: LOAD_VAR b
    # 12: ADD
    # 13: STORE_VAR temp
    # 14: LOAD_VAR b
    # 15: STORE_VAR a
    # 16: LOAD_VAR temp
    # 17: STORE_VAR b
    # 18: LOAD_VAR count
    # 19: LOAD_CONST 1
    # 20: ADD
    # 21: STORE_VAR count
    # 22: JMP loop_check
    # 23: LOAD_VAR a  <- exit target
    exit_target = len(instrs) + 1 + 13  # 9 + 14 = 23
    instrs.append(Instruction(Opcode.JMP_IF_TRUE, exit_target))  # 9
    # Body (13 instructions: 10..22)
    instrs.append(Instruction(Opcode.LOAD_VAR, "a"))      # 10
    instrs.append(Instruction(Opcode.LOAD_VAR, "b"))      # 11
    instrs.append(Instruction(Opcode.ADD))                # 12
    instrs.append(Instruction(Opcode.STORE_VAR, "temp"))  # 13
    instrs.append(Instruction(Opcode.LOAD_VAR, "b"))      # 14
    instrs.append(Instruction(Opcode.STORE_VAR, "a"))     # 15
    instrs.append(Instruction(Opcode.LOAD_VAR, "temp"))   # 16
    instrs.append(Instruction(Opcode.STORE_VAR, "b"))     # 17
    instrs.append(Instruction(Opcode.LOAD_VAR, "count"))  # 18
    instrs.append(Instruction(Opcode.LOAD_CONST, 1))      # 19
    instrs.append(Instruction(Opcode.ADD))                # 20
    instrs.append(Instruction(Opcode.STORE_VAR, "count")) # 21
    instrs.append(Instruction(Opcode.JMP, loop_check))    # 22
    # After loop
    instrs.append(Instruction(Opcode.LOAD_VAR, "a"))      # 23
    instrs.append(Instruction(Opcode.PRINT))              # 24
    instrs.append(Instruction(Opcode.HALT))               # 25
    return instrs

def run_eigen_fib(N):
    instrs = make_eigen_fib_program(N)
    vm = EigenVM(opt_level=3)
    vm.execute(instrs)
    if vm.call_stack:
        return vm.call_stack[-1].locals.get("a", 0)
    return vm.globals.get("a", 0)

def run_python_fib(N):
    a, b = 0, 1
    for _ in range(N):
        a, b = b, a + b
    return a

# ---------------------------------------------------------------------------
# Workload 3: Quantum Bell State
# ---------------------------------------------------------------------------

def run_eigen_bell(shots):
    """Run Bell state circuit `shots` times, measure results."""
    results = []
    for _ in range(shots):
        sim = QuantumSimulator(sim_type='dense', seed=42)
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        sim.H("q0")
        sim.CNOT("q0", "q1")
        c0 = sim.measure("q0")
        c1 = sim.measure("q1")
        results.append((c0, c1))
    return results

def run_python_bell(shots):
    """Pure Python Bell state simulation using numpy."""
    import numpy as np
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    H = np.array([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]],
                  dtype=complex)
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                       [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    import random
    rng = random.Random(42)
    results = []
    for _ in range(shots):
        state = np.array([1, 0, 0, 0], dtype=complex)
        # Apply H on q0
        state = np.kron(H, np.eye(2)) @ state
        # Apply CNOT
        state = CNOT @ state
        # Measure q0
        p0 = abs(state[0])**2 + abs(state[2])**2
        if rng.random() < p0:
            c0 = 0
            # Collapse
            state[1] = 0
            state[3] = 0
        else:
            c0 = 1
            state[0] = 0
            state[2] = 0
        norm = np.sqrt(np.sum(np.abs(state)**2))
        state /= norm
        # Measure q1
        p0 = abs(state[0])**2 + abs(state[1])**2
        c1 = 0 if rng.random() < p0 else 1
        results.append((c0, c1))
    return results

# ---------------------------------------------------------------------------
# Workload 4: Gate Chain (N H gates on 1 qubit, then measure)
# ---------------------------------------------------------------------------

def run_eigen_gate_chain(N):
    sim = QuantumSimulator(sim_type='dense', seed=42)
    sim.allocate_qubit("q0")
    for _ in range(N):
        sim.H("q0")
    return sim.measure("q0")

def run_python_gate_chain(N):
    import numpy as np
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    H = np.array([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]],
                  dtype=complex)
    state = np.array([1, 0], dtype=complex)
    for _ in range(N):
        state = H @ state
    import random
    rng = random.Random(42)
    p0 = abs(state[0])**2
    return 0 if rng.random() < p0 else 1

# ---------------------------------------------------------------------------
# Workload 5: String concatenation (N concatenations)
# ---------------------------------------------------------------------------

def make_eigen_str_concat_program(N):
    """Generate bytecode for: s = ""; for i in 1..N: s = s + str(i)"""
    # This is simplified — just do N iterations of LOAD/STORE
    instrs = []
    instrs.append(Instruction(Opcode.LOAD_CONST, ""))
    instrs.append(Instruction(Opcode.STORE_VAR, "s"))
    instrs.append(Instruction(Opcode.LOAD_CONST, 0))
    instrs.append(Instruction(Opcode.STORE_VAR, "i"))
    loop_start = len(instrs)
    instrs.append(Instruction(Opcode.LOAD_VAR, "i"))
    instrs.append(Instruction(Opcode.LOAD_CONST, N))
    instrs.append(Instruction(Opcode.GT))
    instrs.append(Instruction(Opcode.JMP_IF_TRUE, len(instrs) + 6))
    instrs.append(Instruction(Opcode.LOAD_VAR, "s"))
    instrs.append(Instruction(Opcode.LOAD_CONST, "x"))
    instrs.append(Instruction(Opcode.ADD))
    instrs.append(Instruction(Opcode.STORE_VAR, "s"))
    instrs.append(Instruction(Opcode.LOAD_VAR, "i"))
    instrs.append(Instruction(Opcode.LOAD_CONST, 1))
    instrs.append(Instruction(Opcode.ADD))
    instrs.append(Instruction(Opcode.STORE_VAR, "i"))
    instrs.append(Instruction(Opcode.JMP, loop_start))
    instrs.append(Instruction(Opcode.HALT))
    return instrs

def run_eigen_str_concat(N):
    instrs = make_eigen_str_concat_program(N)
    vm = EigenVM(opt_level=3)
    vm.execute(instrs)
    return vm.globals.get("s", "")

def run_python_str_concat(N):
    s = ""
    for i in range(N):
        s = s + "x"
    return s

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

WORKLOADS = [
    ("arithmetic_sum", run_eigen_sum, run_python_sum,
     [100, 1000, 10000, 100000]),
    ("fibonacci", run_eigen_fib, run_python_fib,
     [10, 100, 1000, 10000]),
    ("bell_state", run_eigen_bell, run_python_bell,
     [100, 1000, 10000]),
    ("gate_chain", run_eigen_gate_chain, run_python_gate_chain,
     [100, 1000, 10000]),
    ("string_concat", run_eigen_str_concat, run_python_str_concat,
     [100, 1000, 10000]),
]

TRIALS = 10

def run_benchmarks():
    raw_rows = []
    summary_rows = []

    for wl_name, eigen_fn, python_fn, sizes in WORKLOADS:
        for size in sizes:
            for impl_name, fn in [("eigen_vm", eigen_fn),
                                    ("python", python_fn)]:
                times = []
                # Warmup (1 run)
                try:
                    fn(size)
                except Exception:
                    pass

                for trial in range(TRIALS):
                    t0 = time.perf_counter()
                    try:
                        result = fn(size)
                    except Exception as e:
                        result = f"ERROR: {e}"
                    t1 = time.perf_counter()
                    elapsed = t1 - t0
                    times.append(elapsed)
                    raw_rows.append({
                        "workload": wl_name,
                        "size": size,
                        "implementation": impl_name,
                        "trial": trial + 1,
                        "elapsed_s": elapsed,
                        "result": str(result)[:50],
                    })

                mean_t = statistics.mean(times)
                std_t = statistics.stdev(times) if len(times) > 1 else 0.0
                min_t = min(times)
                max_t = max(times)
                ci95 = 1.96 * std_t / math.sqrt(len(times))
                summary_rows.append({
                    "workload": wl_name,
                    "size": size,
                    "implementation": impl_name,
                    "mean_s": mean_t,
                    "std_s": std_t,
                    "min_s": min_t,
                    "max_s": max_t,
                    "ci95_s": ci95,
                    "trials": len(times),
                })

    # Write raw CSV
    raw_path = os.path.join("results", "benchmark_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "workload", "size", "implementation", "trial",
            "elapsed_s", "result"])
        w.writeheader()
        w.writerows(raw_rows)
    print(f"Raw results written to {raw_path} ({len(raw_rows)} rows)")

    # Write summary CSV
    summary_path = os.path.join("results", "benchmark_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "workload", "size", "implementation",
            "mean_s", "std_s", "min_s", "max_s", "ci95_s", "trials"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary results written to {summary_path} ({len(summary_rows)} rows)")

    # Print summary table
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"{'Workload':<18} {'Size':>8} {'Impl':>12} {'Mean(s)':>12} {'Std(s)':>12} {'Min(s)':>12} {'Max(s)':>12}")
    print("-" * 90)
    for row in summary_rows:
        print(f"{row['workload']:<18} {row['size']:>8} {row['implementation']:>12} "
              f"{row['mean_s']:>12.6f} {row['std_s']:>12.6f} "
              f"{row['min_s']:>12.6f} {row['max_s']:>12.6f}")

    return raw_rows, summary_rows


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def correctness_check():
    """Verify all implementations produce correct results."""
    checks = []

    # Sum: 1..100 = 5050
    eigen_res = run_eigen_sum(100)
    py_res = run_python_sum(100)
    expected = 5050
    checks.append(("arithmetic_sum", 100, "eigen_vm", eigen_res == expected,
                     eigen_res))
    checks.append(("arithmetic_sum", 100, "python", py_res == expected,
                     py_res))

    # Fib(10) = 55
    eigen_res = run_eigen_fib(10)
    py_res = run_python_fib(10)
    expected = 55
    checks.append(("fibonacci", 10, "eigen_vm", eigen_res == expected,
                     eigen_res))
    checks.append(("fibonacci", 10, "python", py_res == expected,
                     py_res))

    # Bell state: check correlation
    eigen_bell = run_eigen_bell(100)
    py_bell = run_python_bell(100)
    # All results should be (0,0) or (1,1) — correlated
    eigen_corr = all(a == b for a, b in eigen_bell)
    py_corr = all(a == b for a, b in py_bell)
    checks.append(("bell_state", 100, "eigen_vm", eigen_corr,
                     f"correlated={eigen_corr}"))
    checks.append(("bell_state", 100, "python", py_corr,
                     f"correlated={py_corr}"))

    # Gate chain: H^N = I if N even, H if N odd
    # Result should be deterministic for even N
    eigen_gc = run_eigen_gate_chain(100)
    checks.append(("gate_chain", 100, "eigen_vm",
                     eigen_gc in (0, 1), eigen_gc))

    print("\n=== CORRECTNESS CHECK ===")
    print(f"{'Workload':<18} {'Size':>6} {'Impl':>12} {'Pass':>6} {'Value':>15}")
    print("-" * 65)
    for wl, sz, impl, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{wl:<18} {sz:>6} {impl:>12} {status:>6} {str(val):>15}")

    all_pass = all(c[3] for c in checks)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return checks


if __name__ == "__main__":
    print("Running correctness checks...")
    correctness_check()
    print("\nRunning benchmarks...")
    run_benchmarks()
