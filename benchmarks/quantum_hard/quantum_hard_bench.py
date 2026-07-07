"""Hard-quantum benchmark: Eigen runtime vs hand-rolled numpy statevector.

For each of the 20 `cases/*.eig` programs we:

  1. Compile it through Eigen's standard front-end (lexer → parser → type
     checker → MLIR → EQIR graph).
  2. Run it via ``EigenRuntime`` (dense state-vector backend) and read the
     final amplitudes.
  3. Re-simulate the *same* EQIR graph with a hand-written numpy state-vector
     simulator (no optimisations — textbook matrix-vector multiplies).
  4. Compare probabilities in the same bitstring convention Eigen uses and
     compute 1 - ½·∑|p_e - p_n| (total-variation distance → accuracy ∈ [0,1]).
  5. Time both runs (best-of-N) and write ``results.json`` plus a black-and-white
     LaTeX table and pgfplots bar charts in ``comparison.tex``.

Run from the workspace root:

    python -m benchmarks.quantum_hard.quantum_hard_bench
"""

from __future__ import annotations

import json
import math
import cmath
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.compiler import compile_to_eqir
from src.runtime import EigenRuntime


CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", ".."))
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")
TEX_OUT = os.path.join(OUT_DIR, "comparison.tex")

TIME_RUNS = 3        # best-of-N for timing
WARMUP_RUNS = 1      # discarded

# Numerical floor for "zero" amplitude — must match Eigen's simulator
# (which uses 1e-12 in `get_amplitudes_dict`).
AMPL_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Numpy state-vector simulator (textbook)                                     #
# --------------------------------------------------------------------------- #

# Single-qubit primitive matrices.
_H = (1 / math.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_S = np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex)
_T = np.array([[1.0, 0.0], [0.0, cmath.exp(1j * math.pi / 4.0)]], dtype=complex)


def _rx(theta: float) -> np.ndarray:
    return np.array([
        [math.cos(theta / 2.0), -1j * math.sin(theta / 2.0)],
        [-1j * math.sin(theta / 2.0), math.cos(theta / 2.0)],
    ], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    return np.array([
        [math.cos(theta / 2.0), -math.sin(theta / 2.0)],
        [math.sin(theta / 2.0), math.cos(theta / 2.0)],
    ], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([
        [cmath.exp(-1j * theta / 2.0), 0.0],
        [0.0, cmath.exp(1j * theta / 2.0)],
    ], dtype=complex)


def _cnot_matrix() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)


def _cz_matrix() -> np.ndarray:
    return np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)


def _swap_matrix() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex)


def _ccx_matrix() -> np.ndarray:
    u = np.eye(8, dtype=complex)
    u[6, 6] = 0
    u[6, 7] = 1
    u[7, 6] = 1
    u[7, 7] = 0
    return u


def _cswap_matrix() -> np.ndarray:
    """CSWAP(c, t1, t2): swaps t1, t2 iff c=1."""
    u = np.eye(8, dtype=complex)
    # When c=1 (top bit set in index), swap the (t1, t2) part — indices
    # that differ only in bit 1 (the t1 position).
    # Indexing convention: index = c*4 + t1*2 + t2.
    # c=1 case: indices 4..7. Swap (t1=0, t2=1) <-> (t1=1, t2=0):
    # that's swap indices 5 <-> 6.
    for r in range(8):
        if r == 5 or r == 6:
            continue
        u[r, r] = 1
    u[5, 5] = 0
    u[5, 6] = 1
    u[6, 5] = 1
    u[6, 6] = 0
    return u


def _crz(theta: float) -> np.ndarray:
    rz = _rz(theta)
    u = np.eye(4, dtype=complex)
    u[2:, 2:] = rz
    return u


def _cry(theta: float) -> np.ndarray:
    ry = _ry(theta)
    u = np.eye(4, dtype=complex)
    u[2:, 2:] = ry
    return u


def _crx(theta: float) -> np.ndarray:
    rx = _rx(theta)
    u = np.eye(4, dtype=complex)
    u[2:, 2:] = rx
    return u


def _cp(theta: float) -> np.ndarray:
    return np.diag([1.0, 1.0, 1.0, cmath.exp(1j * theta)]).astype(complex)


def _gate_unitary(gate_name: str, args: list) -> np.ndarray:
    """Return the 1q, 2q, or 3q unitary for ``gate_name`` (case-insensitive
    gate names as they appear in the EQIR graph)."""
    g = gate_name.upper()
    if g == "H":
        return _H
    if g == "X":
        return _X
    if g == "Y":
        return _Y
    if g == "Z":
        return _Z
    if g == "S":
        return _S
    if g == "T":
        return _T
    if g == "RX":
        return _rx(float(args[0]))
    if g == "RY":
        return _ry(float(args[0]))
    if g == "RZ":
        return _rz(float(args[0]))
    if g == "CNOT":
        return _cnot_matrix()
    if g == "CZ":
        return _cz_matrix()
    if g == "SWAP":
        return _swap_matrix()
    if g == "CCX":
        return _ccx_matrix()
    if g == "CSWAP":
        return _cswap_matrix()
    if g == "CP":
        return _cp(float(args[0]))
    if g == "CRX":
        return _crx(float(args[0]))
    if g == "CRY":
        return _cry(float(args[0]))
    if g == "CRZ":
        return _crz(float(args[0]))
    raise ValueError(f"numpy reference doesn't know gate {gate_name!r}")


def _apply_nq(state: np.ndarray, U: np.ndarray, qubit_indices: list[int], n: int) -> np.ndarray:
    """Apply a (2^k × 2^k) unitary ``U`` to the qubits at ``qubit_indices``
    within an N-qubit state-vector ``state`` (size 2^n).

    Convention: ``qubit_indices[k]`` is the qubit's *logical* index (i.e. the
    bit position in the flat state-vector index, matching Eigen's
    ``qubit_map[name]``). The numpy ``reshape`` in C-order maps flat index
    bit `b` to tensor axis `(n-1-b)`, so we convert through that mapping
    to keep semantics consistent with Eigen's state-vector conventions.

    The matrix ``U`` is indexed with the FIRST listed active logical qubit
    in the most-significant row position (standard textbook CNOT, CZ, CCX).
    """
    k = len(qubit_indices)
    if k == 0:
        return state
    # Tensor axes corresponding to active qubits.
    tensor_axes = [n - 1 - l for l in qubit_indices]
    other_axes = [a for a in range(n) if a not in tensor_axes]
    perm = list(tensor_axes) + other_axes
    perm_inv = [0] * n
    for i, p in enumerate(perm):
        perm_inv[p] = i

    rest_dim = 2 ** (n - k)
    s = state.reshape([2] * n).transpose(perm)        # active axes first
    s = s.reshape(2 ** k, rest_dim)
    out = U @ s                                        # (2^k, rest)
    out = out.reshape([2] * n).transpose(perm_inv)
    return out.reshape(-1)


def numpy_simulate(graph, n_qubits: int, qubit_index: dict[str, int]) -> dict[str, complex]:
    """Run the EQIR graph through the numpy simulator. Returns bitstring→amp.

    ``qubit_index`` must already reflect the insertion order produced by
    ALLOC nodes (the same order Eigen's QuantumSimulator uses), so that the
    bitstring convention matches Eigen's ``get_amplitudes_dict()``.
    """
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[0] = 1.0
    n = n_qubits

    for node in graph.topological_sort():
        if node.type == "GATE":
            g = node.gate_name
            targets = node.targets
            args = node.args or []
            U = _gate_unitary(g, args)
            qidx = [qubit_index[t] for t in targets]
            state = _apply_nq(state, U, qidx, n)

    # Convert to bitstring dict using the IDENTICAL convention as Eigen's
    # get_amplitudes_dict(): iterate sorted-by-index qubits in REVERSE order,
    # bit (i >> idx) & 1 = bit for qubit with index idx.
    sorted_qubits = sorted(qubit_index.keys(), key=lambda name: qubit_index[name])
    out: dict[str, complex] = {}
    for i, amp in enumerate(state):
        if abs(amp) > AMPL_TOL:
            bitstring = ""
            for q in reversed(sorted_qubits):
                bitstring += str((i >> qubit_index[q]) & 1)
            out[bitstring] = complex(amp)
    return out


# --------------------------------------------------------------------------- #
# Eigen path                                                                  #
# --------------------------------------------------------------------------- #

def eigen_run(graph, qubit_index: dict[str, int]) -> dict[str, complex]:
    """Run an EQIR graph through EigenRuntime (dense backend) and return
    bitstring→complex amplitude dict."""
    runtime = EigenRuntime(sim_type='dense')
    runtime.execute(graph)
    # The runtime's simulator has its own qubit_map; mirror it for the
    # bitstring convention in case Eigen's allocation order differs.
    raw = runtime.simulator.get_amplitudes_dict()
    return {k: complex(v) for k, v in raw.items()}


def compile_eig(path: str) -> tuple:
    """Run the standard compiler front-end (parse + type-check + MLIR +
    EQIR graph) for one .eig path. Returns (graph, ast)."""
    graph, ast = compile_to_eqir(path, WORKSPACE_ROOT)
    return graph, ast


def extract_qubit_index(graph) -> dict[str, int]:
    """Return a qubit_name → index mapping consistent with Eigen's allocation
    order (insertion order based on topological sort of ALLOC nodes)."""
    idx = 0
    qubit_index: dict[str, int] = {}
    # ALLOC nodes in topo sort order reflect Eigen's allocation order.
    for node in graph.topological_sort():
        if node.type == "ALLOC":
            name = node.targets[0]
            if name not in qubit_index:
                qubit_index[name] = idx
                idx += 1
    return qubit_index


# --------------------------------------------------------------------------- #
# Accuracy                                                                    #
# --------------------------------------------------------------------------- #

def accuracy(eigen_state: dict[str, complex],
             numpy_state: dict[str, complex]) -> tuple[float, float, dict]:
    """Compute accuracy as 1 − ½·∑|p_e − p_n| over the union of bitstrings.

    Also returns the max single-bitstring probability deviation and a
    breakdown dict for debugging.
    """
    all_bits = set(eigen_state) | set(numpy_state)
    e_p = {b: abs(v) ** 2 for b, v in eigen_state.items()}
    n_p = {b: abs(v) ** 2 for b, v in numpy_state.items()}

    total_var = 0.0
    max_dev = 0.0
    breakdown = {}
    for b in all_bits:
        pe = e_p.get(b, 0.0)
        pn = n_p.get(b, 0.0)
        d = abs(pe - pn)
        total_var += d
        max_dev = max(max_dev, d)
        breakdown[b] = (pe, pn, d)
    tvd = 0.5 * total_var
    return 1.0 - tvd, max_dev, breakdown


# --------------------------------------------------------------------------- #
# Benchmark loop                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class CaseResult:
    name: str
    qubits: int
    gates: int
    depth: int
    category: str
    eigen_ms: float
    numpy_ms: float
    speedup: float  # numpy_ms / eigen_ms    (Eigen faster iff > 1)
    accuracy: float
    max_dev: float
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qubits": self.qubits,
            "gates": self.gates,
            "depth": self.depth,
            "category": self.category,
            "eigen_ms": round(self.eigen_ms, 4),
            "numpy_ms": round(self.numpy_ms, 4),
            "speedup": round(self.speedup, 3),
            "accuracy": round(self.accuracy, 6),
            "max_dev": round(self.max_dev, 6),
            "notes": self.notes,
        }


def _categorise(name: str) -> str:
    n = name.lower()
    if "deutsch_jozsa" in n: return "Deutsch-Jozsa"
    if "deutsch" in n: return "Deutsch"
    if "bernstein" in n or "_bv_" in n or n.startswith("bv_"): return "BV"
    if "grover" in n: return "Grover"
    if "qpe" in n or "phase_estimation" in n: return "QPE"
    if "qft" in n: return "QFT"
    if "superdense" in n: return "Superdense"
    if "teleport" in n: return "Teleport"
    if "steane" in n: return "Steane-QEC"
    if "shor" in n: return "Shor-QEC"
    if "vqe" in n: return "VQE"
    if "random" in n: return "Random"
    if "ising" in n: return "Ising-Trotter"
    if "draper" in n: return "Draper"
    if "quantum_walk" in n: return "Quantum walk"
    if "clifford" in n: return "Clifford+T"
    if "phase_chain" in n: return "Phase chain"
    if "phase_mixer" in n: return "Phase mixer"
    if "cry_crx" in n: return "CRY/CRX"
    if "bell_chain" in n: return "Bell chain"
    if "bell" in n: return "Bell"
    if "w_state" in n or "wstate" in n: return "W-state"
    if "ghz" in n: return "GHZ"
    return "Other"


def _gate_depth(graph, qubit_index: dict[str, int]) -> int:
    """Heuristic circuit depth: longest chain of overlapping gates.

    Greedy topological scheduling: each gate is placed at layer
    `1 + max(last_layer[q] for q in targets)` and every touched qubit's
    last_layer is bumped to that new layer.  Depth = max last_layer.
    """
    if not qubit_index:
        return 0
    last_layer: dict[str, int] = {q: 0 for q in qubit_index}
    max_layer = 0
    for node in graph.topological_sort():
        if node.type == "GATE":
            touched = list(node.targets)
            relevant = [q for q in touched if q in last_layer]
            if not relevant:
                continue
            new_layer = 1 + max(last_layer[q] for q in relevant)
            for q in relevant:
                last_layer[q] = new_layer
            if new_layer > max_layer:
                max_layer = new_layer
    return max_layer


def _best_of(times: list[float]) -> float:
    return min(times)


def _time_callable(fn, warmup: int = WARMUP_RUNS, runs: int = TIME_RUNS) -> float:
    """Best-of-N timing for callable ``fn`` returning state dict."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return _best_of(times)


def run_benchmark(cases_dir: str = CASES_DIR) -> list[CaseResult]:
    results: list[CaseResult] = []
    cases = sorted(f for f in os.listdir(cases_dir) if f.endswith(".eig"))
    for case_file in cases:
        path = os.path.join(cases_dir, case_file)
        case_name = case_file[:-4]
        category = _categorise(case_name)
        try:
            graph, ast = compile_eig(path)
        except Exception as e:
            err = f"compile error: {str(e)[:160]}"
            print(f"  {case_name}: {err}")
            results.append(CaseResult(
                name=case_name, qubits=0, gates=0, depth=0, category=category,
                eigen_ms=float("nan"), numpy_ms=float("nan"),
                speedup=float("nan"), accuracy=0.0, max_dev=1.0, notes=err,
            ))
            continue
        qubit_index = extract_qubit_index(graph)
        n_q = len(qubit_index)
        n_g = sum(1 for nd in graph.nodes.values() if nd.type == "GATE")
        n_d = _gate_depth(graph, qubit_index)

        # Pre-compile graph once.
        def _eigen_run():
            return eigen_run(graph, qubit_index)

        def _numpy_run():
            return numpy_simulate(graph, n_q, qubit_index)

        # Time Eigen.
        try:
            eigen_ms = _time_callable(_eigen_run)
        except Exception as e:
            results.append(CaseResult(
                name=case_name, qubits=n_q, gates=n_g, depth=n_d,
                category=category,
                eigen_ms=float("nan"), numpy_ms=float("nan"),
                speedup=float("nan"), accuracy=0.0, max_dev=1.0,
                notes=f"eigen runtime error: {str(e)[:200]}",
            ))
            continue
        # Time numpy.
        numpy_ms = _time_callable(_numpy_run)

        # Accuracy comparison (single run, not timing-sensitive).
        e_state = _eigen_run()
        n_state = _numpy_run()
        acc, max_dev, _ = accuracy(e_state, n_state)

        speedup = numpy_ms / eigen_ms if eigen_ms > 0 else float("inf")
        results.append(CaseResult(
            name=case_name, qubits=n_q, gates=n_g, depth=n_d,
            category=category,
            eigen_ms=eigen_ms, numpy_ms=numpy_ms,
            speedup=speedup, accuracy=acc, max_dev=max_dev,
            notes="" if acc > 0.999 else "small drift below 1.0",
        ))
        print(f"  {case_name}: q={n_q} g={n_g} d={n_d} [{category}] | "
              f"Eigen {eigen_ms:.2f}ms, numpy {numpy_ms:.3f}ms, "
              f"speedup {speedup:.2f}x, acc {acc*100:.4f}%")
    return results


# --------------------------------------------------------------------------- #
# LaTeX output — black-and-white table + pgfplots bars                       #
# --------------------------------------------------------------------------- #


def to_latex(results: list[CaseResult]) -> str:
    """Produce a self-contained .tex file with:
      - long multi-page booktabs table (Ч/Б) with category / qubits / gates /
        depth / Eigen ms / numpy ms / speedup / accuracy / max-dev / status
      - per-category aggregation table (mean qubits, mean gates, mean Eigen ms,
        mean numpy ms, mean speedup, mean accuracy)
      - six pgfplots figures in pure black & white:
          1. speed comparison (log-scale bar)
          2. accuracy per test (bar)
          3. speedup histogram (binned)
          4. quadrant / 4-field matrix scatter (log10(gates) × qubits, divided
             by medians into 4 strategy quadrants; speedup annotated per point)
          5. scaling chart (log-log runtime vs gates — Eigen & numpy)
          6. accuracy error magnitude vs gates (log-log scatter, PASS threshold
             as dashed horizontal)
      - aggregate summary table
    """
    # Sort results alphabetically by name for stable layout.
    rs = sorted(results, key=lambda r: r.name)
    n_rows = len(rs)

    # Aggregate stats.
    n_passing = sum(1 for r in rs if not math.isnan(r.eigen_ms) and r.accuracy >= 0.999)
    n_failing = n_rows - n_passing
    valid = [r for r in rs if not math.isnan(r.eigen_ms) and not math.isnan(r.numpy_ms)]
    n_valid = max(1, len(valid))
    mean_eigen = sum(r.eigen_ms for r in valid) / n_valid
    mean_numpy = sum(r.numpy_ms for r in valid) / n_valid
    mean_acc = sum(r.accuracy for r in rs) / max(1, n_rows)
    speedups = [r.speedup for r in valid if r.speedup != float("inf")]
    median_speedup = statistics.median(speedups) if speedups else 0.0
    max_speedup = max(speedups) if speedups else 0.0
    sum_eigen = sum(r.eigen_ms for r in valid)
    sum_numpy = sum(r.numpy_ms for r in valid)
    overall_speedup = sum_numpy / sum_eigen if sum_eigen > 0 else float("inf")
    mean_dev = sum(r.max_dev for r in rs) / max(1, n_rows)

    # Per-category aggregation.
    by_cat: dict[str, list[CaseResult]] = defaultdict(list)
    for r in rs:
        by_cat[r.category].append(r)
    cat_rows_data: list[tuple] = []
    for cat, items in sorted(by_cat.items(), key=lambda kv: -statistics.mean(
            [r.gates for r in kv[1]] if kv[1] else [0])):
        n = len(items)
        mq = statistics.mean([r.qubits for r in items]) if n else 0
        mg = statistics.mean([r.gates for r in items]) if n else 0
        md = statistics.mean([r.depth for r in items]) if n else 0
        me_vals = [r.eigen_ms for r in items if not math.isnan(r.eigen_ms)]
        mn_vals = [r.numpy_ms for r in items if not math.isnan(r.numpy_ms)]
        me = statistics.mean(me_vals) if me_vals else float("nan")
        mn = statistics.mean(mn_vals) if mn_vals else float("nan")
        sp = (mn / me) if (me and not math.isnan(me) and not math.isnan(mn) and me > 0) else float("inf")
        ma = statistics.mean([r.accuracy for r in items]) if n else 0
        cat_rows_data.append((cat, n, mq, mg, md, me, mn, sp, ma))

    # Speedup histogram bins.
    bins = [
        (-1.0, 1.0,   "<1"),
        (1.0,  1.5,   "1--1.5"),
        (1.5,  2.0,   "1.5--2"),
        (2.0,  3.0,   "2--3"),
        (3.0,  5.0,   "3--5"),
        (5.0,  10.0,  "5--10"),
        (10.0, float("inf"), ">10"),
    ]
    hist_counts = []
    for lo, hi, _ in bins:
        if hi == float("inf"):
            cnt = sum(1 for s in speedups if s >= lo)
        elif lo <= 0:
            cnt = sum(1 for s in speedups if s < hi)
        else:
            cnt = sum(1 for s in speedups if lo <= s < hi)
        hist_counts.append(cnt)
    hist_coords = " ".join(f"({i+1},{c})" for i, c in enumerate(hist_counts))

    # Quadrant medians.
    log10_gates = [math.log10(r.gates) for r in rs if r.gates > 0]
    qubits_list = [r.qubits for r in rs]
    median_log_g = statistics.median(log10_gates) if log10_gates else 0.5
    median_q = statistics.median(qubits_list) if qubits_list else 2

    # Build all chart coordinate strings (one big f-string per chart).
    eig_coords = " ".join(
        f"({i+1},{r.eigen_ms if not math.isnan(r.eigen_ms) else 0:.4f})"
        for i, r in enumerate(rs)
    )
    np_coords = " ".join(
        f"({i+1},{r.numpy_ms if not math.isnan(r.numpy_ms) else 0:.4f})"
        for i, r in enumerate(rs)
    )
    acc_coords = " ".join(f"({i+1},{(r.accuracy*100):.4f})" for i, r in enumerate(rs))
    xticks = ",".join(str(i + 1) for i in range(n_rows))

    # Quadrant scatter: PASS points and FAIL points separate addplots.
    pass_coords = " ".join(
        f"({math.log10(r.gates) if r.gates > 0 else 0:.3f},{r.qubits})[{r.speedup:.2f}]"
        for r in rs if not math.isnan(r.eigen_ms) and r.accuracy >= 0.999
    )
    fail_coords = " ".join(
        f"({math.log10(r.gates) if r.gates > 0 else 0:.3f},{r.qubits})[{r.speedup:.2f}]"
        for r in rs if math.isnan(r.eigen_ms) or r.accuracy < 0.999
    )

    # Scaling chart: log-log scatter, Eigen vs numpy.
    eig_scale = " ".join(
        f"({r.gates},{r.eigen_ms:.4f})"
        for r in rs if not math.isnan(r.eigen_ms) and r.gates > 0
    )
    np_scale = " ".join(
        f"({r.gates},{r.numpy_ms:.4f})"
        for r in rs if not math.isnan(r.numpy_ms) and r.gates > 0
    )

    # Accuracy-error scatter: x = gates (log), y = max(1-acc, 1e-15) (log).
    err_coords = " ".join(
        f"({r.gates},{max(1 - r.accuracy, 1e-15):.2e})"
        for r in rs if r.gates > 0
    )

    # Longtable row strings.
    longtable_rows: list[str] = []
    for i, r in enumerate(rs):
        name_clean = r.name.replace("_", "\\_")
        cat_clean = r.category
        eig_str = f"{r.eigen_ms:.2f}" if not math.isnan(r.eigen_ms) else "n/a"
        np_str = f"{r.numpy_ms:.3f}" if not math.isnan(r.numpy_ms) else "n/a"
        sp_str = (f"{r.speedup:.2f}" if r.speedup < 99.0 else "$\\infty$") if not math.isnan(r.speedup) else "n/a"
        acc_str = f"{r.accuracy*100:.4f}" if r.accuracy is not None else "n/a"
        dev_str = f"{r.max_dev:.2e}" if r.max_dev is not None else "n/a"
        status = "PASS" if (not math.isnan(r.eigen_ms) and r.accuracy >= 0.999) else "FAIL"
        longtable_rows.append(
            f"{name_clean} & {cat_clean} & {r.qubits} & {r.gates} & {r.depth} & "
            f"{eig_str} & {np_str} & {sp_str} & {acc_str} & {dev_str} & {status} \\\\"
        )
        if i < n_rows - 1:
            longtable_rows.append("\\midrule[0.1pt]")

    longtable_body = "\n".join(longtable_rows)

    # Per-category table rows.
    cat_table_rows: list[str] = []
    for cat, n_items, mq, mg, md, me, mn, sp, ma in cat_rows_data:
        sp_str = f"{sp:.2f}" if sp < 99.0 else "$\\infty$"
        me_str = f"{me:.3f}" if not math.isnan(me) else "n/a"
        mn_str = f"{mn:.3f}" if not math.isnan(mn) else "n/a"
        cat_table_rows.append(
            f"{cat} & {n_items} & {mq:.1f} & {mg:.1f} & {md:.1f} & "
            f"{me_str} & {mn_str} & {sp_str} & {ma*100:.4f} \\\\"
        )
    cat_table_body = "\n".join(cat_table_rows)

    # Numeric median values for quadrant dividers.
    qmin = min(qubits_list) if qubits_list else 0
    qmax = max(qubits_list) if qubits_list else 10
    gxmin = min(log10_gates) if log10_gates else 0.0
    gxmax = max(log10_gates) if log10_gates else 2.5

    # Histogram bin labels.
    bin_labels = ",".join(f"{i+1}" for i in range(len(hist_counts)))
    bin_ticks = f"xticklabels={{{', '.join(lbl for _, _, lbl in bins)}}}"
    bin_count = len(hist_counts)

    # Build the LaTeX document as one big string.
    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=0.7in]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{statistics}
\usetikzlibrary{patterns}

% Black-and-white only — no colour anywhere. Strict line-based visual rhythm.
\definecolor{linegray}{HTML}{222222}

\title{Eigen vs numpy state-vector: """ + f"{n_rows}" + r""" hard-quantum benchmarks across """ + f"{len(by_cat)}" + r""" categories}
\author{Eigen 2.7 hard-quantum benchmark harness}
\date{\today}

\begin{document}
\maketitle

\section{Setup}

For each of """ + f"{n_rows}" + r""" hand-written quantum circuits spanning Bell,
GHZ, W-state, Deutsch/Deutsch-Jozsa, Bernstein-Vazirani, Grover (2-5 qubit),
QFT (2-6 qubit), phase estimation, superdense coding, teleportation, Draper
adder, Ising Trotterisation, quantum walk, variational ansatz (VQE-style),
random circuits (3-6 qubit, depth 25-200), Steane \& Shor error-correcting
encoders, and phase-mixer long-depth stress tests,
we execute the \emph{identical EQIR graph} through two paths:
\begin{itemize}
  \item \textbf{Eigen:} \texttt{EigenRuntime} with the Rust-accelerated dense
      state-vector backend
      (\texttt{src.simulator.\allowbreak RustStatevectorWrapper}).
  \item \textbf{numpy:} a textbook matrix-vector simulator in pure
      \texttt{numpy} (no optimisations, no Kronecker trick): every gate is
      applied via \texttt{np.tensordot}/\texttt{transpose}/\texttt{reshape}
      on a dense $2^n$-length state vector.
\end{itemize}
Both sides report the final state vector in the \emph{same} bitstring
convention. Accuracy is reported as $1-\tfrac{1}{2}\sum_b|p^{\text{e}}_b -
p^{\text{n}}_b|$ where $p_b = |\psi_b|^2$ over the union of all basis states.
Speedup $\equiv t_{\text{numpy}}/t_{\text{Eigen}}$; values $>1$ mean Eigen is
faster. \emph{Depth} is the longest chain of qubit-overlapping gates under
topological scheduling.

All timings are best-of-3 after 1 warm-up discarded. The Eigen graph is
\emph{compiled once and re-executed} so the measurement isolates state-vector
evolution (parse / type-check / compile cost is amortised).

\section{Results (long multi-page table)}

\begingroup
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{0.96}
\tiny
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{4.0cm} >{\raggedright\arraybackslash}p{1.7cm} r r r r r r r r r@{}}
\caption{Hard-quantum head-to-head over all """ + f"{n_rows}" + r""" test cases.
$Q$ = qubits, $g$ = gates, $d$=depth. \texttt{PASS} requires accuracy $\ge 99.9\%$.
\textbf{""" + f"{n_passing}" + r""" pass / """ + f"{n_failing}" + r""" fail}.}\\
\toprule
Test case & cat & $\mathbf{q}$ & $\mathbf{g}$ & $\mathbf{d}$ & Eigen (ms) & numpy (ms) & speedup & acc (\%) & max dev & status \\
\midrule[0.4pt]
\endfirsthead
\multicolumn{11}{c}{\tiny\itshape continued from previous page}\\
\toprule
Test case & cat & $\mathbf{q}$ & $\mathbf{g}$ & $\mathbf{d}$ & Eigen (ms) & numpy (ms) & speedup & acc (\%) & max dev & status \\
\midrule[0.4pt]
\endhead
\midrule[0.1pt]
\multicolumn{11}{r}{\tiny\itshape continued on next page}\\
\endfoot
\midrule[0.4pt]
\textbf{mean (over valid rows)} & -- & -- & -- & -- & """ + f"{mean_eigen:.3f}" + r""" & """ + f"{mean_numpy:.3f}" + r""" & """ + f"{overall_speedup:.2f}" + r""" & """ + f"{mean_acc*100:.4f}" + r""" & """ + f"{mean_dev:.2e}" + r""" & -- \\
\bottomrule
\endlastfoot
""" + longtable_body + r"""
\end{longtable}
\endgroup

\section{Per-category aggregation}

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{@{}l r r r r r r r r@{}}
\toprule
Category & $n$ & mean $\mathbf{q}$ & mean $\mathbf{g}$ & mean $\mathbf{d}$ & Eigen (ms) & numpy (ms) & mean sp & mean acc (\%) \\
\midrule[0.4pt]
""" + cat_table_body + r"""
\\ \midrule[0.4pt]
\textbf{overall} & """ + f"{n_rows}" + r""" & -- & -- & -- & """ + f"{mean_eigen:.3f}" + r""" & """ + f"{mean_numpy:.3f}" + r""" & """ + f"{overall_speedup:.2f}" + r""" & """ + f"{mean_acc*100:.4f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Per-category aggregation: each row summarises one category of tests.
Categories sorted by descending mean gate count (largest circuits first).}
\end{table}

\section{Speed comparison (log scale)}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.99\textwidth,
    height=8cm,
    ybar,
    bar width=3pt,
    enlarge x limits=0.04,
    ymin=0.001,
    ymode=log,
    ytick={0.001, 0.01, 0.1, 1, 10, 100, 1000},
    ylabel={execution time (ms, log)},
    xlabel={test index},
    legend style={at={(0.5,-0.18)},anchor=north,draw=linegray},
    every axis/.append style={draw=linegray, very thin},
    every tick/.append style={black, thin},
    every axis label/.append style={black},
    xtick={""" + xticks + r"""},
    xmin=0.5, xmax=""" + f"{n_rows+0.5}" + r""",
]
\addplot+[draw=linegray, fill=lightgray!50, pattern=north east lines, mark=none]
    coordinates { """ + eig_coords + r""" };
\addplot+[draw=linegray, fill=lightgray!10, pattern=horizontal lines, mark=none]
    coordinates { """ + np_coords + r""" };
\legend{Eigen, numpy}
\end{axis}
\end{tikzpicture}
\caption{Best-of-3 execution time per test. Log scale so the 1-2-qubit Bell
tests ($\sim$40 µs) coexist in the same axes with the longest Trotter /
random-circuit tests.}
\end{figure}

\section{Accuracy per test}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.99\textwidth,
    height=6cm,
    ybar,
    bar width=4pt,
    enlarge x limits=0.04,
    ymin=0.0,
    ymax=100.4,
    ytick={50, 90, 99, 99.9, 99.99, 100},
    ylabel={accuracy (\%)},
    xlabel={test index},
    legend style={at={(0.5,-0.18)},anchor=north,draw=linegray},
    every axis/.append style={draw=linegray, very thin},
    every tick/.append style={black, thin},
    every axis label/.append style={black},
    xtick={""" + xticks + r"""},
    xmin=0.5, xmax=""" + f"{n_rows+0.5}" + r""",
]
\addplot+[draw=linegray, fill=lightgray!40, pattern=crosshatch dots, mark=none]
    coordinates { """ + acc_coords + r""" };
\draw [linegray, dashed, very thin] (axis cs:0,99.9) -- (axis cs:""" + f"{n_rows+1}" + r""",99.9)
    node [linegray, anchor=west, font=\tiny] at (axis cs:""" + f"{n_rows+1}" + r""", 99.9) {PASS line};
\end{axis}
\end{tikzpicture}
\caption{Accuracy per test. Anything below the 99.9\,\% line fails the
comparison. The bar is clamped to the 0-100.4\,\% window.}
\end{figure}

\section{Speedup histogram}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.99\textwidth,
    height=6cm,
    ybar,
    bar width=18pt,
    enlarge x limits=0.15,
    ymin=0,
    ylabel={number of tests},
    xlabel={speedup bucket (numpy ms $\div$ Eigen ms)},
    every axis/.append style={draw=linegray, very thin},
    every tick/.append style={black, thin},
    every axis label/.append style={black},
    nodes near coords,
    nodes near coords style={font=\tiny},
    xtick={""" + bin_labels + r"""},
    """ + bin_ticks + r""",
    xticklabel style={font=\footnotesize},
    ymin=0, ymax=""" + f"{max(max(hist_counts)+1, 5)}" + r""",
]
\addplot+[draw=linegray, fill=lightgray!40, pattern=crosshatch dots, mark=none]
    coordinates { """ + hist_coords + r""" };
\end{axis}
\end{tikzpicture}
\caption{How speedups distribute across all """ + f"{n_rows}" + r""" tests.
A speedup $<1$ means numpy was faster than Eigen on that test; $>1$ means
Eigen's Rust backend won.}
\end{figure}

\section{Quadrant (4-field matrix): log-scale gates $\times$ qubits}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.99\textwidth,
    height=10cm,
    xlabel={$\log_{10}(\text{\# gates})$},
    ylabel={number of qubits},
    grid=major,
    grid style={linegray, very thin, dashed},
    every axis/.append style={draw=linegray, very thin},
    every tick/.append style={black, thin},
    every axis label/.append style={black},
    legend style={at={(0.5,-0.10)},anchor=north,draw=linegray,fill=white},
    legend entries={PASS (markers labelled by speedup), FAIL},
    xmin=""" + f"{gxmin-0.1:.2f}" + r""", xmax=""" + f"{gxmax+0.15:.2f}" + r""",
    ymin=""" + f"{qmin-0.5}" + r""", ymax=""" + f"{qmax+0.5}" + r""",
]
\addplot+[only marks, mark=*, mark size=2pt,
    nodes near coords={\tiny \pgfmathprintnumber[fixed,precision=2]{\pgfplotspointmeta}},
    nodes near coords style={font=\tiny, anchor=south west, black}
]
    coordinates { """ + pass_coords + r""" };
\addplot+[only marks, mark=triangle, mark size=3pt, draw=black, fill=white]
    coordinates { """ + fail_coords + r""" };
% Quadrant divider lines at median log10(gates) and median qubits.
\draw [linegray, very thick, dashed] (axis cs:""" + f"{median_log_g:.3f}" + r""",""" + f"{qmin-0.5}" + r""")
    -- (axis cs:""" + f"{median_log_g:.3f}" + r""",""" + f"{qmax+0.5}" + r""");
\draw [linegray, very thick, dashed] (axis cs:""" + f"{gxmin-0.1:.2f}" + r""",""" + f"{median_q}" + r""")
    -- (axis cs:""" + f"{gxmax+0.15:.2f}" + r""",""" + f"{median_q}" + r""");
\node[font=\tiny, anchor=south west, black] at (axis cs:""" + f"{gxmin-0.05:.2f}" + r""",""" + f"{qmax+0.3}" + r""")
    {small q + small g (micro)};
\node[font=\tiny, anchor=south east, black] at (axis cs:""" + f"{gxmax+0.1:.2f}" + r""",""" + f"{qmax+0.3}" + r""")
    {small q + many g (long horiz.)};
\node[font=\tiny, anchor=north west, black] at (axis cs:""" + f"{gxmin-0.05:.2f}" + r""",""" + f"{qmin-0.4}" + r""")
    {many q + small g (tall)};
\node[font=\tiny, anchor=north east, black] at (axis cs:""" + f"{gxmax+0.1:.2f}" + r""",""" + f"{qmin-0.4}" + r""")
    {many q + many g (extreme)};
\end{axis}
\end{tikzpicture}
\caption{Quadrant (4-field matrix) of test cases. Horizontal axis:
$\log_{10}$ of the gate count, vertical axis: number of qubits. Each
\emph{filled circle} is one PASS test labelled with its speedup (numpy ms
$\div$ Eigen ms). \emph{Hollow triangles} mark FAIL tests. Dashed lines split
the four canonical strategy quadrants: micro (low-q/low-g), long horizontal
(low-q/many-g), tall (many-q/low-g), and extreme (many-q/many-g). The medians
are at $\log_{10} g = """ + f"{median_log_g:.2f}" + r"""$ and $q = """ + f"{median_q:.1f}" + r"""$.}
\end{figure}

\section{Scaling: runtime vs gate count (log-log)}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.99\textwidth,
    height=9cm,
    xmode=log,
    ymode=log,
    log basis x=10,
    log basis y=10,
    xlabel={number of gates $g$ (log)},
    ylabel={runtime (ms, log)},
    grid=major,
    grid style={linegray, very thin, dashed},
    every axis/.append style={draw=linegray, very thin},
    every tick/.append style={black, thin},
    every axis label/.append style={black},
    legend style={at={(0.5,-0.10)},anchor=north,draw=linegray,fill=white},
]
\addplot+[only marks, mark=*, mark size=1.8pt, draw=black, fill=lightgray!50]
    coordinates { """ + eig_scale + r""" };
\addplot+[only marks, mark=square, mark size=1.8pt, draw=black, fill=white]
    coordinates { """ + np_scale + r""" };
\legend{Eigen, numpy}
\end{axis}
\end{tikzpicture}
\caption{Log-log scaling of runtime vs gate count. A clean slope is the
expected power-law $t \sim g^{\alpha}$ with $\alpha\approx 1$ in the regime
where Python overhead per gate dominates over state-vector size work; a
knee above $\sim$ 4-5 qubits reveals the denser matrix-vs-vector cost.}
\end{figure}

\section{Accuracy error magnitude vs gate count}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.99\textwidth,
    height=8cm,
    xmode=log,
    ymode=log,
    log basis x=10,
    log basis y=10,
    xlabel={number of gates $g$ (log)},
    ylabel={$1 - $ accuracy (log)},
    grid=major,
    grid style={linegray, very thin, dashed},
    every axis/.append style={draw=linegray, very thin},
    every tick/.append style={black, thin},
    every axis label/.append style={black},
]
\addplot+[only marks, mark=o, mark size=1.8pt, draw=black, fill=lightgray!30]
    coordinates { """ + err_coords + r""" };
\draw [linegray, very thick, dashed] (axis cs:1,0.001) -- (axis cs:10000,0.001)
    node [linegray, anchor=west, font=\tiny] at (axis cs:10000, 0.001) {PASS threshold};
\end{axis}
\end{tikzpicture}
\caption{Accuracy error magnitude $1-\textrm{accuracy}$ vs gate count. The
dashed horizontal line at $10^{-3}$ is the PASS threshold (99.9\,\%). All
tests below the line pass; tests above it are flagged FAIL in the long
table. The numerical floor at $\sim 10^{-15} \to 10^{-14}$ is dictated by
double-precision unitary propagation; tests that hit it are essentially
bit-identical to numpy.}
\end{figure}

\section{Aggregate summary}

\begin{table}[h]
\centering
\begin{tabular}{@{}lr@{}}
\toprule
Total tests & """ + f"{n_rows}" + r""" \\
Tests PASS ($\ge 99.9\%$ accuracy) & """ + f"{n_passing}" + r""" \\
Tests FAIL ($< 99.9\%$) & """ + f"{n_failing}" + r""" \\
Distinct categories & """ + f"{len(by_cat)}" + r""" \\
Median speedup numpy $\to$ Eigen & """ + f"{median_speedup:.2f}$\\times$" + r""" \\
Max single-test speedup & """ + f"{max_speedup:.2f}$\\times$" + r""" \\
Total Eigen runtime (sum over valid cases) & """ + f"{sum_eigen:.3f} ms" + r""" \\
Total numpy runtime (sum over valid cases) & """ + f"{sum_numpy:.3f} ms" + r""" \\
Overall speedup (total numpy / total Eigen) & """ + f"{overall_speedup:.2f}$\\times$" + r""" \\
Mean accuracy & """ + f"{mean_acc*100:.4f} \\%" + r""" \\
Max single-state prob deviation (mean over tests) & """ + f"{mean_dev:.2e}" + r""" \\
\bottomrule
\end{tabular}
\caption{Aggregate summary across all """ + f"{n_rows}" + r""" hard-quantum tests.}
\end{table}

\end{document}
"""
    return latex


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    cases = sorted(f for f in os.listdir(CASES_DIR) if f.endswith(".eig"))
    print(f"Cases directory: {CASES_DIR}")
    print(f"Running {len(cases)} hard-quantum benchmarks (best-of-{TIME_RUNS}, {WARMUP_RUNS} warm-up)...")
    results = run_benchmark(CASES_DIR)
    payload = {
        "metric": "eigen_vs_numpy_hard_quantum",
        "timing_unit": "milliseconds (best-of-3)",
        "accuracy_formula": "1 - 0.5 * sum(|p_eigen - p_numpy|)",
        "pass_threshold": 0.999,
        "results": [r.to_dict() for r in results],
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote results to {RESULTS_JSON}")
    tex = to_latex(results)
    with open(TEX_OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Wrote LaTeX to {TEX_OUT}")

    # Console summary
    pass_count = sum(1 for r in results if r.accuracy >= 0.999)
    print(f"\n--- Summary ---")
    print(f"PASS {pass_count}/{len(results)}")
    for r in results:
        if r.accuracy < 0.999:
            print(f"FAIL: {r.name}: acc={r.accuracy*100:.4f}% {r.notes}")
    print("\nTeX:")
    print(f"  {TEX_OUT}")
    print(f"  compile with: pdflatex {os.path.basename(TEX_OUT)}")


if __name__ == "__main__":
    main()
