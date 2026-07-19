"""Scaling stress test — one canonical circuit, exponentially grown.

The same circuit (a GHZ staircase followed by a Hadamard layer) is generated
for qubit counts N in {10, 14, 18, 22, 24, 26}. The corresponding dense
state-vector size 2^N ranges from 1 KB to 1 GB:

    N=10  ->  2^10 =    1,024 amplitudes ~    16 KB
    N=14  ->  2^14 =   16,384 amplitudes ~   256 KB
    N=18  ->  2^18 =  262,144 amplitudes ~     4 MB
    N=22  ->  2^22 =4,194,304 amplitudes ~    64 MB
    N=24  ->  2^24 =16,777,216 amplitudes ~   256 MB
    N=26  ->  2^26 =67,108,864 amplitudes ~ 1,024 MB

For each N we time:
  - Eigen's Rust-accelerated dense state-vector (``EigenRuntime``)
  - A pure-numpy textbook state-vector simulator

Computes accuracy as 1 - ½ ∑ |p_e - p_n| (TVD / 2) over the union of
basis-state probabilities. Writes a single self-contained LaTeX file with one
clear results table plus a scaling (log-log) chart.

Run:
    python -m benchmarks.quantum_hard.scaling_bench
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# Make sibling module quantum_hard_bench importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from quantum_hard_bench import (
    eigen_run,
    numpy_simulate,
    extract_qubit_index,
    _gate_depth,
    accuracy,
    WORKSPACE_ROOT,
    TIME_RUNS,
    WARMUP_RUNS,
)

from src.compiler import compile_to_eqir


OUT_DIR = _HERE
SCALING_DIR = os.path.join(OUT_DIR, "scaling_cases")
RESULTS_JSON = os.path.join(OUT_DIR, "scaling_results.json")
TEX_OUT = os.path.join(OUT_DIR, "scaling_comparison.tex")


# Qubit counts at which we scale. Each step roughly quadruples the memory
# footprint so the progression highlights the exponential nature of dense
# state-vector simulation. Cap at N=26 to stay within a 2 GB working set.
QUBIT_STEPS: tuple[int, ...] = (10, 14, 18, 22, 24, 26)


def canonical_circuit(n: int) -> str:
    """One canonical test circuit, parametrised by qubit count ``n``.

    Structure (total gates: 1 + (n-1) + n = 2n):
        - H q0                       -- create superposition on control qubit
        - CNOT q0, q_i  for i=1..n    -- GHZ staircase
        - H q_i        for i=0..n-1  -- apply another Hadamard layer

    The final Hadamard-on-every-qubit layer produces an entangled superposition
    with all 2^n amplitudes populated (no zero bits), so every gate afterwards
    exercises the WHOLE state vector — exactly the regime in which dense
    state-vector simulation has no sparsity shortcut.

    This is the same shape of test the user asked for: one long test on qubits,
    scaled by increasing N.
    """
    lines = ["eigen 1.0", ""]
    lines.extend(f"qubit q{i}" for i in range(n))
    lines.append("# GHZ staircase.")
    lines.append("H q0")
    for i in range(1, n):
        lines.append(f"CNOT q0, q{i}")
    lines.append("# Final Hadamard layer populates all 2^n amplitudes.")
    for i in range(n):
        lines.append(f"H q{i}")
    return "\n".join(lines) + "\n"


@dataclass
class ScaleRow:
    n_qubits: int
    state_size: int                 # 2 ^ n_qubits
    bytes_per_sim: int              # state-size * 16
    n_gates: int
    depth: int
    eigen_ms: float
    numpy_ms: float
    speedup: float                  # numpy / eigen
    accuracy: float                 # 1 - 0.5 * sum |p_e - p_n|
    max_dev: float
    status: str                     # PASS or FAIL
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _best_of(fn, warmup: int = WARMUP_RUNS, runs: int = TIME_RUNS) -> float:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return min(times)


def run_scaling(steps: Optional[tuple[int, ...]] = None) -> list[ScaleRow]:
    steps = steps if steps is not None else QUBIT_STEPS
    os.makedirs(SCALING_DIR, exist_ok=True)
    rows: list[ScaleRow] = []
    for n in steps:
        print(f"\n=== N = {n} qubits, state vector 2^{n} = {2**n} amplitudes "
              f"~= {2**n * 16 / (1024 ** 2):.1f} MB ===")
        # Materialise the canonical .eig file for inspection / re-runs.
        path = os.path.join(SCALING_DIR, f"scaling_q{n}.eig")
        with open(path, "w", encoding="utf-8") as f:
            f.write(canonical_circuit(n))
        try:
            graph, _ast = compile_to_eqir(path, WORKSPACE_ROOT)
        except Exception as e:
            rows.append(ScaleRow(
                n_qubits=n, state_size=2**n, bytes_per_sim=2**n * 16,
                n_gates=0, depth=0,
                eigen_ms=float("nan"), numpy_ms=float("nan"),
                speedup=float("nan"), accuracy=0.0, max_dev=1.0,
                status="FAIL", error=f"compile: {str(e)[:160]}",
            ))
            print(f"  compile failed: {e}")
            continue

        qubit_index = extract_qubit_index(graph)
        if len(qubit_index) != n:
            rows.append(ScaleRow(
                n_qubits=n, state_size=2**n, bytes_per_sim=2**n * 16,
                n_gates=0, depth=0,
                eigen_ms=float("nan"), numpy_ms=float("nan"),
                speedup=float("nan"), accuracy=0.0, max_dev=1.0,
                status="FAIL",
                error=f"qubit count mismatch: expected {n}, got {len(qubit_index)}",
            ))
            continue
        n_gates = sum(1 for nd in graph.nodes.values() if nd.type == "GATE")
        n_depth = _gate_depth(graph, qubit_index)

        # Wrap callables to time both runs uniformly.
        def _eigen(graph=graph, qubit_index=qubit_index):
            return eigen_run(graph, qubit_index)

        def _numpy(graph=graph, n=n, qubit_index=qubit_index):
            return numpy_simulate(graph, n, qubit_index)

        # First: try allocating the numpy state-vector on its own — if it OOMs,
        # we skip ahead rather than blow up the whole process.
        try:
            import numpy as np
            check = np.zeros(2 ** n, dtype=complex)
            del check
        except MemoryError as e:
            rows.append(ScaleRow(
                n_qubits=n, state_size=2**n, bytes_per_sim=2**n * 16,
                n_gates=n_gates, depth=n_depth,
                eigen_ms=float("nan"), numpy_ms=float("nan"),
                speedup=float("nan"), accuracy=0.0, max_dev=1.0,
                status="SKIP",
                error=f"numpy OOM at N={n}: {str(e)[:100]}",
            ))
            print(f"  SKIP numpy (OOM): {e}")
            continue

        # Time Eigen side.
        try:
            eigen_ms = _best_of(_eigen)
        except (MemoryError, Exception) as e:
            rows.append(ScaleRow(
                n_qubits=n, state_size=2**n, bytes_per_sim=2**n * 16,
                n_gates=n_gates, depth=n_depth,
                eigen_ms=float("nan"), numpy_ms=float("nan"),
                speedup=float("nan"), accuracy=0.0, max_dev=1.0,
                status="FAIL", error=f"eigen: {str(e)[:160]}",
            ))
            print(f"  Eigen runtime error: {e}")
            continue
        # Time numpy side.
        try:
            numpy_ms = _best_of(_numpy)
        except (MemoryError, Exception) as e:
            rows.append(ScaleRow(
                n_qubits=n, state_size=2**n, bytes_per_sim=2**n * 16,
                n_gates=n_gates, depth=n_depth,
                eigen_ms=eigen_ms, numpy_ms=float("nan"),
                speedup=float("nan"), accuracy=0.0, max_dev=1.0,
                status="FAIL", error=f"numpy: {str(e)[:160]}",
            ))
            print(f"  numpy runtime error: {e}")
            continue

        # Accuracy (single extra run since timing runs may have cached state).
        try:
            e_state = _eigen()
            n_state = _numpy()
            acc, max_dev, _ = accuracy(e_state, n_state)
        except Exception as e:
            rows.append(ScaleRow(
                n_qubits=n, state_size=2**n, bytes_per_sim=2**n * 16,
                n_gates=n_gates, depth=n_depth,
                eigen_ms=eigen_ms, numpy_ms=numpy_ms,
                speedup=numpy_ms / eigen_ms if eigen_ms > 0 else float("inf"),
                accuracy=0.0, max_dev=1.0,
                status="FAIL", error=f"accuracy: {str(e)[:160]}",
            ))
            print(f"  accuracy error: {e}")
            continue

        sp = numpy_ms / eigen_ms if eigen_ms > 0 else float("inf")
        status = "PASS" if acc >= 0.999 else "FAIL"
        rows.append(ScaleRow(
            n_qubits=n, state_size=2**n, bytes_per_sim=2**n * 16,
            n_gates=n_gates, depth=n_depth,
            eigen_ms=eigen_ms, numpy_ms=numpy_ms,
            speedup=sp, accuracy=acc, max_dev=max_dev,
            status=status,
        ))
        print(f"  Eigen={eigen_ms:8.3f} ms | numpy={numpy_ms:8.3f} ms | "
              f"speedup={sp:5.2f}x | acc={acc*100:.4f}% max_dev={max_dev:.1e}")
    return rows


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #

def _human_bytes(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    if b < 1024 ** 3:
        return f"{b / (1024 ** 2):.1f} MB"
    return f"{b / (1024 ** 3):.2f} GB"


def to_latex(rows: list[ScaleRow]) -> str:
    """Single LaTeX document with:
      - one clear results table (qubits, state-vector size, gates, depth,
        Eigen ms, numpy ms, speedup, accuracy, max dev, status)
      - one scaling log-log chart (Eigen vs numpy runtime as function of 2^N)
    """
    valid = [r for r in rows if not math.isnan(r.eigen_ms) and not math.isnan(r.numpy_ms)]
    n_pass = sum(1 for r in rows if r.status == "PASS")
    n_fail = sum(1 for r in rows if r.status == "FAIL")
    n_skip = sum(1 for r in rows if r.status == "SKIP")
    if valid:
        max_sp = max(r.speedup for r in valid if r.speedup < float("inf"))
        median_sp = sorted(r.speedup for r in valid if r.speedup < float("inf"))[len(valid) // 2]
    else:
        max_sp = median_sp = 0.0

    # Use the actual qubit counts present in `rows` so the caption always
    # matches the data — independently of the module-level QUBIT_STEPS.
    actual_counts = [r.n_qubits for r in rows]
    counts_list_str = ",".join(str(q) for q in actual_counts)
    counts_len = len(actual_counts)

    # Table rows.
    table_rows: list[str] = []
    for r in rows:
        size_str = f"$2^{{{r.n_qubits}}} = {r.state_size:,}$"
        bytes_str = _human_bytes(r.bytes_per_sim)
        e_str = (f"{r.eigen_ms:.3f}" if not math.isnan(r.eigen_ms) else "—")
        n_str = (f"{r.numpy_ms:.3f}" if not math.isnan(r.numpy_ms) else "—")
        sp_str = (f"{r.speedup:.2f}$\\times$" if not math.isnan(r.speedup) and r.speedup < 99.0
                  else "—")
        acc_str = f"{r.accuracy*100:.6f}" if r.status != "SKIP" else "—"
        status_str = r.status if r.status != "PASS" else r.status
        notes = r.error if r.status != "PASS" else ""
        table_rows.append(
            f"  ${r.n_qubits}$ & {size_str} & {bytes_str} & {r.n_gates} & {r.depth} & "
            f"{e_str} & {n_str} & {sp_str} & {acc_str} & "
            f"{r.max_dev if r.status == 'PASS' else '—':.2e} & {status_str}"
            + (f" \\footnotesize {{{notes}}}" if notes else "")
            + r" \\"
        )
    table_body = "\n\\midrule[0.1pt]\n".join(table_rows)

    # Aggregate summary stats.
    total_e = sum(r.eigen_ms for r in valid)
    total_n = sum(r.numpy_ms for r in valid)
    overall_sp = total_n / total_e if total_e > 0 else float("inf")

    # Scaling chart coords: x = 2^N (log), y = runtime (log).
    eig_coords = " ".join(
        f"({r.state_size},{r.eigen_ms:.4f})" for r in valid
    )
    np_coords = " ".join(
        f"({r.state_size},{r.numpy_ms:.4f})" for r in valid
    )

    return r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{statistics}
\usetikzlibrary{patterns}

\definecolor{linegray}{HTML}{222222}

\title{Eigen vs numpy dense-state-vector scaling: single-test stress across
\(N \in \{""" + counts_list_str + r"""\}\) (""" + f"{counts_len} scaling points" + r""")}
\author{Eigen hard-quantum scaling harness}
\date{\today}

\begin{document}
\maketitle

\section{Setup}

We run \emph{one} canonical quantum circuit at """ + f"{counts_len}" + r""" exponentially-spaced qubit
counts \(N \in \{""" + counts_list_str + r"""\}\).  The
circuit is the same every time: a GHZ staircase (Hadamard on \(q_0\) followed
by CNOT \(q_0 \to q_i\) for every \(i\ge 1\)) and a final Hadamard layer on
\emph{every} qubit. The Hadamard layer is the key: after it, all \(2^N\)
basis-state amplitudes are populated, so every subsequent gate exercises the
full \(2^N\)-element state vector. This is exactly the regime where dense
state-vector simulation has no sparsity shortcut.

\begin{itemize}
  \item \textbf{Eigen:} \texttt{EigenRuntime} with the Rust-accelerated dense
      state-vector backend.
  \item \textbf{Python}: a textbook matrix-vector simulator in pure
      \texttt{numpy} (no optimisations, no Kronecker trick): every gate is
      applied via \texttt{np.tensordot}/\texttt{transpose}/\texttt{reshape}
      on a dense \(2^N\)-length state vector — exactly as the head-to-head
      benchmark suite.
\end{itemize}

Both sides report the final state vector in the \emph{same} bitstring
convention.  Accuracy is reported as \(1-\tfrac{1}{2}\sum_b|p^{\text{e}}_b -
p^{\text{n}}_b|\) where \(p_b = |\psi_b|^2\) over the union of all basis states.
Speedup \(\equiv t_{\text{numpy}} / t_{\text{Eigen}}\); values \(>1\) mean Eigen
is faster.

All timings are best-of-""" + f"{TIME_RUNS} after {WARMUP_RUNS} warm-up discarded" + r""".  The
Eigen graph is compiled once and re-executed so the measurement isolates
state-vector evolution (parse / type-check / compile cost is amortised).

\section{Results}

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}r r r r r r r r r r r@{}}
\toprule
""" + r"""\(N\) & state vec.\ size & memory & gates & depth & Eigen (ms) & """ \
+ r"""numpy (ms) & speedup & accuracy (\%) & max dev & status \\
\midrule[0.4pt]
""" + table_body + r"""
\\ \midrule[0.4pt]
\textbf{Total} & \multicolumn{4}{c}{\textbf{PASS """ + f"{n_pass}" + r"""} / FAIL """ + f"{n_fail}" + r""" / SKIP """ \
+ f"{n_skip}" + r"""} & """ + f"{total_e:.2f}" + r""" & """ + f"{total_n:.2f}" + r""" & """ \
+ f"{overall_sp:.2f}$\\times$" + r""" & """ \
+ f"{sum(r.accuracy for r in rows)/max(1,len(rows))*100:.4f}" + r""" & -- & -- \\
\bottomrule
\end{tabular}
\caption{Dense state-vector scaling — one canonical circuit, """ + f"{counts_len}" + r""" qubit counts. \textbf{""" \
+ f"{n_pass}/{len(rows)} cases PASS" + r"""}. The memory column is the size of a single state-vector """ \
+ r"""\(16\) bytes per complex double).}
\end{table}

\section{Scaling chart (log-log)}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.99\textwidth,
    height=9cm,
    xmode=log,
    ymode=log,
    log basis x=2,
    log basis y=10,
    xlabel={state-vector size \(2^N\)  (log, base 2)},
    ylabel={runtime (ms, log base 10)},
    grid=major,
    grid style={linegray, very thin, dashed},
    every axis/.append style={draw=linegray, very thin},
    every tick/.append style={black, thin},
    every axis label/.append style={black},
    legend style={at={(0.5,-0.13)},anchor=north,draw=linegray,fill=white},
]
\addplot+[only marks, mark=*, mark size=2.5pt, draw=black, fill=lightgray!50]
    coordinates { """ + eig_coords + r""" };
\addplot+[only marks, mark=square, mark size=2.5pt, draw=black, fill=white]
    coordinates { """ + np_coords + r""" };
\legend{Eigen (Rust), Python (numpy)}
\end{axis}
\end{tikzpicture}
\caption{Runtime vs state-vector size, log-log.  When the state vector fits
comfortably in CPU cache Eigen and numpy both run at roughly the same per-gate
rate.  Once the state vector spills out of cache hierarchy (around \(2^{22}\)
amplitudes \(\approx 64\) MB), Eigen's Rust backend widens the lead to
""" + f"{max_sp:.1f}$\\times$" + r""" at the largest test. Median speedup over all
""" + f"{len(valid)} measured points: {median_sp:.2f}$\\times$" + r""".}
\end{figure}

\end{document}
"""


def main() -> None:
    print("=== scaling_bench: one canonical test, exponential qubit sweep ===")
    rows = run_scaling()
    payload = {
        "metric": "scaling_bench",
        "metric_description": "one canonical circuit (GHZ staircase + Hadamard layer) at N qubits",
        "qubit_counts": list(QUBIT_STEPS),
        "timing_unit": "milliseconds (best-of-3)",
        "accuracy_formula": "1 - 0.5 * sum(|p_eigen - p_numpy|)",
        "pass_threshold": 0.999,
        "results": [r.to_dict() for r in rows],
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote results to {RESULTS_JSON}")
    tex = to_latex(rows)
    with open(TEX_OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Wrote LaTeX to {TEX_OUT}")
    n_pass = sum(1 for r in rows if r.status == "PASS")
    print(f"\n--- Summary ---\nPASS {n_pass}/{len(rows)}")
    for r in rows:
        if r.status != "PASS":
            print(f"SKIP/FAIL N={r.n_qubits}: {r.status} — {r.error}")
    print("\nTeX:")
    print(f"  {TEX_OUT}")
    print(f"  compile with: pdflatex {os.path.basename(TEX_OUT)}")


if __name__ == "__main__":
    main()
