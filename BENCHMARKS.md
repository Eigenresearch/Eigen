# Eigen Benchmarks

Eigen ships two benchmark suites plus a regression-tracking infrastructure
module. This document describes how to run them, how to interpret the
output, and known limitations.

---

## 1. Benchmark scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `benchmarks/run_benchmarks.py` | Core 5-workload suite (arithmetic, fibonacci, Bell, array, gate chain). | `results/benchmark_raw.csv`, `results/benchmark_summary.csv` |
| `benchmarks/run_expanded_benchmarks.py` | Expanded 12-workload suite (6 classical + 9 quantum) at 2-3 sizes each. | `results/benchmark_raw.csv`, `results/benchmark_summary.csv`, plus stdout correctness check. |
| `src/benchmark_infrastructure.py` | Regression-tracking pipeline (`BenchmarkRun`, `BenchmarkHistory`, `RegressionReport`). | JSON history + CI summary text. |
| `benchmarks/aot_vs_vm.py` | AOT-compiled binary vs VM execution comparison. | stdout. |
| `benchmarks/env_fingerprint.py` | Records the CPU/OS/Python/NumPy/native fingerprint for reproducibility. | `results/env_fingerprint.json`. |
| `benchmarks/quantum_hard/` | Hard quantum workloads (QFT, phase estimation, quantum walks). | stdout. |

Each `.eig` benchmark under `benchmarks/` (bell.eig, ghz.eig, grover.eig,
teleportation.eig, factorial.eig, fibonacci.eig, arrays.eig, maps.eig) is a
standalone Eigen program; running it through `eigen run` produces a single
timing. Aggregated timings across all of them are persisted to
`benchmarks/results.json`.

---

## 2. How to run benchmarks

### From the CLI

```bash
# Run the expanded suite (writes results/ CSVs + prints correctness table)
python benchmarks/run_expanded_benchmarks.py

# Run only the core 5-workload suite
python benchmarks/run_benchmarks.py

# Run a single .eig benchmark
eigen run benchmarks/bell.eig

# Generate an HTML dashboard (if supported by your Eigen build)
eigen bench --html
```

### From Python

```python
from benchmarks.run_expanded_benchmarks import run_benchmarks, correctness_check

correctness_check()   # prints the correctness table
run_benchmarks()       # writes results/benchmark_raw.csv and benchmark_summary.csv
```

The expanded suite runs `TRIALS = 10` trials per `(workload, size, impl)`
configuration by default. Each trial is timed with `time.perf_counter()`.

---

## 3. Workloads

### Classical (Eigen VM vs CPython)

| Workload | Sizes | What it tests |
|----------|-------|---------------|
| `arithmetic_sum` | 100, 1000, 10000, 100000 | `LOAD/STORE/ADD/JMP` loop throughput. |
| `fibonacci` | 10, 100, 1000, 10000 | Iterative `CALL/RET` and tight loop. |
| `factorial` | 10, 50, 100 | `MUL` accumulator loop. |
| `nested_loop` | 10, 50, 100 | Nested `JMP` + `MUL`/`ADD`. |
| `string_concat` | 100, 1000, 10000 | String allocation and concatenation. |

### Quantum (Eigen Simulator vs Python+NumPy)

| Workload | Sizes | What it tests |
|----------|-------|---------------|
| `bell_state` | 100, 1000, 10000 shots | `Q_ALLOC/H/CNOT/MEASURE` round-trip. |
| `gate_chain` | 100, 1000, 10000 gates | `Q_GATE` throughput on a single qubit. |
| `ghz_state` | 2, 3, 4 qubits | Multi-qubit entanglement depth. |
| `random_clifford` | 100, 1000 gates | Random Clifford circuit dispatch. |
| `multi_measure` | 2, 3, 4 qubits | End-to-end measurement pipeline. |
| `entangle_chain` | 2, 3, 4, 5 qubits | Linear CNOT chain (norm ≈ 1). |
| `dense_gate_apply` | 100, 1000, 10000 gates | Pure `apply_1qubit_gate` kernel. |

### Hard quantum workloads

`benchmarks/quantum_hard/cases/` contains larger circuits including QFT,
phase estimation, and quantum walks. Each is a standalone `.eig` file
intended for ad-hoc timing under `eigen run`.

---

## 4. Output format

### `results/benchmark_raw.csv`

One row per trial:

```
workload,size,implementation,trial,elapsed_s,result
arithmetic_sum,100,eigen_vm,1,0.000512,...
arithmetic_sum,100,python,1,0.000031,...
...
```

### `results/benchmark_summary.csv`

One row per `(workload, size, implementation)`:

```
workload,size,implementation,mean_s,std_s,min_s,max_s,ci95_s,trials
```

`ci95_s = 1.96 * std / sqrt(n)` — the 95% confidence interval half-width
on the mean.

### Correctness check

`run_expanded_benchmarks.correctness_check()` prints a table that compares
Eigen VM and CPython outputs against known-good values (e.g. `sum(1..100) =
5050`, `fib(10) = 55`, `factorial(5) = 120`, Bell state always returns
correlated bits). The run aborts with `Overall: SOME FAILED` if any check
fails.

### `benchmarks/results.json`

Aggregated single-shot timings for the `.eig` files in `benchmarks/`. This
is a quick smoke-test artifact, not a statistically rigorous measurement.

---

## 5. Interpretation

### Quantum workloads — Eigen VM is 3.3–5.2× faster than Python+NumPy

This advantage comes from the Rust `eigen_native.RustStatevector` kernel
(used when available) and the in-place, cached-index gate application in
`PythonDenseStatevector`. The advantage grows with shot count and gate
count because the per-shot VM setup cost is amortized.

### Classical workloads — CPython is 45–250× faster

The Eigen VM is a bytecode interpreter written in Python. For pure
classical arithmetic it cannot beat CPython's bytecode loop, which is
implemented in C and uses frame-level fast-paths the Eigen VM replicates
only partially. The VM exists to host quantum operations; classical code
is the supporting cast.

### Confidence intervals

`ci95_s` is the 95% CI half-width on the mean. When comparing two
implementations, treat differences smaller than the larger CI as
indistinguishable from noise. With `TRIALS = 10` and the default
workloads, CIs are typically 5–15% of the mean.

### Speedup column

The README headline number (`3.3×–5.2×` for quantum, `0.008×–0.022×` for
classical) is `python_time / eigen_time`. A number >1 means Eigen is
faster; <1 means CPython is faster.

---

## 6. Regression tracking

`src/benchmark_infrastructure.py` provides a small dataclass-based pipeline:

```python
from src.benchmark_infrastructure import (
    BenchmarkRun, BenchmarkHistory, compare_against_baseline,
    format_ci_summary,)

baseline = BenchmarkHistory.load("results/baseline.json")
current  = BenchmarkHistory()
current.add(BenchmarkRun("bell_state", "2.8.0", duration_ms=120.5))
current.add(BenchmarkRun("gate_chain", "2.8.0", duration_ms=2.4))

report = compare_against_baseline(baseline, current)
print(format_ci_summary(report))
```

A benchmark is flagged as a **regression** when its current `duration_ms`
exceeds `110%` of the baseline's `duration_ms` (the §9.2 ">10%" threshold).
The CI summary lists each benchmark with `REGRESSION` / `OK` / `NEW` /
`IMPROVED` tags.

`env_fingerprint.py` captures CPU model, OS, Python version, NumPy version,
and whether `eigen_native` was available — store this alongside the JSON
history so comparisons across machines are interpretable.

---

## 7. Methodology

- **Timer:** `time.perf_counter()` (monotonic, nanosecond resolution on
  Linux/macOS, ~100 ns on Windows).
- **Warmup:** the expanded suite calls `fn(size)` once before the timed
  trials to amortize module imports and JIT compilation.
- **Trials:** `TRIALS = 10` per configuration. Override by editing the
  constant in `run_expanded_benchmarks.py`.
- **Implementation column:** `eigen_vm` (Eigen VM + simulator) vs `python`
  (CPython reference implementation of the same workload).
- **Environment:** the README's headline numbers were measured on Intel
  Core i5-10400F @ 2.90 GHz, 12 cores, 13.9 GB RAM, Python 3.13.11,
  NumPy 2.5.0, with `eigen_native` (Rust) built in release mode.

---

## 8. Known limitations

- **Native extension availability.** Without `eigen_native`, quantum
  speedups collapse to roughly 1× because the Python dense simulator and
  CPython reference use the same NumPy primitives. Always build the Rust
  extension (`cd native/rust && maturin develop --release`) before
  benchmarking.
- **Classical workloads are not representative of VM optimization targets.**
  Do not use them to track VM-side performance work; the VM is designed for
  quantum dispatch.
- **No statistical comparison across releases** in the default CSV output.
  Use `BenchmarkHistory.compare_against_baseline` for that.
- **`string_concat` is allocation-bound** and highly sensitive to GC
  pressure; expect high variance.
- **`dense_gate_apply` bypasses the VM** and calls
  `QuantumSimulator.apply_1qubit_gate` directly. It measures the simulator
  kernel, not the VM dispatch overhead.
- **CI variance.** Shared CI runners exhibit 20–40% timing noise; the 10%
  regression threshold should be treated as a soft signal, not a hard
  gate. Re-run locally to confirm.
- **Windows timer granularity.** On Windows, `perf_counter` has ~100 ns
  granularity; sub-microsecond workloads (`gate_chain` at 100 gates) have
  high relative uncertainty. Increase `TRIALS` or aggregate over more
  shots.
- **No memory benchmarks.** The suite measures wall-clock time only; it
  does not track RSS or allocator pressure. Use `memray` or `tracemalloc`
  separately if memory is a concern.
