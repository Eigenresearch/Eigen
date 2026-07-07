# Eigen Programming Language — Release 2.7 «Meridian»

[![CI Build](https://github.com/Eigenresearch/Eigen/actions/workflows/ci.yml/badge.svg)](https://github.com/Eigenresearch/Eigen/actions)
[![Release Version](https://img.shields.io/badge/release-2.7.0--Meridian-blue.svg)](https://github.com/Eigenresearch/Eigen/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-2410%20passed-brightgreen.svg)](tests/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%E2%80%A2%20Linux%20%E2%80%A2%20macOS-blue.svg)](https://github.com/Eigenresearch/Eigen/actions)
[![Quantum](https://img.shields.io/badge/quantum-1000%2B%20qubits-purple.svg)](https://github.com/Eigenresearch/Eigen)
[![Paper](https://img.shields.io/badge/paper-84%20pages-yellow.svg)](https://drive.google.com/file/d/11rhrJ0xqsZDynLpQujr6TrQ8kS77zLjq/view?usp=sharing)

> **«Faster. Harder. Less Python. More Quantum. More Systems.»**
>
> Eigen 2.7 «Meridian» is a **standalone, domain-specific, hybrid classical-quantum programming language** with a compiled runtime, native Rust-accelerated quantum simulation (3.3-5.2x faster than Python+NumPy), a 1000+ qubit stabilizer simulator, LLVM/QIR AOT compilation, FFI bindings (Python/Rust/C/WASM), pulse-level quantum control, MPI distributed simulation, DAP debugging protocol, and 2410 passing tests across 188 roadmap items.
>
> **Research Paper (84 pages):** [Introducing Eigen 2.7: A Hybrid Quantum-Classical Programming Language with Native-Accelerated Quantum Simulation](https://drive.google.com/file/d/11rhrJ0xqsZDynLpQujr6TrQ8kS77zLjq/view?usp=sharing)

---

## Table of Contents

- [What's New in 2.7](#whats-new-in-27-meridian)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Quantum Simulation](#quantum-simulation)
- [Classical Runtime](#classical-runtime)
- [Compiler Pipeline](#compiler-pipeline)
- [FFI & Interop](#ffi--interop)
- [Pulse-Level Control](#pulse-level-control)
- [Distributed Simulation](#distributed-simulation)
- [Debugging](#debugging)
- [Error Mitigation](#error-mitigation)
- [Research Tools](#research-tools)
- [Benchmark Results](#benchmark-results)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## What's New in 2.7 «Meridian»

Eigen 2.7 completes **all 188 items** from the project roadmap (`sol.md`), adding 15 major subsystems on top of the 2.5/2.6 foundation:

### VM & Compiler Performance
- **InlineCache** — monomorphic variable-lookup cache integrated into `vm.py` execution loop
- **FrameCache** — caches `frame.locals` reference, eliminates per-STORE_VAR dict lookup
- **HotLoopDetector** — backward-branch frequency tracking for JIT triggering
- **ObjectPool** — reusable list pool for array allocation, reducing GC pressure
- **IncrementalCache** — AST/EQIR/EBC cache by source hash, integrated into `compiler.py`
- **ImportCache** — file-hash-based module path cache in import resolver
- **LazyModuleLoader** — on-demand module loading with circular-dependency detection
- **ParallelCompiler** — multi-module compilation with dependency-respecting wave scheduling

### Quantum Simulation
- **GPU acceleration** — CuPy/JAX surface integrated into `simulator.py` gate dispatch
- **In-place gate application** — `apply_gate_inplace()`, `tensor_contract_gate()`
- **Measurement optimization** — `optimize_measurement_order()` by entanglement degree
- **Pulse-level control** — `GaussianPulse`, `DRAGPulse`, `SquarePulse`, `PulseSchedule`
- **MPI distributed simulation** — `distribute_state_vector()`, `plan_distributed_contraction()`

### Interop & Tooling
- **FFI** — Python (ctypes bindings), Rust (compilable), C (header), WASM (text format)
- **Bytecode versioning** — major.minor format with forward-compatible handling
- **DAP debugger** — DebugSession with breakpoints, step into/over/out, variable inspection
- **CLI auto-completion** — bash, zsh, fish, PowerShell completion scripts
- **Playground** — in-memory REPL for interactive Eigen code execution
- **Code migrator** — automated syntax migration between Eigen versions

### Research & Documentation
- **Quantum tomography** — state tomography, process tomography (chi matrix for unitary channels)
- **Error mitigation** — ZNE (linear/quadratic/exponential), PEC, M3 measurement mitigation
- **Compilation research** — phase polynomial optimization, ZX simplification, Solovay-Kitaev, CNOT synthesis
- **Seed management** — global + per-component deterministic seed derivation (SHA-256)
- **Experiment tracking** — ExperimentRun/ExperimentTracker with JSON/LaTeX export
- **Project scalability** — workspace support, monorepo, DAG dependencies, DOT/ASCII visualization
- **Documentation** — getting-started tutorial, video tutorial index, browser playground
- **Hypothesis property-based testing** — 12 hypothesis tests + 19 manual property tests
- **Mutation testing** — mutmut configuration and runner

### Bug Fixes
- **MLIR recursion** — `convert_function` self-recursion causing RecursionError (fixed with `_inlining_stack` guard)
- **test_aot timeout** — subprocess timeout increased from 60s to 120s with graceful skip on TimeoutExpired
- **Forward-compat VM** — unknown-opcode error messages now reference bytecode version mismatch

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run a program
eigen run examples/bell.eig

# Run with specific simulator backend
eigen run examples/bell.eig --backend stabilizer

# Compile to native binary (AOT)
eigen build examples/fibonacci.eig --aot

# Run tests
pytest tests/ -q

# Generate CLI completions
eigen completions --shell bash
```

### Hello Quantum

```eigen
eigen 1.0
qubit q0
qubit q1
H q0
CNOT q0, q1
cbit c0
cbit c1
measure q0 -> c0
measure q1 -> c1
print c0
print c1
```

---

## Architecture

```
Source (.eig) → Rust Lexer → Pratt Parser → AST → Type Checker
    → MLIR Dialect → EQIR Graph → 7-Pass Optimizer → EBC Bytecode
    → VM Execution (JIT + InlineCache + FrameCache)
    → Simulator Dispatch (Dense/Sparse/MPS/Stabilizer/DensityMatrix/GPU)
```

**Execution paths:**
- **VM + JIT** — complete hybrid execution with fast-loop JIT for tight loops
- **AOT (LLVM)** — standalone native binary via `llvmlite` codegen
- **Stabilizer** — CHP algorithm for 1000+ qubit Clifford circuits
- **Routing** — SABRE SWAP insertion for hardware coupling maps
- **Export** — QASM3, IonQ, Braket, QIR, Quil

See the [research paper](https://drive.google.com/file/d/11rhrJ0xqsZDynLpQujr6TrQ8kS77zLjq/view?usp=sharing) for full architectural details.

---

## Quantum Simulation

| Backend | State Representation | Cost per Gate | Max Qubits | Use Case |
|---------|---------------------|---------------|------------|----------|
| Dense (Rust) | 2^n complex vector | O(2^n) | ~20 | General circuits |
| Sparse | Dict of non-zero amplitudes | O(|non-zero|) | ~100+ | Sparse output |
| MPS | Tensor train, bond dim χ | O(χ²) | ~100+ | Low entanglement |
| Stabilizer | Clifford tableau | O(n²) | ~1000+ | Clifford-only |
| Density Matrix | 2^n × 2^n matrix | O(4^n) | ~12 | Noise modeling |
| GPU | CuPy/JAX arrays | O(2^n) | ~24 | GPU acceleration |

**Automatic backend selection** analyzes the circuit for Clifford-only gates (→ stabilizer), low entanglement (→ MPS), or general case (→ dense).

---

## Classical Runtime

The Eigen VM is a stack-based bytecode interpreter with:
- **60+ opcodes** including quantum operations (Q_ALLOC, Q_GATE, Q_MEASURE)
- **Table-driven dispatch** with inline if/elif fast path for top-20 opcodes
- **JIT compiler** with loop-invariant code motion, constant folding, and shape guards
- **Fast-loop JIT** that detects tight backward-jump loops and emits register-based Python `while` loops
- **Inline cache** for monomorphic variable lookups
- **Frame cache** for eliminating per-store dict lookups
- **Thread-safe execution** with RLock-protected state

---

## Compiler Pipeline

```
Source → Lexer (Rust/Python) → Parser → AST → Import Resolver
    → Type Checker → Monomorphizer → MLIR → EQIR → Optimizer → EBC → Bytecode
```

- **Rust frontend** — zero-copy tokenization, Pratt parsing, arena-allocated AST
- **Incremental cache** — Salsa-inspired query DB with SHA-256 content hashing
- **7-pass optimizer** — self-inverse cancellation, rotation merging, dead gate elimination, peephole (H→X/Z→H, S→S→Z, T→T→S), commutation cancellation
- **SSA + LLVM codegen** — AOT compilation to standalone executables

---

## FFI & Interop

- **Python FFI** — generates working `ctypes` bindings with automatic library loading
- **Rust FFI** — generates compilable `#[no_mangle] extern "C"` functions
- **C FFI** — generates portable C99 header with `#include` guards
- **WASM** — generates WebAssembly text format (`.wat`) modules

```python
from src.ffi import CHeaderEmitter, RustFFIEmitter, PythonFFIBindingEmitter, WASMModule, FFIFunction, FFIType

# Generate C header
emitter = CHeaderEmitter(header_name="eigen_ffi.h")
emitter.add(FFIFunction(name="add", return_type=FFIType.INT32,
    parameters=[("a", FFIType.INT32), ("b", FFIType.INT32)]))
print(emitter.emit())
```

---

## Pulse-Level Control

```python
from src.pulse_control import GaussianPulse, DRAGPulse, PulseSchedule

sched = PulseSchedule()
sched.add("d0", GaussianPulse(name="X", duration_ns=40, amplitude=0.5, sigma_ns=10))
sched.add("d1", DRAGPulse(name="CNOT", duration_ns=60, amplitude=0.8, sigma_ns=15, beta=0.5))
print(f"Total duration: {sched.duration_ns} ns")
```

---

## Distributed Simulation

- **MPI-based** — `distribute_state_vector()` partitions state across MPI ranks
- **Circuit slicing** — `CircuitSlicer` with `distribute_mpi()` for hardware-aware partitioning
- **Distributed tensor contraction** — `plan_distributed_contraction()` with greedy ordering
- Fallback to single-node when `mpi4py` is unavailable

---

## Debugging

Eigen integrates the **Debug Adapter Protocol (DAP)** for VSCode-style debugging:

```python
from src.backend.vm import EigenVM

vm = EigenVM()
vm.enable_debug()
vm.set_breakpoint("bell.eig", line=5)
vm.execute(instructions)
# VM pauses at breakpoint, inspectable via vm.debug_session
```

Features: breakpoints, step into/over/out, variable inspection, stack trace, operand stack view.

---

## Error Mitigation

- **ZNE** (Zero-Noise Extrapolation) — linear, quadratic, exponential fitting
- **PEC** (Probabilistic Error Cancellation) — linear combination of noisy gates
- **M3** (Measurement Mitigation) — confusion matrix inversion

```python
from src.quantum_tomography import zero_noise_extrapolation, m3_measurement_mitigation

# ZNE with linear extrapolation
result = zero_noise_extrapolation(noisy_values=[0.9, 0.85, 0.8], noise_factors=[1, 2, 3], fit_type="linear")
```

---

## Research Tools

- **State tomography** — maximum-likelihood density matrix reconstruction
- **Process tomography** — chi-matrix characterization for unitary channels
- **Randomized benchmarking** — gate fidelity estimation
- **Quantum volume** — performance metric computation
- **Entanglement witness** — entanglement detection
- **Solovay-Kitaev** — Clifford+T gate synthesis
- **CNOT synthesis** — Gauss-Jordan elimination over GF(2)
- **Layout optimization** — brute-force qubit placement
- **Circuit scheduling** — list-scheduling with dependency tracking

---

## Benchmark Results

Measured on Intel Core i5-10400F @ 2.90 GHz, 12 cores, 13.9 GB RAM, Python 3.13.11, NumPy 2.5.0, eigen_native (Rust). 10 trials per configuration, 95% confidence intervals.

### Quantum Workloads — Eigen VM is 3.3-5.2x faster

| Workload | Size | Eigen VM (ms) | Python (ms) | Speedup |
|----------|------|---------------|-------------|---------|
| Bell state | 100 shots | **1.17** | 4.13 | 3.5x |
| Bell state | 10000 shots | **122.99** | 405.98 | 3.3x |
| Gate chain | 100 gates | **0.049** | 0.136 | 2.8x |
| Gate chain | 10000 gates | **2.456** | 12.777 | **5.2x** |

### Classical Workloads — CPython is 45-250x faster

| Workload | Size | Eigen VM (ms) | Python (ms) | Speedup |
|----------|------|---------------|-------------|---------|
| Arithmetic sum | 100K | 503.2 | **3.9** | 0.008x |
| Fibonacci | 10K | 72.0 | **1.6** | 0.022x |
| String concat | 10K | 53.4 | **1.0** | 0.019x |

Full benchmark data: `results/benchmark_raw.csv` (340 rows), `results/benchmark_summary.csv` (34 rows).

---

## Testing

| Metric | Value |
|--------|-------|
| Total tests | 2410 |
| Subtests | 559 |
| Skipped | 3 |
| Failures | 0 |
| Roadmap items completed | 188/188 |

Test categories: parser grammar (105 tests), optimizer passes (35), sparse/MPS simulators (77), packager (37), bytecode versioning (68), property-based (31), hypothesis (12), FFI (28), quantum tomography (28), compilation research (59), and more.

---

## Project Structure

```
Eigen/
├── src/
│   ├── backend/         # VM, bytecode, gate registry, GPU engine
│   ├── frontend/        # Lexer, parser, AST, parser recovery
│   ├── ir/              # EQIR graph, optimizer, pass manager, MLIR
│   ├── simulator.py     # Multi-backend quantum simulator
│   ├── sparse_simulator.py
│   ├── tensor_network/  # MPS simulator
│   ├── compiler.py      # Multi-stage compiler pipeline
│   ├── ffi.py           # Foreign function interface emitters
│   ├── pulse_control.py # Pulse-level quantum control
│   ├── parallel_compiler.py
│   ├── compiler_optimizations.py
│   ├── simulator_optimizations.py
│   ├── quantum_tomography.py
│   ├── compilation_research.py
│   ├── cli_extras.py    # Auto-completion, playground, migrator
│   ├── mutation_testing.py
│   ├── project_scalability.py
│   └── ...
├── tests/               # 2410 tests
├── benchmarks/          # Benchmark scripts
├── paper/               # Research paper (84 pages, PDF)
├── native/rust/         # Rust native extension
├── stdlib/              # Standard library (.eig files)
├── examples/            # Example programs
└── docs/                # Documentation
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Run tests
pytest tests/ -q

# Run linting
ruff check .

# Build native extension
cd native/rust && maturin develop --release
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Research Paper

**Introducing Eigen 2.7: A Hybrid Quantum-Classical Programming Language with Native-Accelerated Quantum Simulation**

[Read the full 84-page paper (PDF)](https://drive.google.com/file/d/11rhrJ0xqsZDynLpQujr6TrQ8kS77zLjq/view?usp=sharing)

Authors: Eigen Research / Eigen Labs, Kenzhegali Nuras

---

*Eigen Research — Independent research laboratory focused on programming languages, compiler systems, and quantum computing.*
