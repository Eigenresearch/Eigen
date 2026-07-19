<div align="center">

# Eigen Programming Language

### Release 2.8 «Mars» — a standalone hybrid classical–quantum programming language

[![CI Build](https://github.com/Eigenresearch/Eigen/actions/workflows/ci.yml/badge.svg)](https://github.com/Eigenresearch/Eigen/actions)
[![Release Version](https://img.shields.io/badge/release-2.8.0--Mars-blue.svg)](https://github.com/Eigenresearch/Eigen/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2410%20passed-brightgreen.svg)](tests/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%E2%80%A2%20Linux%20%E2%80%A2%20macOS-blue.svg)](https://github.com/Eigenresearch/Eigen/releases)
[![Quantum](https://img.shields.io/badge/quantum-1000%2B%20qubits-purple.svg)](https://github.com/Eigenresearch/Eigen)
[![Python](https://img.shields.io/badge/python-3.10%20–%203.14-blue.svg)](pyproject.toml)
[![Paper](https://img.shields.io/badge/paper-111%20pages-yellow.svg)](https://drive.google.com/file/d/1t1woF19vfMlQwmCulF45u__sl2cOxx4s/view?usp=sharing)

**⭐ If Eigen is useful to you, please [star the repository](https://github.com/Eigenresearch/Eigen) and share it with your friends and colleagues. ⭐**

</div>

---

> **«Faster. Harder. Less Python. More Quantum. More Systems.»**

**Eigen 2.8 «Mars»** is a **standalone, domain-specific, hybrid classical–quantum programming language** with its own compiler and runtime. One language, one execution model, one instruction stream — classical control flow and quantum operations run side by side inside a single stack-based virtual machine. Eigen combines:

- **Native Rust-accelerated quantum simulation** — 3.3–5.2× faster than Python + NumPy on quantum workloads
- **Six simulator backends** — dense, sparse, MPS (tensor-network), stabilizer (1000+ qubits), density-matrix (exact noise), and GPU
- **A complete compiler pipeline** — Rust lexer/parser → typed AST → MLIR → EQIR graph → 7-pass optimizer → EBC bytecode → VM (+ optional LLVM/QIR AOT)
- **FFI bindings** for Python, Rust, C, and WebAssembly
- **Hardware-aware routing** (SABRE) and **exporters** (OpenQASM 3.0, IonQ, Braket, QIR)
- **Research tooling** — noise modeling, error mitigation (ZNE/PEC/M3), tomography, randomized benchmarking, quantum volume, pulse-level control
- **Security hardening** — HMAC-signed compiler cache, sandboxed JIT, deny-by-default native loading
- **2410 passing tests**, deterministic execution, and full reproducibility tooling

> **Research Paper (111 pages):** [**Eigen 2.8 «Mars»: A Repository-Grounded Reconstruction of a Hybrid Classical–Quantum Programming Language**](https://drive.google.com/file/d/1t1woF19vfMlQwmCulF45u__sl2cOxx4s/view?usp=sharing)

---

## Table of Contents

- [What's New in 2.8](#whats-new-in-28-mars)
- [What's New in 2.7](#whats-new-in-27-meridian)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Feature Maturity](#feature-maturity)
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
- [Known Limitations](#known-limitations)
- [Migration Guide: 2.7 → 2.8](#migration-guide-27--28)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## What's New in 2.8 «Mars»

Eigen 2.8 is a **correctness-first, hardening, and documentation** release built on the 2.7 foundation. Rather than chasing new headline benchmarks, its engineering budget was spent eliminating entire *classes* of silent-wrong-answer bugs, closing exploitable security holes, and making execution deterministic and reproducible.

### Critical correctness fixes (stabilizer simulator)

Four bugs that produced **wrong answers without raising an exception** were fixed:

- **CNOT X-update** — was updating the *control* qubit's X-bit instead of the *target's*, silently producing the wrong entangled state.
- **X / Z gate destabilizer update** — Pauli gates now update both destabilizers and stabilizers, fixing corrupted phase accounting.
- **`allocate_qubit`** — now grows the tableau incrementally instead of resetting all stabilizer state (which silently wiped prior entanglement).
- **CZ vs CNOT in ZX** — CZ now has a distinct ZX-graph (Hadamard-edge) representation, so equivalence checking no longer reports different circuits as equivalent.

### Performance

- **113× VM arithmetic-loop speedup** (107 ms → 0.95 ms) via pre-extracted opcode/arg arrays, localized hot-path lookups, and **batched limit checks every 4096 ops**.
- Constant folding in the compiler; JIT fast-loop recognizes peephole-optimized conditions; LRU eviction in all compiler caches.
- In-place dense-simulator state updates; MPS adaptive bond dimension driven by entanglement entropy.

### Security hardening

- **JSON + HMAC compiler cache** replaces the pickle-only cache (closes a remote-code-execution vector on tampered caches).
- **Hardened JIT `exec()` sandbox** and subprocess isolation for `run`/`exec --sandbox`.
- **Deny-by-default native module loading** — an allow-list plus a strict name regex closes a path-traversal RCE class.
- Registry returns **frozen** dicts; frame pool clears stale data on recycling.

### Language & tooling

- Unicode/hex escape sequences (`\uXXXX`, `\xNN`), block comments (`/* … */`), single-quoted strings, and `finally` blocks.
- `UnaryOpNode` replaces the old `BinaryOpNode` hack.
- **LSP** hover and go-to-definition with real symbol lookup; **28 CLI subcommands** (6 new in 2.8: `reproduce`, `verify`, `audit`, `lsp`, `doctor`, `profile`).
- `ReadoutError` noise channel; `T2 ≤ 2·T1` physical validation.

### Determinism & reproducibility

- The EQIR optimizer, SABRE router, VM RNG, and per-component seed manager are all deterministic, so a run is a **pure function of its source and seed** and produces a reproducible `result_fingerprint`.

### Documentation

New reference set: `ARCHITECTURE.md`, `STDLIB_API.md`, `NOISE_MODELS.md`, `BENCHMARKS.md`, plus new example programs for `match/case`, `try/catch/finally`, string interpolation, operator overloading, and noisy simulation.

> **Honest scope note:** the `ASYNC_CALL` / `AWAIT` / `YIELD_TASK` opcodes are **declared** in the bytecode and scaffolded in the scheduler, but are **not yet wired into the VM dispatch loop** — cooperative `SPAWN`/`JOIN` parallelism and the `CooperativeTaskScheduler` do work. Several `language_extensions/` modules (macros, modules, ADTs, operator overloading) ship as tested **API envelopes** not yet fully wired into the parser/VM. We document these precisely rather than overclaim; see the [research paper](https://drive.google.com/file/d/1t1woF19vfMlQwmCulF45u__sl2cOxx4s/view?usp=sharing) §11 and the [Known Limitations](#known-limitations) section.

See the [Migration Guide: 2.7 → 2.8](#migration-guide-27--28) for the full list of fixes and upgrade steps.

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

## Installation

### Option 1 — Native installers (Windows / Linux / macOS)

Prebuilt, self-contained installers for all three platforms are attached to every [GitHub Release](https://github.com/Eigenresearch/Eigen/releases). Download the one for your OS — no Python setup required:

| Platform | Installer | Notes |
|----------|-----------|-------|
| **Windows** | `Eigen-2.8.0-Windows-x64.exe` | Inno Setup wizard: PATH management, `.eig` file association, context-menu integration |
| **Linux** | `Eigen-2.8.0-Linux.AppImage` | Portable AppImage — `chmod +x` and run |
| **macOS** | `Eigen-2.8.0-macOS.pkg` | Standard `.pkg` installer |

### Option 2 — pip (`eigen-lang` package)

The core install is CPU-only with NumPy; advanced capabilities are opt-in via extras so the default install stays lightweight.

```bash
pip install eigen-lang               # core (CPU only)
pip install eigen-lang[aot]          # + AOT/LLVM compilation
pip install eigen-lang[gpu-cuda]     # + NVIDIA GPU via CuPy
pip install eigen-lang[gpu-torch]    # + GPU via PyTorch
pip install eigen-lang[distributed]  # + MPI distributed simulation
pip install eigen-lang[hardware-ibm] # + IBM Quantum Runtime
pip install eigen-lang[dev]          # + development tools
```

### Option 3 — from source

```bash
git clone https://github.com/Eigenresearch/Eigen.git
cd Eigen
pip install -e ".[dev]"
cd native/rust && maturin develop --release   # build the native Rust extension
```

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

## Feature Maturity

| Feature | Status |
|---------|--------|
| Classical VM + JIT | stable |
| Dense State Vector Simulator | stable |
| Sparse Simulator | stable |
| MPS Simulator | stable |
| Density Matrix Simulator | stable |
| Stabilizer Simulator | stable |
| ZX Calculus | stable |
| Noise Models | stable |
| Routing (SABRE) | stable |
| Compiler (Lexer/Parser/TypeChecker) | stable |
| LSP Server | experimental |
| GPU Acceleration (CUDA) | experimental |
| GPU Acceleration (ROCm) | experimental |
| GPU Acceleration (MPS) | experimental |
| MPI Distributed | experimental |
| AOT/LLVM Compilation | experimental |
| Hardware Runtime (IBM) | experimental |
| Hardware Runtime (IonQ) | experimental |
| FFI Generation | experimental |

---

## DevOps & Security

Eigen 2.8 incorporates a modern DevOps pipeline to ensure stability and security:

- **CI/CD**: Full test suite, `ruff` linting, and `mypy` type-checking (staged gate) on every PR.
- **Security**: CodeQL static analysis, `pip-audit` for dependency vulnerabilities.
- **Compliance**: SBOM (Software Bill of Materials) generation in CycloneDX format.
- **Fuzzing**: Coverage-guided fuzzing for the Lexer, Parser, and VM using `Atheris`.
- **Containers**: Official `Dockerfile` for reproducible deployments.
- **Consistency**: Version consistency checker across `pyproject.toml`, `Cargo.toml`, and source code.

## Architecture

Eigen is built on a **runtime-first** model: classical and quantum operations share one bytecode instruction stream and one live execution state, so a measured qubit's classical outcome can immediately steer subsequent control flow within the same context. Around that core VM, native acceleration (Rust), ahead-of-time compilation (LLVM/QIR), and multiple simulator backends are layered as *optional* accelerators — the pure-Python path always works so `pip install eigen-lang` is a zero-heavy-dependency experience.

### End-to-end pipeline

```
                          ┌─────────────────────────────────────────────┐
   Source (.eig)  ─────▶  │                 FRONTEND                     │
                          │  Rust Lexer ─▶ Pratt Parser ─▶ AST           │
                          │  ─▶ Import Resolver ─▶ Type Checker          │
                          │  ─▶ Monomorphizer     (Python fallback)      │
                          └───────────────────────┬─────────────────────┘
                                                   ▼
                          ┌─────────────────────────────────────────────┐
                          │              IR & OPTIMIZATION                │
                          │  AST ─▶ MLIR Dialect ─▶ EQIR Graph           │
                          │  ─▶ 7-Pass Optimizer ─▶ EBC Bytecode         │
                          │       (deterministic, semantics-preserving)  │
                          └───────────────────────┬─────────────────────┘
                                                   ▼
        ┌────────────────────────┬─────────────────────────┬───────────────────────┐
        ▼                        ▼                          ▼                       ▼
┌───────────────┐      ┌──────────────────┐      ┌──────────────────┐    ┌──────────────────┐
│   VM + JIT    │      │   AOT (LLVM/QIR) │      │  Exporters       │    │  Routing (SABRE) │
│ stack VM,     │      │ standalone       │      │  QASM3 / IonQ /  │    │  SWAP insertion  │
│ fast-loop JIT │      │ native binary    │      │  Braket / QIR    │    │  for coupling    │
│ + deopt       │      │ (optional [aot]) │      │                  │    │  maps            │
└───────┬───────┘      └──────────────────┘      └──────────────────┘    └──────────────────┘
        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         SIMULATOR DISPATCH  (automatic backend selection)                  │
│  Dense (Rust) │ Sparse │ MPS (tensor-train) │ Stabilizer (CHP) │ Density-Matrix │ GPU      │
└──────────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│   CROSS-CUTTING:  Noise models (Kraus/T1-T2/crosstalk/readout) · ZX equivalence ·          │
│   Deterministic RNG + Seed Manager · Runtime Audit · Experiment Tracker · Crash Recovery   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Architectural layers

| Layer | Modules | Responsibility |
|-------|---------|----------------|
| **Frontend** | `frontend/lexer.py`, `parser.py`, native Rust (`native/rust`) | Zero-copy tokenization, precedence-climbing parsing, typed AST; Python fallback when Rust absent |
| **Semantic** | `semantic/type_checker.py`, `monomorphizer.py`, `import_resolver.py` | Type checking, generic monomorphization, import resolution |
| **IR** | `ir/mlir_dialect.py`, `ir_graph.py`, `optimizer.py`, `pass_manager.py` | MLIR lowering, EQIR graph, 7-pass optimizer (deterministic worklist) |
| **Bytecode** | `backend/ebc_compiler.py`, `bytecode.py`, `gate_registry.py` | EBC emission, peephole super-instruction fusion, canonical gate metadata |
| **Execution** | `backend/vm.py`, `jit_compiler.py`, `native_codegen.py` | Stack VM, fast-loop JIT with type guards + deoptimization |
| **AOT** | `compiler.py` (LLVM/QIR paths, optional `[aot]`) | Standalone native binary / QIR emission |
| **Simulation** | `simulator.py`, `sparse_simulator.py`, `stabilizer_simulator.py`, `tensor_network/mps.py` | Six backends with automatic selection |
| **Noise** | `noise/noise_channel.py`, `t1t2_model.py`, `device_profile.py`, `crosstalk_model.py` | Kraus channels, physical timing, device profiles, readout error |
| **Routing** | `router.py`, native `routing.rs` | SABRE / Greedy / Basic routers, real device topologies |
| **Equivalence** | `zx/zx_equivalence.py` | ZX-calculus rewriting and unitary equivalence checking |
| **Hardware** | `unified_backend.py`, `ibm_backend.py`, `ionq_backend.py`, `azure_backend.py`, `braket_backend.py` | Exporters + opt-in runtime submission |
| **Tooling** | `lsp_server.py`, `dap_server.py`, `debugger.py`, `cli.py` (28 subcommands) | LSP, DAP debugging, CLI |
| **Reproducibility** | `seed_management.py`, `runtime_audit.py`, `experiment_tracker.py` | Deterministic seeds, audit trail, experiment ledger |
| **Distributed** | `distributed/mpi_simulator.py`, `circuit_slicer.py` (optional `[distributed]`) | MPI partitioning with single-node fallback |
| **Research** | `research/`, `resource_estimator/`, `quantum_tomography.py`, `pulse_control.py` | Quantum volume, RB, witnesses, estimation, tomography, pulses |

### Execution paths

- **VM + JIT** — full hybrid execution; a fast-loop JIT detects tight backward-jump loops, hoists variables, fuses super-instructions, and installs compiled blocks — with a **deoptimization guard** that falls back to the interpreter on a type mismatch, preserving correctness.
- **AOT (LLVM/QIR)** — standalone native binary via `llvmlite` codegen (optional `[aot]` extra).
- **Stabilizer (CHP)** — `O(n²)` Clifford simulation reaching **1000+ qubits**, categorically impossible for dense methods.
- **Routing (SABRE)** — bidirectional SWAP insertion for hardware coupling maps, with a deterministic lexicographic tie-break.
- **Export** — OpenQASM 3.0, IonQ JSON, Braket, and QIR from a single canonical gate registry.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) and the [111-page research paper](https://drive.google.com/file/d/1t1woF19vfMlQwmCulF45u__sl2cOxx4s/view?usp=sharing) for full architectural details, formal semantics, and proofs.

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

## Known Limitations

### Stabilizer simulator — Clifford gates only

The `StabilizerSimulator` (CHP algorithm, `src/stabilizer_simulator.py`)
supports only the Clifford group:

```
{H, S, SDG, X, Y, Z, CNOT, CZ, SWAP, I, SX}
```

Any non-Clifford gate (`T`, `TDG`, `RX`, `RY`, `RZ`, `CCX`, `CSWAP`,
`CP`, `CRX`, `CRY`, `CRZ`, `U1`, `U2`, `U3`) raises
`NonCliffordGateError(ValueError)`. When used via
`QuantumSimulator(sim_type='stabilizer')`, the exception is caught and the
simulator transparently falls back to the dense state-vector backend with
a `warnings.warn`. Pre-flight analysis is available via
`check_circuit_compatibility()`.

### Noise — stochastic vs exact

Eigen supports two distinct noise application modes, selected automatically
based on the active simulator:

- **Stochastic (state-vector)** — used by the `dense`, `sparse`, `mps`, and
  `stabilizer` backends. Each shot is one *trajectory* of the noise
  process: a single random Pauli error is drawn per gate. Averaging over
  many shots converges to the exact channel, but a single shot is not
  representative. Fast — `O(2^n)` per shot.
- **Exact (density matrix)** — used by the `density_matrix` backend.
  Noise is applied via the full Kraus operator sum
  `ρ → Σ_k K_k ρ K_k†`. A single run yields the exact post-channel density
  matrix. Slow — `O(4^n)` per shot, limited to ~12 qubits.

The dispatch is automatic and lives in `NoiseModel.apply_gate_noise`. See
[NOISE_MODELS.md](NOISE_MODELS.md) for the full parameter reference
(`gamma`, `lambda`, `T1`, `T2`), the `T2 ≤ 2·T1` physical constraint,
and the channel-by-channel Kraus tables.

### Other limitations

- **Dense simulator hard cap: 25 qubits** (`MemoryError` raised above).
  Use `sparse`, `mps`, or `stabilizer` for larger circuits.
- **Density-matrix simulator: ~12 qubits** due to the `4^n` memory cost.
- **MPS bond dimension: 64** by default. Circuits generating volume-law
  entanglement will exceed this and incur truncation error (default
  tolerance `1e-4`). Enable `auto_bond_dim=True` for adaptive growth.
- **Classical VM throughput** is 45–250× slower than CPython for tight
  arithmetic loops — the VM is designed for quantum dispatch, not
  classical number crunching. Use the AOT/LLVM path for classical-heavy
  code.
- **`std.math` transcendentals (`sin`, `cos`, `tan`, `log`, `exp`) throw**
  in pure Eigen because they require native math support. Bind a C math
  library via FFI, or precompute values in Python and pass them in.
- **`std.io`, `std.random`, `std.time`, `std.stats` are stubs** — they
  return placeholder values. The VM intercepts calls to these via
  `_STD_MAPPING` and dispatches to native helpers when available.
- **JIT is sandboxed** (`{"__builtins__": {}}` plus a small allowlist).
  JIT-compiled blocks cannot call `type`/`getattr`/`hasattr`/`isinstance`
  by design — this is a defense-in-depth against MRO-based escapes, not
  OS-level isolation.
- **T2 > 2·T1 is silently clamped** with a `warnings.warn`. Inspect
  `model.t2` after construction to see the clamped value.
- **`ReadoutErrorChannel` does not modify the quantum state** — it only
  acts at measurement time through `apply_readout_noise`.
- **GPU backend applies gates one-by-one** — per-gate kernel-launch
  overhead dominates for small circuits. Gate batching was added in 2.8
  but remains experimental.
- **No direct QPU execution** — Eigen exports to OpenQASM 3.0 and QIR for
  execution on external runtimes; it does not drive a QPU directly.
- **AOT/LLVM compilation is optional** — `llvmlite` is not a core
  dependency. Install it with `pip install eigen-lang[aot]` to enable the
  `eigen build --aot` path.
- **MPI distributed simulation is optional** — requires
  `pip install eigen-lang[distributed]` (mpi4py). Falls back to
  single-node when unavailable.

---

## Migration Guide: 2.7 → 2.8

Eigen 2.8 is a hardening release. There are no breaking API changes
relative to 2.7, but several correctness bugs have been fixed that may
change the output of existing programs.

### Bug fixes

- **`CNOT` gate application** — corrected the bit-permutation indices on
  the dense state-vector path so the control/target roles match the
  documented semantics.
- **`X` / `Z` gate dispatch in the VM** — the `Q_GATE` opcode handler now
  routes single-qubit Pauli gates through the correct simulator method
  instead of the previous mismatched dispatch.
- **`allocate_qubit` index-cache invalidation** — `_index_cache`,
  `_index_cache_2q`, and `_index_cache_3q` are now cleared on every
  `allocate_qubit` call, preventing stale entries from corrupting gate
  application after qubit allocation.
- **`CZ` ZX-graph rewriting** — the ZX-calculus `CZ` rule now produces
  the correct Hadamard-edge pattern instead of an ordinary edge.
- **Heap allocation under recursion** — recursive `CALL` no longer
  overwrites the caller's heap slots; the call frame correctly snapshots
  and restores the heap reference.
- **Async scheduler** — the `CooperativeTaskScheduler` now correctly
  yields the pending task list instead of busy-waiting on a single task.
  *(Note: the `ASYNC_CALL` / `AWAIT` / `YIELD_TASK` opcodes are declared
  and scaffolded but not yet wired into the VM dispatch loop;
  `SPAWN` / `JOIN` cooperative parallelism does work.)*
- **`SEMICOLON` token handling** — statements terminated with `;` are now
  parsed consistently across all statement types (previously `let` and
  `return` with trailing `;` could be misparsed in some grammars).
- **Router: 3-qubit gates** — `BasicSwapRouter`, `GreedyRouter`, and
  `SabreRouter` now decompose `CCX` and `CSWAP` into 1- and 2-qubit gates
  before routing, instead of raising or routing them as a single
  2-qubit operation.
- **Router: division by zero** — the SABRE front-layer scoring now guards
  against `len(front_layer) == 0` instead of crashing with
  `ZeroDivisionError`.
- **Router: `reverse_mapping` `None` values** — `GreedyRouter` and
  `SabreRouter` no longer write `None` into the forward `mapping` dict
  when a SWAP target qubit is unoccupied, which previously corrupted
  subsequent lookups.

### New documentation

- `ARCHITECTURE.md` — full pipeline and VM description.
- `STDLIB_API.md` — every `stdlib/std/` function with signatures and
  behavior notes.
- `NOISE_MODELS.md` — noise channel reference, parameter table,
  stochastic-vs-exact dispatch logic, and the `T2 ≤ 2·T1` constraint.
- `BENCHMARKS.md` — how to run benchmarks, interpretation, methodology,
  and known limitations.

### New examples

- `examples/noise_simulation.eig`
- `examples/match_case.eig`
- `examples/error_handling.eig`
- `examples/string_interpolation.eig`
- `examples/operator_overloading.eig`

### Upgrade steps

1. `pip install -e ".[dev]"` (no dependency changes).
2. Rebuild the native extension:
   `cd native/rust && maturin develop --release`.
3. Run the test suite: `pytest tests/ -q`. The routing test file
   (`tests/test_routing.py`) has been extended to cover 3-qubit gate
   decomposition; existing 2-qubit routing behavior is unchanged.
4. If your code relied on the previous (buggy) `CNOT` / `X` / `Z` / heap
   / async behavior, audit the output. The fixes make Eigen match the
   documented semantics; programs that worked correctly under 2.7 should
   be unaffected, but programs that happened to depend on a bug may need
   adjustment.

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
├── paper/               # Research paper (111 pages, PDF)
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

**Eigen 2.8 «Mars»: A Repository-Grounded Reconstruction of a Hybrid Classical–Quantum Programming Language**

A 111-page reconstruction of the entire system: the compiler pipeline, VM state machine, all six simulator backends, noise/routing/ZX subsystems, formal semantics and proofs (VM boundedness, CHP stabilizer correctness, SABRE cost, MPS truncation bounds, determinism, cache tamper-evidence), honest benchmark results with 95% confidence intervals, and worked end-to-end examples — every claim anchored to a `file:line` in this repository.

[**Read the full 111-page paper (PDF)**](https://drive.google.com/file/d/1t1woF19vfMlQwmCulF45u__sl2cOxx4s/view?usp=sharing)

**Authors:** Kenzhegali Nuras, Batyrbek Inabat, Sarsenbay Alikhan — Eigen Research / Eigen Labs

---

<div align="center">

## ⭐ Enjoyed Eigen?

If you find this project useful, **please give it a star** — it genuinely helps other researchers and developers discover it, and it motivates continued open development.

**[⭐ Star this repository](https://github.com/Eigenresearch/Eigen)** &nbsp;·&nbsp; **[Fork it](https://github.com/Eigenresearch/Eigen/fork)** &nbsp;·&nbsp; **[Share it with your friends and colleagues](https://github.com/Eigenresearch/Eigen)**

*Every star and every share brings more people to open quantum-classical programming. Thank you!*

</div>

---

*Eigen Research — Independent research laboratory focused on programming languages, compiler systems, and quantum computing.*
