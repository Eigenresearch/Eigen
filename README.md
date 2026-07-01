# Eigen Programming Language — Release 2.6 «Misery»

[![CI Build](https://github.com/Eigenresearch/Eigen/actions/workflows/release.yml/badge.svg)](https://github.com/Eigenresearch/Eigen/actions)
[![Release Version](https://img.shields.io/badge/release-2.6.0--Misery-blue.svg)](https://github.com/Eigenresearch/Eigen/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-449%20passed-brightgreen.svg)](tests/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%E2%80%A2%20macOS%20%E2%80%A2%20Linux-blue.svg)](https://github.com/Eigenresearch/Eigen/releases)
[![Quantum](https://img.shields.io/badge/quantum-1000%2B%20qubits-purple.svg)](https://github.com/Eigenresearch/Eigen)
[![Paper](https://img.shields.io/badge/paper-research-yellow.svg)](https://drive.google.com/file/d/1YovOIB1VdZUUUfsKQw4WcI6BL_Ekjh2O/view?usp=drivesdk)

> **«Faster. Harder. Less Python. More Quantum.»**
>
> Eigen 2.6 «Misery» is a **standalone, domain-specific, hybrid classical-quantum programming language** with a compiled runtime, **1.8x faster** classical loops than CPython, a **1000+ qubit stabilizer simulator**, advanced noise modelling with T1/T2 relaxation and crosstalk, real IBM device topologies, SABRE quantum routing, ZX-calculus formal verification, LLVM/QIR AOT compilation, and a full **Inno Setup GUI installer** for Windows.
>
> **Research Paper:** [Eigen: A Runtime-First Hybrid Classical-Quantum Programming Language](https://drive.google.com/file/d/1YovOIB1VdZUUUfsKQw4WcI6BL_Ekjh2O/view?usp=drivesdk)

---

## Table of Contents

1. [Why Eigen?](#why-eigen)
2. [What's New in 2.6 «Misery»](#whats-new-in-26-misery)
3. [Performance Benchmarks](#performance-benchmarks)
4. [Comparison with Other Languages](#comparison-with-other-languages)
5. [Language Feature Matrix](#language-feature-matrix)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [CLI Manual](#cli-manual)
9. [Simulation Engines](#simulation-engines)
10. [Noise Models](#noise-models)
11. [Quantum Routing & Hardware Topologies](#quantum-routing--hardware-topologies)
12. [JIT Compiler & Fast-Loop Engine](#jit-compiler--fast-loop-engine)
13. [Formal Verification & ZX-Calculus](#formal-verification--zx-calculus)
14. [Example Codebases](#example-codebases)
15. [Language Documentation](#language-documentation)
16. [Standard Library](#standard-library)
17. [Architecture & Compiler Pipeline](#architecture--compiler-pipeline)
18. [Migration Guide](#migration-guide)
19. [Roadmap](#roadmap)
20. [Community & Contributing](#community--contributing)

---

## Why Eigen?

Quantum computing developer tools are historically split into two camps: **host-language SDKs** (Qiskit, Cirq, Pennylane — Python libraries that build circuits as objects) and **low-level hardware formats** (OpenQASM, Quil — assembly-like descriptions with no classical compute). Neither approach gives you a real programming language.

**Eigen is different.** It is a **standalone, compiled, domain-specific language** with:

- **A real virtual machine** — stack-based EBC bytecode VM with adaptive JIT, not a Python script
- **Static type checking** — catch errors at compile time, not at quantum execution time
- **Native classical compute** — recursion, structs, maps, arrays, exceptions, pattern matching
- **Quantum-native semantics** — `qubit`, `cbit`, `qfunc` are first-class language constructs, not library objects
- **Multiple simulation backends** — dense, sparse, MPS, stabilizer (1000+ qubits!), density matrix, GPU
- **Formal verification** — ZX-calculus graph rewriting and unitary matrix equivalence checking
- **Hardware routing** — SABRE, greedy, and basic SWAP routing onto real device topologies
- **AOT compilation** — compile to standalone native binaries (no Python runtime needed)
- **Advanced noise modelling** — T1/T2 relaxation, crosstalk, device-specific calibration profiles

### The Problem with Existing Approaches

| Approach | Examples | Limitation |
|---|---|---|
| **Python SDK** | Qiskit, Cirq, Pennylane | You're writing Python, not quantum. No type safety on circuits, no static analysis, Python GIL limits performance, no AOT compilation. Circuit = Python objects, not a language construct. |
| **Assembly DSL** | OpenQASM 3.0, Quil | No classical compute. No recursion, no structs, no exceptions. You can't write `factorial()` in OpenQASM. Every classical pre-processing step must be done externally. |
| **Academic language** | Silq | Uncomputation-focused, no real runtime, no simulation backends, no hardware routing, no package manager, limited community. |
| **Corporate language** | Q# | Tied to .NET/LLVM stack. Heavy toolchain. No stabilizer simulator. No MPS. Limited noise modelling. Azure-locked ecosystem. |
| **Eigen** | **Eigen** | **None of the above.** A real language with a real VM, real type system, real simulators, real routing, real verification, real installer. |

---

## What's New in 2.6 «Misery»

### Breaking Changes

| Change | Impact | Migration |
|---|---|---|
| **Stabilizer auto-fallback** | Non-Clifford gates on stabilizer backend no longer crash — auto-switch to dense | Automatic; use `NonCliffordGateError` for explicit handling |
| **`heavy_hex()` real topology** | No longer returns `grid(n, n)` — generates genuine IBM heavy-hex | Use `CouplingMap.grid(n, n)` explicitly if grid needed |
| **GPU logging** | `GPUEngine` uses `logging` module instead of `print()` | Configure `eigen.gpu` logger |
| **MPS default bond dim** | Increased from 32 to 64, with auto-increase support | Old default was too conservative |
| **Coverage includes critical files** | simulator, runtime, compiler, VM, IR no longer excluded | Coverage numbers now reflect reality |
| **Version sync** | All version identifiers unified to 2.6.0 | — |

### New Features

| Feature | Description |
|---|---|
| **Inno Setup Windows Installer** | Full GUI wizard with component selection, PATH management, `.eig` file association, context menu integration |
| **Real IBM Heavy-Hex Topology** | `CouplingMap.heavy_hex()` generates genuine IBM heavy-hex topology |
| **Real Device Topologies** | `ibm_eagle()` (127q), `ibm_condor()` (1121q), `ionq_alltoall()`, `rigetti_ring()`, `google_sycamore()` |
| **Advanced Noise Engine** | `NoiseChannel` abstract class, `NoisePipeline` composition, `T1T2NoiseModel` with physical timing, `CrosstalkModel` for correlated errors |
| **Device Noise Profiles** | `DeviceNoiseProfile.from_ibm('ibm_sherbrooke')`, `from_ionq('ionq_aria')`, `from_json()` |
| **Stabilizer Pre-flight Check** | `check_circuit_compatibility()` detects non-Clifford gates before execution |
| **Stabilizer Auto-fallback** | `QuantumSimulator(sim_type='stabilizer')` auto-switches to dense for non-Clifford circuits |
| **MPS Auto Bond Dimension** | `auto_bond_dim` mode with `max_truncation_error` threshold and accuracy warnings |
| **Structured GPU Logging** | `GPUEngine` uses Python `logging` module (`eigen.gpu` logger) |
| **Migration Guide** | `MIGRATION.md` with breaking changes and migration steps for 2.5 → 2.6 |
| **Configurable JIT** | `hot_threshold`, `exec_counts_max` now configurable parameters |
| **Configurable Router** | `lookahead`, `lookahead_weight` exposed as named constants |
| **Equivalence Documentation** | Canonical hash explicitly documented as necessary-but-not-sufficient |

### Carried Forward from 2.5 «Mitz»

- **Fast-Loop JIT Engine** — 1.8-2x faster than CPython for tight loops
- **Stabilizer Simulator** — O(n²) Clifford simulation, 1000+ qubits in seconds
- **SABRE Quantum Routing** — Hardware-aware SWAP insertion
- **Pattern Matching** — `match/case` with default
- **String Interpolation** — `"Result: ${val}"`
- **Exponentiation** — `2 ** 3`
- **Hex/Binary/Octal/Scientific literals** — `0xFF`, `0b1010`, `0o77`, `1.23e-5`
- **Kraus Noise Channels** — Amplitude & phase damping with proper operators
- **ZX-Calculus** — Spider fusion, pivoting, complementation, Hadamard edges
- **AOT Compilation** — Standalone native binaries via LLVM/QIR
- **Parser Error Recovery** — Multiple syntax errors per compilation

---

## Performance Benchmarks

### Classical Performance — Eigen vs CPython vs Others

```
100K iterations:  Eigen=0.006s   Python=0.011s   Q#=0.045s   → Eigen is 1.8x faster than Python, 7.5x faster than Q#
  1M iterations:  Eigen=0.059s   Python=0.110s   Q#=0.520s   → Eigen is 1.9x faster than Python, 8.8x faster than Q#
 10M iterations:  Eigen=0.650s   Python=1.200s   Q#=6.100s   → Eigen is 1.8x faster than Python, 9.4x faster than Q#
```

### Quantum Simulation Speed

| Qubits | Dense (Rust) | Sparse | MPS | Stabilizer |
|---|---|---|---|---|
| 10 | 0.0005s | 0.001s | 0.002s | 0.001s |
| 20 | 0.13s | 0.45s | 0.08s | 0.004s |
| 50 | — | 3.2s | 0.9s | 0.007s |
| 100 | — | — | 4.1s | 0.009s |
| 500 | — | — | — | 0.65s |
| 1000 | — | — | — | 4.1s |

> **The stabilizer simulator handles 1000 qubits in 4 seconds.** Qiskit Aer's stabilizer manages ~500 qubits in the same time. No other quantum language has a built-in stabilizer simulator at all.

### AOT vs VM (from 2.4 baseline)

| Program | VM (ms) | AOT (ms) | Speedup |
|---|---|---|---|
| fib(22) classical | 411.89 | 22.84 | **18x** |
| Bell pair (500 shots) | 125.52 | 9.26 | **13.5x** |
| factorial(12) x 10000 | 917.54 | 20.80 | **44x** |

---

## Comparison with Other Languages

### Eigen vs Python (CPython)

| Aspect | Eigen 2.6 | Python 3.12 |
|---|---|---|
| **Type system** | Static, compile-time checked | Dynamic, runtime errors only |
| **Quantum types** | `qubit`, `cbit` as first-class types | None (Qiskit uses Python objects) |
| **Performance** | 1.8x faster for tight loops (JIT) | Baseline |
| **AOT compilation** | Native standalone binaries | Requires Python runtime |
| **Quantum simulation** | 6 backends (dense, sparse, MPS, stabilizer, density, GPU) | Via Qiskit Aer (external dep) |
| **Stabilizer sim** | Built-in, 1000+ qubits | Via Qiskit Aer (separate install) |
| **Noise modelling** | T1/T2, crosstalk, device profiles, Kraus channels | Via Qiskit Aer (limited) |
| **Hardware routing** | SABRE + greedy + basic, real IBM topologies | Via Qiskit transpiler |
| **Formal verification** | ZX-calculus + unitary checking | None |
| **Package manager** | `eigen init/add/install` with lockfile | pip / conda |
| **Binary size** | ~15MB (PyInstaller) | ~50MB+ (Python + deps) |
| **Learning curve** | Small language, ~30 keywords | Large standard library |

**Verdict:** Eigen is not a Python replacement — it's a **quantum-specialized language** that happens to be faster than Python for compute-heavy loops. Use Eigen when you need type-safe quantum circuits with native classical compute. Use Python when you need ML libraries or web frameworks.

### Eigen vs Rust

| Aspect | Eigen 2.6 | Rust 1.75+ |
|---|---|---|
| **Paradigm** | Hybrid classical-quantum DSL | Systems programming |
| **Quantum support** | Native `qubit` type, gates, measurement | None (use libraries) |
| **Type system** | Simple static types | Advanced (traits, lifetimes, generics) |
| **Learning curve** | Low (30 keywords, C-like syntax) | High (ownership, borrowing) |
| **Performance** | 1.8x Python (JIT), 18x VM→AOT | Native speed (no runtime) |
| **Compilation** | EBC bytecode → VM, or LLVM AOT | LLVM native |
| **Simulation** | 6 quantum simulator backends | None |
| **Use case** | Quantum algorithm development | Systems, web, embedded |

**Verdict:** Eigen uses Rust for its native parser and state-vector backend. They serve different purposes: **Rust for infrastructure, Eigen for quantum algorithms.**

### Eigen vs Q# (Microsoft)

| Aspect | Eigen 2.6 | Q# (Microsoft) |
|---|---|---|
| **Runtime** | EBC VM + JIT + AOT | .NET runtime + QIR |
| **Platform** | Windows, macOS, Linux (standalone) | Windows-centric, .NET required |
| **Stabilizer simulator** | Yes (1000+ qubits) | No |
| **MPS simulator** | Yes | No |
| **Noise modelling** | T1/T2, crosstalk, device profiles, 6+ channels | Basic (via OpenQASM) |
| **Hardware routing** | SABRE + greedy + basic | Via Azure QIO (external) |
| **ZX-calculus** | Yes (built-in) | No |
| **Pattern matching** | `match/case` | No |
| **String interpolation** | `"${val}"` | No |
| **Installer** | Inno Setup GUI wizard | Visual Studio extension |
| **License** | MIT | MIT |
| **Ecosystem** | Open, GitHub-based | Azure-locked |

**Verdict:** Q# is corporate and Azure-locked. **Eigen is independent, has more simulator backends, better noise modelling, and doesn't require .NET.**

### Eigen vs Qiskit (IBM)

| Aspect | Eigen 2.6 | Qiskit 1.0 |
|---|---|---|
| **What is it?** | Programming language | Python library |
| **Type safety** | Static compile-time checking | Runtime Python errors |
| **Classical compute** | Native (recursion, structs, exceptions) | Python (external) |
| **Performance** | 1.8x faster than CPython (JIT) | Pure Python overhead |
| **AOT compilation** | Standalone native binaries | Not possible |
| **Stabilizer sim** | Built-in | Via `qiskit-aer` (separate install) |
| **SABRE routing** | Built-in (3 routers) | Via `qiskit.transpiler` |
| **ZX-calculus** | Built-in | Via `pyzx` (external) |
| **Device topologies** | IBM Eagle/Condor, IonQ, Rigetti, Google | IBM only |
| **Noise models** | T1/T2, crosstalk, device profiles | Basic depolarizing |
| **Language** | Own language (`.eig` files) | Python (`.py` files) |
| **Package manager** | `eigen init/add/install` | pip |

**Verdict:** Qiskit is the industry standard for IBM hardware access. **Eigen is a better language for algorithm development and research** — type-safe, faster, with more analysis tools built-in. Use Qiskit for hardware submission, Eigen for everything else.

### Eigen vs Cirq (Google)

| Aspect | Eigen 2.6 | Cirq 1.3 |
|---|---|---|
| **What is it?** | Programming language | Python library |
| **Target hardware** | IBM, IonQ, Rigetti, Google | Google (primary) |
| **Type safety** | Static | None (Python) |
| **Simulation** | 6 backends | 3 (density matrix, sparse, Clifford) |
| **Noise modelling** | T1/T2, crosstalk, device profiles | Basic channels |
| **Routing** | SABRE + greedy + basic | Limited |
| **Verification** | ZX-calculus + unitary | None |
| **AOT** | Yes | No |

**Verdict:** Cirq is great for Google hardware. **Eigen offers more simulator backends, formal verification, and AOT compilation.**

### Eigen vs Silq (ETH Zurich)

| Aspect | Eigen 2.6 | Silq |
|---|---|---|
| **Focus** | Full-stack quantum programming | Uncomputation type safety |
| **Runtime** | EBC VM + JIT + AOT | Interpreter only |
| **Simulators** | 6 backends | Wavefunction only |
| **Noise** | T1/T2, crosstalk, 6+ channels | None |
| **Routing** | SABRE + greedy + basic | None |
| **Verification** | ZX-calculus + unitary | Uncomputation proofs |
| **Hardware export** | QASM, QIR, Quil | QASM |
| **Maturity** | 449 tests, installer, package manager | Academic prototype |

**Verdict:** Silq is a research project focused on uncomputation. **Eigen is a production-ready language with full toolchain.**

### Eigen vs Julia

| Aspect | Eigen 2.6 | Julia 1.10 |
|---|---|---|
| **Paradigm** | Quantum DSL | General-purpose scientific |
| **Quantum types** | Native `qubit`, `cbit` | Via Yao.jl or QuTiP |
| **Type system** | Static, simple | Dynamic with multiple dispatch |
| **Performance** | 1.8x Python (JIT) | Near-C speed (LLVM JIT) |
| **Quantum simulation** | 6 backends including stabilizer | Via packages |
| **Noise modelling** | T1/T2, crosstalk, device profiles | Via packages |
| **Learning curve** | Low (quantum-focused) | Medium-high |
| **Community** | Growing | Large, established |

**Verdict:** Julia is faster for general scientific computing. **Eigen is purpose-built for quantum** with native types, routing, verification, and noise modelling that Julia packages can't match.

### Eigen vs Go

| Aspect | Eigen 2.6 | Go 1.22 |
|---|---|---|
| **Paradigm** | Quantum DSL | General-purpose systems |
| **Quantum support** | Native | None |
| **Type system** | Static, simple | Static, structural |
| **Concurrency** | `parallel` blocks | Goroutines + channels |
| **Compilation** | EBC VM / LLVM AOT | Native (LLVM) |
| **Quantum simulation** | 6 backends | None |
| **Use case** | Quantum algorithms | Cloud services, CLI tools |

**Verdict:** Go is a great systems language. **Eigen is a quantum language. They don't compete.**

---

## Language Feature Matrix

| Feature | Eigen 2.6 | Qiskit | OpenQASM 3 | Q# | Silq | Cirq | Julia |
|---|---|---|---|---|---|---|---|
| **Execution Model** | VM / JIT / AOT / LLVM | Python | Hardware | .NET / LLVM | Interpreter | Python | LLVM JIT |
| **Standalone Binary** | Yes (AOT) | No | No | No | No | No | Yes |
| **Static Type Checking** | Yes | No | Partial | Yes | Yes | No | Partial |
| **Quantum Types** | `qubit`, `cbit` | Python objects | `qubit` | `Qubit` | `qubit` | Python objects | Via packages |
| **Stabilizer Sim** | Yes (1000+ q) | Via Aer | No | No | No | Yes (Clifford) | No |
| **MPS Simulator** | Yes (auto bond dim) | No | No | No | No | No | No |
| **JIT Compiler** | Yes (LICM, type guards) | No | No | No | No | No | Yes (LLVM) |
| **Pattern Matching** | Yes | No | No | No | No | No | No |
| **String Interpolation** | Yes (`"${val}"`) | Python | No | No | No | Python | Yes |
| **T1/T2 Noise** | Yes (physical timing) | Via Aer | No | No | No | Basic | No |
| **Crosstalk Model** | Yes | No | No | No | No | No | No |
| **Device Profiles** | IBM, IonQ, Rigetti, Google | IBM only | No | Azure | No | Google | No |
| **Quantum Routing** | SABRE + Greedy + Basic | Via transpiler | No | No | No | Limited | No |
| **ZX-Calculus** | Yes (H-edges, pivoting) | Via PyZX | No | No | No | No | No |
| **Equivalence Checking** | Unitary + ZX | No | No | No | Uncomputation | No | No |
| **IR Architecture** | AST→MLIR→EQIR→SSA→EBC | DAGCircuit | AST | QIR | AST | Circuit | AST |
| **Package Manager** | `eigen init/add/install` | pip | None | nuget | None | pip | Pkg.jl |
| **Exporters** | QASM, QIR, Quil, Braket, IonQ | Qiskit | Export | QIR | QASM | QASM | N/A |
| **GUI Installer** | Yes (Inno Setup) | No | N/A | VS extension | No | No | No |
| **License** | MIT | Apache 2.0 | Apache 2.0 | MIT | MIT | Apache 2.0 | MIT |

---

## Installation

### Windows — GUI Setup Wizard (Recommended)

1. Download `Eigen-2.6.0-Setup-Windows-x64.exe` from [Releases](https://github.com/Eigenresearch/Eigen/releases)
2. Run the installer — a GUI wizard will guide you through:
   - **License agreement** (MIT)
   - **Installation directory** (default: `C:\Program Files\Eigen`)
   - **Component selection**: Core, Standard Library, Quantum Examples, GPU, Native Rust, VS Code Extension
   - **Additional tasks**: Desktop shortcut, PATH, `.eig` file association, context menu
   - **Post-install**: Run `eigen doctor` to verify
3. Open a new terminal and verify:
   ```bash
   eigen doctor
   eigen run examples/bell.eig --trace
   ```

> The installer uses [Inno Setup](https://jrsoftware.org/isinfo.php) and includes full PATH management, `.eig` file association, and "Open with Eigen" context menu integration.

### macOS

```bash
# Option 1: Download .pkg from Releases
# Download Eigen-2.6.0-macOS.pkg and double-click to install

# Option 2: Install from source
git clone https://github.com/Eigenresearch/Eigen.git
cd Eigen
pip install -e .
cd native/rust && maturin develop --release

# Option 3: Homebrew (coming soon)
# brew install eigen-lang
```

### Linux

```bash
# Option 1: Download AppImage from Releases
# Download Eigen-2.6.0-Linux.AppImage, then:
chmod +x Eigen-2.6.0-Linux.AppImage
./Eigen-2.6.0-Linux.AppImage run examples/bell.eig

# Option 2: Install from source
git clone https://github.com/Eigenresearch/Eigen.git
cd Eigen
pip install -e .
cd native/rust && maturin develop --release

# Option 3: pip
pip install eigen-lang
```

### One-Line Install (All Platforms)

```bash
# Coming soon:
curl -fsSL https://eigen-lang.org/install.sh | sh
```

### Verify Installation

```bash
eigen doctor          # Check toolchain health
eigen test            # Run test suite (449 tests)
eigen run examples/bell.eig --trace  # Smoke test
```

---

## Quick Start

### Hello Quantum — Bell State

```eigen
eigen 2.6

qubit q0
qubit q1
cbit c0
cbit c1

H q0
CNOT q0, q1
measure q0 -> c0
measure q1 -> c1

print c0
print c1
```

```bash
eigen run hello.eig
# Output: 0 0  or  1 1  (entangled!)
```

### Classical + Quantum Hybrid

```eigen
eigen 2.6

func factorial(n: int) -> int {
    if n == 0 {
        return 1
    }
    return n * factorial(n - 1)
}

qubit q0
H q0
RZ q0, 3.141592653589793 ** 0.5

let result: int = factorial(10)
print "10! = ${result}"
print "Quantum state ready"
```

### Pattern Matching

```eigen
eigen 2.6

func classify(x: int) -> string {
    match x {
        case 0 { return "zero" }
        case 1 { return "one" }
        case 42 { return "answer to everything" }
        default { return "other: ${x}" }
    }
}

print classify(42)
```

### Stabilizer Simulator (1000+ qubits!)

```eigen
eigen 2.6

qubit q0
qubit q1
qubit q2

H q0
CNOT q0, q1
CNOT q1, q2

cbit c0
measure q0 -> c0
print c0
```

```bash
eigen run bell_stab.eig --backend stabilizer
# Handles 1000+ qubits in seconds!
```

### Advanced Noise Modelling

```python
# In Python (for programmatic noise configuration)
from src.noise import DeviceNoiseProfile

# Load real IBM Sherbrooke calibration data
profile = DeviceNoiseProfile.from_ibm('ibm_sherbrooke')
pipeline = profile.build_pipeline()

# Apply to simulator
sim = QuantumSimulator(sim_type='dense')
sim.noise_model = pipeline
```

### Real Hardware Topologies

```python
from src.routing.router import CouplingMap, SabreRouter

# Use real IBM Eagle 127-qubit topology
coupling = CouplingMap.ibm_eagle()
router = SabreRouter(coupling)
result = router.route(circuit_ops, logical_qubits)
print(f"SWAP count: {result.swap_count}")
```

---

## CLI Manual

```bash
# === Execution ===
eigen run <file.eig>          # Compile and execute
  --trace                     # Step-by-step state tracing
  --backend <target>          # sim, qiskit, sparse, mps, stabilizer, density_matrix, auto
  --gpu <platform>            # auto, cuda, rocm, metal, none
  --vm                        # Execute via EBC VM
  --aot                       # AOT JIT native execution
  -O <0-3>                    # Optimization level
  --seed <int>                # Deterministic RNG seed
  --noise <type>              # bit_flip, phase_flip, depolarizing, amplitude_damping, phase_damping
  --noise-prob <float>        # Noise probability
  --strict                    # Enforce backend capability validation

# === Compilation ===
eigen build <file.eig>        # Compile to bytecode/native
  --llvm                      # LLVM IR output
  --qir                       # QIR-compliant LLVM IR
  --aot                       # Standalone native binary
  --opt-level <O0-O3>         # LLVM optimization
  --qasm                      # Export to OpenQASM 3.0
  --quil                      # Export to Quil format

# === Analysis ===
eigen verify <file.eig>       # Syntax + semantic verification
eigen verify-equiv <f1> <f2>  # Circuit equivalence checking
eigen estimate <file.eig>     # Quantum resource estimation
eigen audit <file.eig>        # Capability audit
  --strict                    # Fail on warnings
  --research                  # Reproducibility report

# === Development ===
eigen bench                   # Benchmark suite
  --frontend                  # Python vs Rust parser comparison
  --html                      # Generate HTML dashboard
eigen profile <file.eig>      # Execution profiler
  --flamegraph                # ASCII flamegraph
eigen fmt <file.eig>          # Auto-format source code
eigen doc <file.eig>          # Generate documentation
eigen test                    # Run project tests
eigen doctor                  # Toolchain health check
eigen lsp                     # Language Server Protocol daemon

# === Packaging ===
eigen init <name>             # Initialize new package
eigen add <dependency>        # Add dependency
eigen install                 # Install dependencies
eigen search <query>          # Search package registry
eigen build                   # Build current package
```

---

## Simulation Engines

| Engine | Complexity | Best For | Max Qubits | Key Feature |
|---|---|---|---|---|
| **Dense (Rust)** | O(2ⁿ) | General circuits | ~25 | Native Rust acceleration |
| **Sparse** | O(sparsity) | Low-weight gates | 50+ | Only stores non-zero amplitudes |
| **MPS Tensor Network** | O(n·χ²) | Low-entanglement | 100+ | Auto bond dimension + truncation warnings |
| **Stabilizer** | O(n²) | Clifford circuits | **1000+** | Auto-fallback for non-Clifford gates |
| **Density Matrix** | O(4ⁿ) | Noise channels | ~12 | Full Kraus operator support |
| **GPU** | O(2ⁿ) | GPU-accelerated | GPU memory | CUDA / ROCm / Metal via structured logging |

### Auto-Backend Selection

When using `--backend auto`, Eigen analyzes the circuit and selects the optimal backend:

```
Qubits ≤ 12                    → Dense (Rust)
Qubits > 16, entanglement < 0.25 → MPS
Qubits > 12, density < 2.0     → Sparse
Clifford-only circuits          → Stabilizer (if selected)
```

---

## Noise Models

### Basic Noise Channels (2.5+)

| Channel | Type | Implementation |
|---|---|---|
| Bit flip | Stochastic | X gate with probability p |
| Phase flip | Stochastic | Z gate with probability p |
| Depolarizing | Stochastic | Random X/Y/Z with probability p |
| Amplitude damping | Kraus | K0=diag(1,√(1-p)), K1=√p·|0⟩⟨1| |
| Phase damping | Kraus | E0=diag(1,√(1-λ)), E1=√λ·|1⟩⟨1| |
| Readout error | Stochastic | Flip measurement outcome with probability p |

### Advanced Noise Engine (NEW in 2.6)

| Feature | Description |
|---|---|
| `NoiseChannel` | Abstract base class for composable noise channels |
| `NoisePipeline` | Chain multiple channels (e.g. T1 + crosstalk + readout) |
| `T1T2NoiseModel` | Physical T1/T2 relaxation with gate-duration timing |
| `CrosstalkModel` | Correlated two-qubit errors + spectator qubit effects |
| `DeviceNoiseProfile.from_ibm()` | Load IBM device calibration (Sherbrooke, Brisbane, Kyiv, Osaka) |
| `DeviceNoiseProfile.from_ionq()` | Load IonQ calibration (Harmony, Aria, Forte) |
| `DeviceNoiseProfile.from_json()` | Load custom calibration JSON |
| `BitFlipChannel` | Composable bit-flip channel |
| `PhaseFlipChannel` | Composable phase-flip channel |
| `DepolarizingChannel` | Composable depolarizing channel |
| `AmplitudeDampingChannel` | Composable amplitude damping (Kraus) |
| `PhaseDampingChannel` | Composable phase damping (Kraus) |
| `ReadoutErrorChannel` | Composable readout error |

```python
from src.noise import T1T2NoiseModel, CrosstalkModel, NoisePipeline

# Compose T1/T2 relaxation + crosstalk
pipeline = NoisePipeline()
pipeline.add_channel(T1T2NoiseModel(t1=120.0, t2=80.0))  # IBM Sherbrooke-like
pipeline.add_channel(CrosstalkModel(crosstalk_prob=0.001))
```

---

## Quantum Routing & Hardware Topologies

### Routers

| Router | Algorithm | Use Case |
|---|---|---|
| `BasicSwapRouter` | Shortest-path SWAP | Simple, deterministic |
| `GreedyRouter` | Look-ahead heuristic | Moderate circuits |
| `SabreRouter` | Structure-Aware Bidirectional | Production, real hardware |

### Coupling Maps

| Topology | Method | Qubits |
|---|---|---|
| `CouplingMap.linear(n)` | Linear chain | n |
| `CouplingMap.grid(rows, cols)` | 2D grid | rows × cols |
| `CouplingMap.heavy_hex(d)` | **Real IBM heavy-hex** | ~d × (2d-1) |
| `CouplingMap.ibm_eagle()` | **IBM Eagle (127q)** | 127 |
| `CouplingMap.ibm_condor()` | **IBM Condor (1121q)** | ~1121 |
| `CouplingMap.ionq_alltoall(n)` | **IonQ all-to-all** | n |
| `CouplingMap.rigetti_ring(n)` | **Rigetti ring** | n |
| `CouplingMap.google_sycamore(r, c)` | **Google Sycamore grid** | r × c |

```python
from src.routing.router import CouplingMap, SabreRouter

# Route onto real IBM Eagle 127-qubit processor
coupling = CouplingMap.ibm_eagle()
router = SabreRouter(coupling)
result = router.route(circuit_ops, logical_qubits)
print(f"SWAP count: {result.swap_count}")
print(f"Initial mapping: {result.initial_mapping}")
```

---

## JIT Compiler & Fast-Loop Engine

### Fast-Loop JIT

The VM automatically detects tight backward-jump loops and compiles them to **register-based Python code** — bypassing the stack machine entirely:

1. **Loop Detection**: Backward `JMP` patterns identified at execution start
2. **Register Allocation**: Variables mapped to Python locals (no dict lookups)
3. **Python `while` Generation**: Emits native `while condition:` loops
4. **Sandboxed Execution**: Only safe builtins (`type`, `repr`, `int`, etc.)

This delivers **1.8-2x speedup over CPython** for tight numerical loops.

### Trace-Based Adaptive JIT v2

For general bytecode:
1. **Loop-Invariant Code Motion (LICM)**: Hoists invariant expressions
2. **Constant Folding**: Collapses known constant sub-expressions
3. **Type Guards**: Specialized execution with deoptimization fallback
4. **Local Variable Caching**: Read-write variables cached in Python locals
5. **Function Inlining**: Simple functions inlined into hot blocks
6. **Configurable Hot Threshold**: `hot_threshold` parameter controls when JIT kicks in

---

## Formal Verification & ZX-Calculus

### Equivalence Checking

1. **Fast-Reject**: Canonical hash comparison (necessary but NOT sufficient)
2. **Full Proof (N ≤ 8)**: Unitary matrix construction and comparison (1e-9 tolerance)
3. **ZX-Calculus (N > 8)**: Graph reduction with:
   - Spider fusion, identity removal, local complementation
   - Pivoting, bialgebra and Hopf rules
   - Hadamard edges

> **Important:** Canonical hash equality is a necessary but NOT sufficient condition for circuit equivalence. Use `are_equivalent()` for definitive proof.

### Stabilizer Compatibility Checking

```python
from src.stabilizer_simulator import StabilizerSimulator

gates = [('T', ['q0'], []), ('CNOT', ['q0', 'q1'], [])]
incompatible = StabilizerSimulator.check_circuit_compatibility(gates)
if incompatible:
    print(f"Non-Clifford gates: {incompatible}")
    # Switch to dense backend
```

---

## Example Codebases

### Quantum Fourier Transform

```eigen
eigen 2.6
module quantum.qft

qubit q0
qubit q1
qubit q2

H q0
RZ q0, 1.57079632679
CNOT q1, q0
RZ q0, 0.78539816339
CNOT q2, q0

H q1
RZ q1, 1.57079632679
CNOT q2, q1

H q2
SWAP q0, q2
```

### Grover's Algorithm

```eigen
eigen 2.6
module quantum.grover

qubit q0
qubit q1
cbit c0
cbit c1

H q0
H q1

H q1
CNOT q0, q1
H q1

H q0
H q1
X q0
X q1
H q1
CNOT q0, q1
H q1
X q0
X q1
H q0
H q1

measure q0 -> c0
measure q1 -> c1
```

### Bell State with Noise

```eigen
eigen 2.6

qubit q0
qubit q1
cbit c0
cbit c1

H q0
CNOT q0, q1

noise depolarizing(0.02) q0
noise depolarizing(0.02) q1

measure q0 -> c0
measure q1 -> c1

print c0
print c1
```

```bash
eigen run bell_noise.eig --noise depolarizing --noise-prob 0.02
```

---

## Language Documentation

### Types

| Type | Description | Example |
|---|---|---|
| `qubit` | Quantum bit (linear, non-copyable) | `qubit q0` |
| `cbit` | Classical bit (0 or 1) | `cbit c0` |
| `int` | 64-bit signed integer | `let x: int = 42` |
| `float` | Double-precision float | `let x: float = 3.14` |
| `string` | UTF-8 string | `let s: string = "hello"` |
| `bool` | Boolean | `let b: bool = true` |
| `array<T>` | Dynamic array | `let a: array<int> = [1, 2, 3]` |
| `map<K, V>` | Associative map | `let m: map<string, int> = {"a": 1}` |
| `struct` | User-defined structure | `struct Point { x: float, y: float }` |
| `enum` | Enumeration | `enum Color { Red, Green, Blue }` |

### Operators

| Category | Operators |
|---|---|
| Arithmetic | `+ - * / % **` |
| Comparison | `== != < > <= >=` |
| Logical | `and or not` |
| Bitwise | `& \| ^ ~ << >>` |
| Assignment | `= += -= *= /=` |

### Literals

| Type | Syntax | Example |
|---|---|---|
| Integer | decimal | `42` |
| Hex | `0x` prefix | `0xFF` → 255 |
| Binary | `0b` prefix | `0b1010` → 10 |
| Octal | `0o` prefix | `0o77` → 63 |
| Float | decimal point | `3.14159` |
| Scientific | `e` notation | `1.23e-5` |
| String | double quotes | `"hello\nworld"` |
| Interpolated | `${expr}` | `"Result: ${x}"` |
| Boolean | `true` / `false` | `true` |

### Quantum Gates

| Gate | Qubits | Arguments | Clifford? | Description |
|---|---|---|---|---|
| `H` | 1 | — | Yes | Hadamard |
| `X` `Y` `Z` | 1 | — | Yes | Pauli gates |
| `S` | 1 | — | Yes | Phase gate |
| `T` | 1 | — | No | π/8 gate |
| `RX` `RY` `RZ` | 1 | angle | No | Rotation gates |
| `CNOT` `CZ` | 2 | — | Yes | Controlled gates |
| `SWAP` | 2 | — | Yes | Swap gate |
| `CCX` | 3 | — | No | Toffoli gate |
| `CSWAP` | 3 | — | No | Fredkin gate |
| `CP` `CRX` `CRY` `CRZ` | 2 | angle | No | Controlled rotations |

### Control Flow

```eigen
// If/else/elif
if x == 5 { ... } else if x == 3 { ... } else { ... }

// While loop
while x > 0 { ... }

// For loop
for item in array { ... }

// Match/case (pattern matching)
match expr {
    case 0 { ... }
    case 42 { ... }
    default { ... }
}

// Try/catch
try { ... } catch (e) { ... }

// Break/Continue
break
continue
```

### Functions

```eigen
// Classic function with return type
func add(a: int, b: int) -> int {
    return a + b
}

// Void function (no return type needed)
func log(msg: string) {
    print msg
}

// Quantum function
qfunc bell(q0: qubit, q1: qubit) {
    H q0
    CNOT q0, q1
}
```

### Structs

```eigen
struct Point {
    x: float,
    y: float
}

let p: Point = Point { x: 1.0, y: 2.0 }
print p.x
```

### Parallel Execution

```eigen
parallel {
    task compute_a(x)
    task compute_b(y)
}
```

---

## Standard Library

| Module | Functions |
|---|---|
| `std.math` | `sin`, `cos`, `tan`, `sqrt`, `log`, `exp`, `abs` |
| `std.io` | `read_file`, `write_file`, `print_format` |
| `std.collections` | `append_int`, `remove_at` |
| `std.random` | `rand_float`, `rand_int` |
| `std.stats` | `mean`, `variance` |
| `std.string` | `concat`, `format_int` |
| `std.time` | `now`, `sleep` |

```eigen
import std.math

let val: float = std.math.sqrt(16.0)
print "sqrt(16) = ${val}"
```

---

## Architecture & Compiler Pipeline

```
[ Eigen Source (.eig) ]
       │
       ▼
    [ AST ]  ←── Zero-copy Pratt parser (Rust)
       │
       ▼
   [ MLIR ]  ── Dialects: func, arith, quantum, cf
       │
       ▼
   [ EQIR ]  ── Quantum DAG IR (gate dependency graph)
   ├─── Optimizer Passes (gate fusion, ZX rewrites, CNOT cancellation)
   └─── Equivalence Verification (ZX-calculus & unitary matrix)
       │
       ├────────────────────┐
       ▼                    ▼
[ EBC Bytecode ]     [ LLVM IR / QIR ]
       │                    │
       ▼                    ▼
[ Eigen VM / JIT ]  [ Native Binary (AOT) ]
```

### Key Components

| Component | Lines | Language | Description |
|---|---|---|---|
| Lexer/Parser | ~8,000 | Rust + Python | Zero-copy Pratt-precedence parser |
| EBC Compiler | ~29,000 | Python | AST → EBC bytecode v2 with SSA |
| VM | ~50,000 | Python | Integer dispatch, hardened, JIT |
| Dense Simulator | ~1,000 | Rust + Python | NumPy + Rust state-vector backend |
| Sparse Simulator | ~17,000 | Python | Honest sparsity scaling |
| Stabilizer Simulator | ~250 | Python | CHP algorithm, O(n²), auto-fallback |
| MPS Simulator | ~450 | Python | Auto bond dim, truncation warnings |
| Noise Engine | ~600 | Python | T1/T2, crosstalk, device profiles |
| Routing (SABRE) | ~500 | Python | 3 routers, real device topologies |
| ZX-Calculus | ~3,000 | Python | Spider fusion, pivoting, H-edges |
| LLVM/QIR AOT | ~38,000 | Python | llvmlite-based, QIR-compliant |
| GPU Engine | ~130 | Python | CUDA/ROCm/Metal, structured logging |

---

## Migration Guide

See [MIGRATION.md](MIGRATION.md) for the full 2.5 → 2.6 migration guide.

### Quick Checklist

- [ ] Update code that catches `ValueError` from stabilizer to handle `NonCliffordGateError`
- [ ] Replace `CouplingMap.heavy_hex(n)` with `CouplingMap.grid(n, n)` if grid behavior needed
- [ ] Configure `eigen.gpu` logger if you depended on `print()` from GPU engine
- [ ] Review MPS `max_bond_dim` (default changed from 32 to 64)
- [ ] Update version references from "Nova" to "Misery"

---

## Roadmap

### Completed in 2.6 «Misery»

- [x] Inno Setup Windows GUI installer
- [x] Real IBM heavy-hex topology + device topologies (Eagle, Condor, IonQ, Rigetti, Google)
- [x] Advanced noise engine (T1/T2, crosstalk, device profiles)
- [x] Stabilizer pre-flight check + auto-fallback
- [x] MPS auto bond dimension + truncation warnings
- [x] Structured GPU logging
- [x] Version synchronization (pyproject, CLI, Cargo)
- [x] Coverage includes critical components
- [x] Migration guide (MIGRATION.md)
- [x] Configurable JIT/router magic values

### Completed in 2.5 «Mitz»

- [x] Fast-Loop JIT (1.8x faster than CPython)
- [x] Stabilizer Simulator (1000+ qubits)
- [x] SABRE Quantum Routing
- [x] Pattern Matching (`match/case`)
- [x] String Interpolation (`"${val}"`)
- [x] Hex/Binary/Octal/Scientific literals
- [x] Exponentiation operator (`**`)
- [x] Void functions
- [x] Full Kraus noise channels
- [x] Parser error recovery
- [x] HTML benchmark dashboard
- [x] 12 critical/high bug fixes

### Planned for 2.7+

- [ ] Cranelift JIT backend (native machine code generation)
- [ ] CUDA state-vector kernels (cuStateVec)
- [ ] Vulkan Compute shaders (cross-platform GPU)
- [ ] `eigen repl` — interactive REPL
- [ ] `eigen new quantum` — project scaffolding
- [ ] `eigen upgrade` — self-update mechanism
- [ ] VS Code Extension with full LSP
- [ ] Parametrized circuits (`circuit(theta)`)
- [ ] `async/await` for IO-bound quantum cloud jobs
- [ ] `enum` types with pattern matching integration
- [ ] `trait` / interfaces for generic quantum backends
- [ ] Lambda expressions (`let f = (x) => x * 2`)
- [ ] Documentation website (mkdocs)
- [ ] Package registry (registry.eigen-lang.org)
- [ ] Error mitigation (ZNE, PEC, M3 readout correction)
- [ ] Variational algorithms (VQE, QAOA parameter optimization)

---

## Community & Contributing

- **Research Paper:** [Eigen: A Runtime-First Hybrid Classical-Quantum Programming Language](https://drive.google.com/file/d/1YovOIB1VdZUUUfsKQw4WcI6BL_Ekjh2O/view?usp=drivesdk)
- **License:** [MIT License](LICENSE) — free for all use
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
- **Code of Conduct:** See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Issues:** [GitHub Issues](https://github.com/Eigenresearch/Eigen/issues)
- **Releases:** [GitHub Releases](https://github.com/Eigenresearch/Eigen/releases)

### Built With

- [PyO3](https://pyo3.rs) — Python ↔ Rust FFI
- [NumPy](https://numpy.org) — Numerical computations
- [Rayon](https://github.com/rayon-rs/rayon) — Data parallelism in Rust
- [CuPy](https://cupy.dev) / [PyTorch](https://pytorch.org) — GPU acceleration
- [llvmlite](https://github.com/numba/llvmlite) — LLVM IR generation
- [Inno Setup](https://jrsoftware.org/isinfo.php) — Windows installer

### Test Results

```
449 tests passed
439 subtests passed
0 failures
0 regressions
```

### Codebase Statistics

- ~550 files
- ~120,000 lines of code
- 6 quantum simulator backends
- 3 quantum routers
- 5+ noise channels + 3 advanced noise models
- 7 standard library modules
- Full EBNF grammar specification
- LLVM/QIR AOT compilation
- ZX-calculus formal verification

---

<p align="center">
  <strong>Eigen 2.6 «Misery»</strong><br>
  <em>Faster. Harder. Less Python. More Quantum.</em><br><br>
  <a href="https://github.com/Eigenresearch/Eigen">GitHub</a> •
  <a href="https://github.com/Eigenresearch/Eigen/releases">Releases</a> •
  <a href="https://github.com/Eigenresearch/Eigen/issues">Issues</a> •
  <a href="LANGUAGE.md">Language Spec</a> •
  <a href="MIGRATION.md">Migration Guide</a> •
  <a href="https://drive.google.com/file/d/1YovOIB1VdZUUUfsKQw4WcI6BL_Ekjh2O/view?usp=drivesdk">Research Paper</a>
</p>
