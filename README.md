# Eigen Programming Language — Release 2.5 «Mitz»

[![CI Build](https://github.com/Eigenresearch/Eigen/actions/workflows/release.yml/badge.svg)](https://github.com/Eigenresearch/Eigen/actions)
[![Release Version](https://img.shields.io/badge/release-2.5.0--Mitz-blue.svg)](https://github.com/Eigenresearch/Eigen/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-445%20passed-brightgreen.svg)](tests/)

> **«Faster. Harder. Less Python.»** — Release 2.5 «Mitz» delivers a **1.8x speedup over CPython** for classical loops, a **stabilizer simulator** that handles **1000+ qubits** in seconds, 8 new language features, 12 critical bug fixes, full Kraus-operator noise channels, SABRE quantum routing, pattern matching, string interpolation, and a comprehensive performance overhaul.

---

## Table of Contents

1. [What's New in 2.5 «Mitz»](#whats-new-in-25-mitz)
2. [Performance Benchmarks](#performance-benchmarks)
3. [Why Eigen?](#why-eigen)
4. [Language Feature Matrix](#language-feature-matrix)
5. [CLI Manual](#cli-manual)
6. [Simulation Engines](#simulation-engines)
7. [JIT Compiler & Fast-Loop Engine](#jit-compiler--fast-loop-engine)
8. [Formal Verification & ZX-Calculus](#formal-verification--zx-calculus)
9. [Installation](#installation)
10. [Quick Start](#quick-start)
11. [Example Codebases](#example-codebases)
12. [Language Documentation](#language-documentation)
13. [Roadmap](#roadmap)

---

## What's New in 2.5 «Mitz»

### Performance Breakthroughs

| Improvement | Before (2.4) | After (2.5) | Gain |
|---|---|---|---|
| **100K classical loop** | 0.71s | **0.006s** | **120x faster** |
| **1M classical loop** | ~7s | **0.059s** | **120x faster** |
| **10M classical loop** | ~70s | **0.65s** | **100x faster** |
| **vs CPython** | 44x slower | **1.8x faster** | **80x improvement** |
| **1000-qubit Clifford** | impossible | **4.1s** | New capability |
| **Print precision** | `1e-06` | `0.000001` | Fixed |

### New Language Features (8)

| Feature | Syntax | Description |
|---|---|---|
| **Exponentiation `**`** | `2 ** 3` | Full support across lexer, parser, VM, JIT, type checker |
| **Hex/Binary/Octal** | `0xFF`, `0b1010`, `0o77` | Low-level literal constants |
| **Scientific notation** | `1.23e-5` | Physical constants in code |
| **String interpolation** | `"Result: ${val}"` | No more concatenation |
| **match/case** | `match x { case 1 { ... } }` | Pattern matching with default |
| **Void functions** | `func foo() { ... }` | No mandatory `-> type` |
| **Block comments** | `/* ... */` | Multi-line commenting |
| **Escape sequences** | `\n \t \r \0 \\` | Proper string escape handling |

### Critical Bug Fixes (12)

| Bug | Severity | Fix |
|---|---|---|
| JIT `exec()` RCE vulnerability | CRITICAL | Sandboxed builtins — no `os`, `sys`, `__import__` |
| `eval()` code injection | CRITICAL | Replaced with AST-based safe evaluator |
| Controlled rotation crash | CRITICAL | CP/CRX/CRY/CRZ now extract angles properly |
| `op_ret` stack underflow | CRITICAL | Guard: push `None` if stack empty |
| try/catch at top level | HIGH | VM-level try stack (not frame-level) |
| Canonical hash order | CRITICAL | Topological order + qubit targets |
| Compound assignment double-eval | HIGH | Temp variable prevents side-effect duplication |
| IR converter drops nodes | CRITICAL | All AST node types handled |
| Optimizer blocked by VarDecl | CRITICAL | Excluded from control-flow check |
| Amplitude damping incorrect | HIGH | Proper Kraus K0/K1 operators |
| Phase damping missing | HIGH | Added Kraus E0/E1 operators |
| Print rounding errors | HIGH | Full-precision float formatting |

### New Capabilities

- **Stabilizer Simulator** — O(n²) Clifford circuit simulation, handles 1000+ qubits
- **SABRE Quantum Routing** — Hardware-aware SWAP insertion for real processors
- **Parser Error Recovery** — Multiple syntax errors per compilation run
- **HTML Benchmark Dashboard** — `eigen bench --html` generates visual reports
- **Shared Gate Registry** — Centralized gate metadata for VM, runtime, simulator
- **Bytecode Version Validation** — Prevents loading incompatible bytecode
- **ZX Hadamard Edges** — Proper H-edge support in ZX-calculus graphs

---

## Performance Benchmarks

### Classical Performance — Eigen vs CPython

```
100K iterations:  Eigen=0.006s   Python=0.011s   → 1.8x FASTER
  1M iterations:  Eigen=0.059s   Python=0.110s   → 1.9x FASTER
 10M iterations:  Eigen=0.650s   Python=1.200s   → 1.8x FASTER
```

### Quantum Simulation

| Qubits | Dense (Rust) | Stabilizer |
|---|---|---|
| 10 | 0.0005s | 0.001s |
| 20 | 0.13s | 0.004s |
| 100 | — | 0.009s |
| 500 | — | 0.65s |
| 1000 | — | 4.1s |

### AOT vs VM (from 2.4 baseline)

| Program | VM (ms) | AOT (ms) | Speedup |
|---|---|---|---|
| fib(22) classical | 411.89 | 22.84 | **18x** |
| Bell pair (500 shots) | 125.52 | 9.26 | **13.5x** |
| factorial(12) x 10000 | 917.54 | 20.80 | **44x** |

---

## Why Eigen?

Quantum computing developer tools are historically split into host-language SDKs (Qiskit, Pennylane) and low-level hardware formats (OpenQASM). Eigen is a **standalone, domain-specific, hybrid classical-quantum language** with native runtime guarantees.

* **Unlike Qiskit**: Qiskit is a Python library — type checking and circuit optimization happen in Python's runtime. Eigen is a compiled language with static type verification, modular namespaces, and compilation to EBC bytecode or LLVM IR.
* **Unlike OpenQASM 3.0**: OpenQASM targets low-level hardware control. Eigen supports recursion, user-defined structs, associative maps, dynamic arrays, and structured try-catch exceptions.
* **Unlike Silq**: Silq uses AST-based type safety for uncomputation. Eigen optimizes at the IR level via CFG, SSA, MLIR dialects, and EQIR DAG gate dependency analysis.
* **Unlike Q#**: Q# relies on .NET/LLVM stack. Eigen is lightweight, compiling to compact stack bytecode executed via a portable VM with adaptive JIT, or native Rust library.

---

## Language Feature Matrix

| Feature | Eigen 2.5 Mitz | Qiskit | OpenQASM 3 | Silq | Q# |
|---|---|---|---|---|---|
| **Execution Model** | VM / Rust FFI / LLVM / AOT | Python | Hardware | Native | VM / LLVM |
| **Classical State** | Full (recursion, exceptions, structs, maps, arrays) | Limited | Static | Limited | Dynamic |
| **Simulators** | Dense, Sparse, MPS, **Stabilizer**, GPU | Aer | Dep. | Wavefunction | Sparse |
| **JIT Compiler** | Yes (fast-loop, LICM, type guards, **register-based**) | No | No | No | No |
| **Pattern Matching** | Yes (`match/case`) | No | No | No | No |
| **String Interpolation** | Yes (`"${val}"`) | Python | No | No | No |
| **Noise Channels** | Kraus operators (amplitude/phase damping) | Yes | No | No | No |
| **Quantum Routing** | SABRE + Greedy + Basic | Yes (via transpiler) | No | No | No |
| **Verification** | Unitary + ZX-Calculus (H-edges) | Library | None | Uncomputation | None |
| **IR Architecture** | AST → MLIR → EQIR DAG → SSA | DAGCircuit | AST | AST | QIR |
| **Package Manager** | `eigen.lock` verification | pip | None | None | nuget |
| **Exporters** | IBM QASM, IonQ, Braket, QIR, QASM3, Quil | Qiskit | Export | None | QIR |

---

## CLI Manual

```bash
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

eigen build <file.eig>        # Compile to bytecode/native
  --llvm                      # LLVM IR output
  --qir                       # QIR-compliant LLVM IR
  --aot                       # Standalone native binary
  --opt-level <O0-O3>         # LLVM optimization
  --lto                       # Link-Time Optimization
  --strip                     # Strip debug symbols

eigen bench                   # Benchmark suite
  --frontend                  # Python vs Rust parser comparison
  --html                      # Generate HTML dashboard

eigen verify <file.eig>       # Syntax + semantic verification
eigen verify-equiv <f1> <f2>  # Circuit equivalence checking
  --method <unitary|zx>       # Verification method

eigen profile <file.eig>      # Execution profiler
eigen fmt <file.eig>          # Auto-format source code
eigen doc                     # Generate documentation
eigen test                    # Run project tests
eigen doctor                  # Toolchain health check
eigen lsp                     # Language Server Protocol daemon
```

---

## Simulation Engines

| Engine | Complexity | Best For | Max Qubits |
|---|---|---|---|
| **Dense (Rust)** | O(2ⁿ) | General circuits | ~25 |
| **Sparse** | O(sparsity) | Low-weight gates | 50+ |
| **MPS Tensor Network** | O(n·χ²) | Low-entanglement | 100+ |
| **Stabilizer** | O(n²) | Clifford circuits | **1000+** |
| **Density Matrix** | O(4ⁿ) | Noise channels | ~12 |
| **GPU** | O(2ⁿ) | GPU-accelerated | GPU memory |

### Noise Models

| Channel | Type | Implementation |
|---|---|---|
| Bit flip | Stochastic | X gate with probability p |
| Phase flip | Stochastic | Z gate with probability p |
| Depolarizing | Stochastic | Random X/Y/Z with probability p |
| Amplitude damping | Kraus | K0=diag(1,√(1-p)), K1=√p·|0⟩⟨1| |
| Phase damping | Kraus | E0=diag(1,√(1-λ)), E1=√λ·|1⟩⟨1| |
| Readout error | Stochastic | Flip measurement outcome with probability p |

---

## JIT Compiler & Fast-Loop Engine

### Fast-Loop JIT (NEW in 2.5)

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

---

## Formal Verification & ZX-Calculus

### Equivalence Checking

1. **Fast-Reject (N ≤ 8)**: Full unitary matrix construction and comparison
2. **ZX-Calculus (N > 8)**: Graph reduction with:
   - Spider fusion, identity removal, local complementation
   - Pivoting, bialgebra and Hopf rules
   - **Hadamard edges** (NEW in 2.5)

### SABRE Quantum Routing

Routes circuits onto hardware coupling maps:
- **BasicSwapRouter**: Shortest-path SWAP insertion
- **GreedyRouter**: Look-ahead heuristic minimization
- **SabreRouter**: Structure-Aware Bidirectional routing with front-layer analysis

```python
from src.routing.router import SabreRouter, CouplingMap
coupling = CouplingMap.linear(7)  # IBM-style linear topology
router = SabreRouter(coupling)
result = router.route(circuit_ops, logical_qubits)
```

---

## Installation

### Windows (Setup Wizard)

1. Download `Eigen-2.5-Windows-x64-Setup.exe` from [Releases](https://github.com/Eigenresearch/Eigen/releases)
2. Run the installer — it will guide you through:
   - Python 3.10+ detection
   - Rust toolchain check
   - Native module compilation
   - PATH configuration
3. Verify: `eigen doctor`

### macOS

```bash
# Install via pip
pip install eigen-lang

# Or from source
git clone https://github.com/Eigenresearch/Eigen.git
cd Eigen
pip install -e .
cd native/rust && maturin develop --release
```

### Linux

```bash
# Install via pip
pip install eigen-lang

# Or from source
git clone https://github.com/Eigenresearch/Eigen.git
cd Eigen
pip install -e .
cd native/rust && maturin develop --release
```

### Verify Installation

```bash
eigen doctor          # Check toolchain health
eigen test            # Run test suite (445 tests)
eigen run examples/bell.eig --trace  # Smoke test
```

---

## Quick Start

### Hello Quantum

```eigen
eigen 2.5

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
```

### Classical + Quantum Hybrid

```eigen
eigen 2.5

func factorial(n: int) -> int {
    if n == 0 {
        return 1
    }
    return n * factorial(n - 1)
}

qubit q0
H q0
RZ q0, 3.141592653589793 ** 0.5  # pi ** 0.5

let result: int = factorial(10)
print result
```

### Pattern Matching

```eigen
eigen 2.5

func classify(x: int) -> string {
    match x {
        case 0 { return "zero" }
        case 1 { return "one" }
        case 42 { return "answer" }
        default { return "other" }
    }
}
```

### String Interpolation

```eigen
eigen 2.5

let name: string = "Eigen"
let version: float = 2.5
print "Running ${name} v${version}"
```

### Stabilizer Simulator (1000 qubits)

```eigen
eigen 2.5

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
```

---

## Example Codebases

### Quantum Fourier Transform

```eigen
eigen 2.5
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
eigen 2.5
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

---

## Language Documentation

### Types

| Type | Description | Example |
|---|---|---|
| `qubit` | Quantum bit | `qubit q0` |
| `cbit` | Classical bit | `cbit c0` |
| `int` | Integer | `let x: int = 42` |
| `float` | Floating point | `let x: float = 3.14` |
| `string` | String | `let s: string = "hello"` |
| `bool` | Boolean | `let b: bool = true` |
| `array<T>` | Dynamic array | `let a: array<int> = [1, 2, 3]` |
| `map<K, V>` | Associative map | `let m: map<string, int> = {"a": 1}` |

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

### Gates

| Gate | Qubits | Arguments | Description |
|---|---|---|---|
| `H` | 1 | — | Hadamard |
| `X` `Y` `Z` | 1 | — | Pauli gates |
| `S` `T` | 1 | — | Phase gates |
| `RX` `RY` `RZ` | 1 | angle | Rotation gates |
| `CNOT` `CZ` | 2 | — | Controlled gates |
| `SWAP` | 2 | — | Swap gate |
| `CCX` | 3 | — | Toffoli gate |
| `CSWAP` | 3 | — | Fredkin gate |
| `CP` `CRX` `CRY` `CRZ` | 2 | angle | Controlled rotations |

### Control Flow

```eigen
// If/else
if x == 5 { ... } else { ... }
elif x == 3 { ... }

// While loop
while x > 0 { ... }

// For loop
for item in array { ... }

// Match/case
match expr {
    case pattern { ... }
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

### Enums

```eigen
enum Color {
    Red,
    Green,
    Blue
}
```

### Parallel Execution

```eigen
parallel {
    task compute_a(x)
    task compute_b(y)
}
```

---

## Roadmap

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
- [x] Print precision fix

### Planned for 2.6

- [ ] VS Code Extension with LSP
- [ ] Cranelift Tier-3 JIT (native machine code)
- [ ] Parametrized circuits (`circuit(theta)`)
- [ ] Documentation website (mkdocs)
- [ ] Cross-platform GUI installer wizard

---

## License

Eigen is released under the [MIT License](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## Acknowledgments

Eigen is built with:
- [PyO3](https://pyo3.rs) — Python ↔ Rust FFI
- [NumPy](https://numpy.org) — Numerical computations
- [Rayon](https://github.com/rayon-rs/rayon) — Data parallelism in Rust
- [CuPy](https://cupy.dev) / [PyTorch](https://pytorch.org) — GPU acceleration
