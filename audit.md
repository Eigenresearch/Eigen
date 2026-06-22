# Eigen Project Audit

This document presents a technical audit of the Eigen 2.1 language, compiler frontend, VM execution engine, DAG-based intermediate representation, and state-vector simulator.

---

## 1. Architectural Strengths

1. **Dual-Execution Engines**: 
   Eigen 2.1 decouples quantum circuit optimization from complex classical control flows by offering:
   - **Eigen VM**: A stack-based execution environment running **Eigen Bytecode (EBC)**. This natively supports recursion, try-catch exceptions, dynamic heap allocations (structs, arrays, maps), and noise channels.
   - **topological Runtime**: A Directed Acyclic Graph (**EQIR v1.1**) scheduler optimized for pure quantum gate sequences and circuit equivalence checking.
2. **Dedicated Diagnostic Engine**:
   Rather than emitting untyped Python warnings, compilation/transpilation issues are collected via a structured `DiagnosticEngine`. This makes error and warning reports easily integrated into CLI, LSP servers, and IDE tools.
3. **Backend Capability Safeguards**:
   The `BackendCapabilities` layer blocks invalid code emission during transpilation. Features unsupported by target backends (like Qiskit) are commented out safely in the transpiler output with warnings emitted, preventing syntax errors in output scripts.
4. **Cbit & Int Type Coercion**:
   Type compatibility rules have been relaxed to allow comparisons and assignments between `cbit` and `int` values, simplifying hybrid classical-quantum control logic.

---

## 2. Technical Audit and Scope Boundaries

1. **State-Vector Simulation Boundaries**:
   The wavefunction simulator represents state amplitudes explicitly as a \(2^N\) complex vector.
   - Per-gate execution time: \(O(2^N)\).
   - Memory usage: Scales as \(2^N\) floats. This restricts classical simulation to roughly \(N \le 25\) qubits on standard workstations.
2. **Equivalence Checking Matrix Blowup**:
   Equivalence is checked by generating complete unitary matrices column-by-column.
   - Matrix size: \(2^{2N}\) complex numbers.
   - Maximum qubit size limit is strictly capped at **8 qubits** to prevent memory exhausts.
3. **Target Exporter Limitations**:
   While the **Eigen VM** supports 100% of language constructs, external backends (like Qiskit) can only represent a subset. Structs, maps, recursion, and try-catch exceptions cannot be represented in Qiskit Aer circuits. The compiler handles this subset difference gracefully via capability profiles.

---

## 3. Runtime Guarantees

Every language construct—recursive functions, loops, structures, arrays, maps, and exception catch blocks—is executed natively by the Eigen VM. Classical execution is considered the source of truth, whereas backend exporters (like the Qiskit backend) are optional compatibility targets.

---

## 4. Backend Compatibility Matrix

| Feature / Subsystem | Eigen VM Target | topological Runtime | Qiskit Backend |
| --- | --- | --- | --- |
| Quantum Gates & Measures | `FULL` | `FULL` | `FULL` |
| Noise Channels | `FULL` | `NONE` | `NONE` |
| Structs / Maps Allocation | `FULL` | `NONE` | `NONE` (Transpiler Warning) |
| Recursion (Call Stack) | `FULL` | `NONE` | `NONE` (Transpiler Warning) |
| Exceptions (Try-Catch) | `FULL` | `NONE` | `NONE` (Transpiler Warning) |
| Dynamic Loops | `FULL` | `NONE` | `NONE` (Transpiler Warning) |

---

## 5. Future Architectural Improvements

1. **Compile Simulator Core to Rust/C++**:
   Porting state vector operations to Rust or C++ would bypass Python loop overhead and enable multi-threading.
2. **Algebraic Decision Diagrams**:
   Replacing full-unitary matrix comparison in the Equivalence Checker with Quantum Decision Diagrams (QDDs) to scale equivalence checks to 50+ qubits.
3. **OpenQASM Target Backend**:
   Developing an exporter targeting OpenQASM 2.0/3.0 to run compiled gates directly on physical quantum hardware.
