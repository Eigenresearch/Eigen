# Eigen Project Audit: Helios Maturity Release (v2.3)

This document presents a technical audit of the **Eigen 2.3 — Helios** language runtime, trace JIT compiler, SSA IR builder, simulation engines, and verification subsystems.

---

## 1. Architectural Strengths

1. **Multi-Tiered Execution Architecture**:
   Eigen 2.3 decouples quantum compilation, optimization, and simulation by offering:
   - **Eigen VM with Trace JIT**: A stack-based execution environment running **Eigen Bytecode (EBC v3)**. Frequently executed bytecode blocks are traced and dynamically compiled to native Python functions using `compile()`, providing 2x-5x execution acceleration.
   - **Static Single Assignment (SSA) IR**: Builds control flow graphs (CFGs), computes dominator frontiers, and places $\phi$ (phi) nodes, establishing a standard compiler research pipeline.
   - **Topological Runtime**: A Directed Acyclic Graph (**EQIR v1.1**) scheduler optimized for gate-sequence optimization and transpilation.
2. **Diverse Simulation Engines**:
   - **Dense Wavefunction**: Ideal for dense, high-entanglement state computations up to $\le 20$ qubits.
   - **Sparse Dictionary**: `SparseQuantumSimulator` efficiently supports large sparse quantum states. Practical limits depend on circuit structure. Some sparse workloads may exceed 100 qubits, while dense superposition-heavy circuits remain exponentially expensive.
   - **Matrix Product State (MPS)**: Utilizes 1D tensor chain structures and SVD-based singular value truncation to simulate low-entanglement circuits up to 100+ qubits.
3. **Formal Verification & ZX-Calculus**:
   - **Unitary Equivalence**: Restricts full-matrix comparison to $N \le 8$ qubits to avoid exponential memory growth.
   - **ZX-Calculus Engine**: Represents circuit wires as spiders (Z, X) and H-boxes. Performs spider fusion, local complementation, pivoting, and Clifford reductions to prove equivalence without matrix representation.
4. **Backend Capability Enforcement**:
   - Compiling with `eigen audit --strict` or `--strict` flags enforces strict capability alignment. If the target backend (e.g. Qiskit) does not support a language construct (like structs, recursion, or maps), the build fails with an error rather than emitting a warning.

---

## 2. Technical Audit and Scope Boundaries

### 2.1 Simulation Boundaries
1. **Wavefunction Vector Simulator**:
   - Per-gate execution time: $\mathcal{O}(2^N)$.
   - Memory usage: Scales as $\mathcal{O}(2^N)$ complex numbers, restricting execution to $N \le 25$ qubits on standard workstations.
2. **Sparse Quantum Simulator**:
   - Efficiently represents states with low active components using key-value maps.
   - For highly entangled or dense superposition states (e.g. $H^{\otimes N}$), the active state count still grows as $2^N$, causing exponential memory and time complexity.
3. **MPS Tensor Network Simulator**:
   - Approximation accuracy depends on the maximum bond dimension ($\chi$).
   - If entanglement grows rapidly (e.g. via deep entangling gates spread across the chain), SVD truncation will introduce approximation errors, or require exponential bond dimension scaling.

### 2.2 Equivalence Checking Boundaries
1. **Unitary Matrix Equivalence**:
   - Requires generating complete unitary matrices of size $2^{2N}$ complex floats.
   - Cap is strictly set at **8 qubits** to prevent out-of-memory crashes.
2. **ZX-Calculus Equivalence**:
   - Performs heuristic rule reductions. Useful for Clifford and Clifford+T circuits.
   - For arbitrary non-Clifford circuits, the reduction might not simplify to a trivial identity, making it a powerful but not complete solver for all circuits.
   - **Equivalence vs. Hashing**: Canonical hashes of circuits do not guarantee equivalence. Different gate combinations (such as $H X H$ and $Z$) can represent equivalent unitaries. The equivalence checker validates equivalence using either ZX reduction rules or direct matrix comparison, not syntactic hash matching.

### 2.3 VM & JIT Compiler Boundaries
- Traced loop blocks are compiled directly. Return addresses for `Opcode.CALL` inside JIT-compiled basic blocks are synchronized before invocation to prevent infinite loop errors.

---

## 3. Runtime Guarantees

Every language construct—recursive functions, loops, structures, arrays, maps, and exception catch blocks—is executed natively by the Eigen VM. Classical execution is considered the source of truth, whereas backend exporters (like the Qiskit backend) are optional compatibility targets.

---

## 4. Backend Capability Matrix

| Feature / Subsystem | Eigen VM (JIT) | topological Runtime | Qiskit Backend | IonQ / AWS Braket |
| :--- | :---: | :---: | :---: | :---: |
| **Quantum Gates & Measures** | `FULL` | `FULL` | `FULL` | `FULL` |
| **Noise Channels** | `FULL` | `NONE` | `NONE` | `NONE` |
| **JIT Optimization** | `FULL` | `NONE` | `NONE` | `NONE` |
| **Structs & Maps** | `FULL` | `NONE` | `NONE` (Strict Error) | `NONE` (Strict Error) |
| **Recursion (Call Stack)** | `FULL` | `NONE` | `NONE` (Strict Error) | `NONE` (Strict Error) |
| **Exceptions (Try-Catch)** | `FULL` | `NONE` | `NONE` (Strict Error) | `NONE` (Strict Error) |
| **Dynamic Loops** | `FULL` | `NONE` | `NONE` (Strict Error) | `NONE` (Strict Error) |

---

## 5. Future Architectural Improvements

1. **C++ Accelerator Core**:
   Porting state vector and tensor network operations to C++/CUDA to bypass Python loop overheads.
2. **Unified QIR Exporter**:
   Developing a full LLVM-based Quantum Intermediate Representation (QIR) generation pass for direct hardware integration.
