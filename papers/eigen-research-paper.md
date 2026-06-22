# Eigen: A Quantum Programming Language with Graph-Based Intermediate Representation and Formal Circuit Equivalence Verification

**Author**: Eigen Development Team  
**Date**: June 2026  

---

### Abstract
We present **Eigen**, a domain-specific programming language and execution framework for quantum computing. Eigen introduces a modular language frontend with strict resource type-checking, a graph-based Intermediate Representation (EQIR v1), a high-performance state-vector simulator, and a formal circuit equivalence checker based on exact unitary matrix comparison. By representing quantum circuits as Directed Acyclic Graphs (DAGs) rather than linear instruction arrays, Eigen decouples logical program flow from physical execution schedules, enabling advanced local graph rewrites (such as gate cancellation and rotation merging) and deterministic execution scheduling. This paper describes the syntax, type system, compilation pipeline, optimization strategies, and equivalence verification math of Eigen v1.0, and evaluates its performance against classical simulation benchmarks.

---

## 1. Introduction

Quantum computing holds the promise of solving complex computational problems in fields like cryptography, chemistry, and optimization that are intractable for classical computers. However, developing, testing, and verifying quantum algorithms remains a significant challenge. Programming quantum hardware requires domain-specific languages (DSLs) that can bridge the gap between high-level algorithmic logic and the physical gate operations executed on NISQ (Noisy Intermediate-Scale Quantum) devices.

Existing quantum programming frameworks, such as IBM's Qiskit, Google's Cirq, and Microsoft's Q#, have made major strides. However, many of these frameworks treat quantum circuits as linear sequences of operations, which imposes artificial execution orderings and complicates compiler optimization and verification.

To address these challenges, we introduce **Eigen**. Eigen is a quantum programming language featuring:
1. A structured, modular syntax with version headers and namespaces.
2. A static type system that distinguishes between quantum and classical data types.
3. A graph-based Intermediate Representation (**EQIR v1**) that models quantum circuits as Directed Acyclic Graphs (DAGs) representing dataflow and sequence dependencies along qubit wires.
4. An optimization module that executes self-inverse gate cancellation and rotation merging directly on the DAG.
5. A mathematically rigorous equivalence checker that validates whether two circuits are equivalent up to a global phase (\(U_1 = e^{i\theta} U_2\)).

---

## 2. Related Work

We compare Eigen conceptually with the four dominant quantum programming frameworks in the industry:

### 2.1 Qiskit (IBM)
Qiskit is a Python-based library. While highly flexible, it acts primarily as an API rather than a distinct compiled language. Qiskit represents circuits internally as lists of instructions, and although it utilizes DAGs for transpile passes, the high-level programmer operates on imperative Python arrays, which can lead to runtime side-effects.

### 2.2 Cirq (Google)
Cirq is also a Python framework, focused primarily on NISQ-era hardware. Cirq organizes circuits into "Moments", representing slices of time where gates execute in parallel. This forces programmers to reason about physical timing constraints too early. Eigen, by contrast, abstracts timing completely through topological DAG dependencies, scheduling execution automatically.

### 2.3 Q# (Microsoft)
Q# is a standalone, compiled language with a rich type system. However, Q# compiles down to classical intermediate representations (LLVM/QIR) that are designed for quantum-classical hybrid hardware, making it heavy and complex to run locally. Eigen provides a lightweight, pure-Python stack with zero external dependencies, making it highly portable.

### 2.4 OpenQASM (AST/IBM)
OpenQASM is a low-level hardware representation language. It lacks modern language features such as modules, structured scoping, and complex variables. Eigen is a high-level language that supports modular imports and classical data structures, but can compile down to clean execution DAGs similar to OpenQASM.

---

## 3. Language Design & Syntax

Eigen is designed to be highly readable, modular, and scaleable. The language enforces a strict file structure starting with a version directive.

### 3.1 Syntax Structure
An Eigen program consists of:
- **Header**: `eigen 1.0` defining version compliance.
- **Module Declaration**: `module <path>` to establish the namespace.
- **Imports**: `import <path>` to load external subroutines.
- **Declarations**: `qubit`, `cbit`, `int`, and `float` variables.
- **Quantum Subroutines (`qfunc`)**: Pure quantum blocks containing gates.
- **Control Flow**: Conditional blocks (`if`) and assertions (`assert`).

### 3.2 Grammatical Example: Bell State
```eigen
eigen 1.0
import quantum.bell

qubit q0
qubit q1

bell(q0, q1)

cbit c0
cbit c1

measure q0 -> c0
measure q1 -> c1

assert c0 == c1
```

---

## 4. Type System & Semantic Analysis

Eigen implements a static type system to catch programming errors prior to simulation.

### 4.1 Primitive Types
- **`qubit`**: Models a physical two-level quantum system. Qubits cannot be copied (no cloning theorem), reassigned, or used in classical arithmetic.
- **`cbit`**: Models a classical bit holding values `0` or `1`.
- **`int`**: Models classical integers.
- **`float`**: Models classical double-precision floats.

### 4.2 Semantic Rules
- **Gate Safety**: A gate can only target variables of type `qubit`. Applying `H c0` (where `c0` is a `cbit`) triggers a compile-time `TypeErrorException`.
- **Measurement Constraints**: The expression `measure q -> c` requires `q` to be a `qubit` and `c` to be a `cbit`.
- **Function Call Validation**: Argument lists must match parameter types. If `qfunc bell(qubit a, qubit b)` is called as `bell(q0, c0)`, the compiler rejects it.

---

## 5. Eigen Quantum Intermediate Representation (EQIR v1)

**EQIR v1** represents a quantum circuit as a Directed Acyclic Graph (DAG) \(G = (V, E)\).

### 5.1 Graph Nodes (\(V\))
Each node \(v \in V\) represents an operation:
- `ALLOC(q)`: Allocates qubit resource `q`.
- `GATE(g, targets, args)`: Unitary gate application.
- `MEASURE(q, c)`: Measurement collapse mapping.
- `TRACE`, `PRINT`, `ASSERT`: Classical operations.

### 5.2 Graph Edges (\(E\))
Edges represent dependency wires. For any two nodes \(u, v \in V\), a directed edge \(u \to v\) is added if \(v\) depends on the resource state modified or read by \(u\).
- **Qubit Wire Edge**: If gate \(u\) is applied to qubit \(q_0\), and gate \(v\) is the next gate to target \(q_0\), an edge \(u \to v\) is established.
- **Classical Dependency Edge**: A conditional node \(v\) depending on a classical bit \(c_0\) is connected to the measurement node \(u\) that wrote to \(c_0\).
- **Barrier Node (`TRACE`)**: The `TRACE` node prints the global state, depending on the last active operation of all allocated qubits.

---

## 6. Graph-Based Optimization Techniques

Because EQIR v1 is structured as a DAG, optimization passes can be written as local graph rewrites.

### 6.1 Redundancy Elimination (Self-Inverse Cancellation)
For any unitary operator \(U\) that is its own inverse (\(U^2 = I\)), applying it twice consecutively results in the identity operation:
\[U^2 |\psi\rangle = I |\psi\rangle = |\psi\rangle\]
In the DAG, if node \(A\) has a single child \(B\) along qubit wire \(q\), and both \(A\) and \(B\) represent the same self-inverse gate \(U \in \{H, X, Y, Z\}\) with identical classical conditions, both nodes are deleted, and the parents of \(A\) are connected directly to the children of \(B\).

### 6.2 Rotation Merging
Rotations about a specific axis \(k \in \{X, Y, Z\}\) are additive:
\[R_k(\theta_2) \cdot R_k(\theta_1) = R_k(\theta_1 + \theta_2)\]
In the DAG, if node \(A\) is a rotation \(R_k(\theta_1)\) and its next child \(B\) along the target qubit wire is \(R_k(\theta_2)\), the optimizer:
1. Replaces the angle parameter of \(A\) with \((\theta_1 + \theta_2) \pmod{2\pi}\).
2. Deletes node \(B\), bypassing it to connect \(A\) to \(B\)'s children.

---

## 7. Formal Equivalence Verification

Eigen implements exact unitary matrix comparison to formally verify circuit equivalence.

### 7.1 Mathematical Equivalence up to Global Phase
Two circuits representing operators \(U_1\) and \(U_2\) are physically equivalent if they differ only by a global phase factor \(e^{i\theta}\):
\[U_1 = e^{i\theta} U_2\]
This means the matrix entries satisfy:
\[u_{1,jk} = e^{i\theta} u_{2,jk} \quad \forall j, k\]

### 7.2 Unitary Matrix Comparison Algorithm
To compare two graphs \(G_1\) and \(G_2\) containing a set of active qubits \(Q\):
1. Construct the identity matrix \(I\) of size \(2^N \times 2^N\) (where \(N = |Q|\)).
2. For each basis state \(|j\rangle\) represented by the \(j\)-th column of \(I\):
   - Set the state vector of a clean quantum simulator to \(|j\rangle\).
   - Execute the quantum gates of \(G_1\) and retrieve the output state vector \(|\psi_{1,j}\rangle\).
   - Repeat for \(G_2\) to retrieve \(|\psi_{2,j}\rangle\).
   - Write these state vectors into the \(j\)-th columns of matrices \(U_1\) and \(U_2\).
3. Find the entry in \(U_2\) with the maximum absolute value, \(U_2[r][c]\).
4. Compute the phase ratio:
   \[g = \frac{U_1[r][c]}{U_2[r][c]}\]
5. Verify that \(|g| \approx 1.0\).
6. Verify that \(|U_1[x][y] - g \cdot U_2[x][y]| < 10^{-5}\) for all \(x, y\).

### 7.3 Qubit Limits and Complexity
Since the unitary matrix size scales exponentially (\(2^{2N}\)), equivalence checks are restricted to \(N \le 8\) qubits. For larger systems, the verification is skipped, and a limitation warning is returned.

---

## 8. Experimental Evaluation

We evaluated the performance of the Eigen compiler frontend, optimizer, and runtime simulator across three primary benchmarks.

### 8.1 Benchmark 1: Bell State Correlation
- **Qubits**: 2
- **Gates**: H, CNOT
- **Execution Time**: 0.14 ms
- **Result**: Perfect correlation measured between qubits (\(c_0 == c_1\)) in all runs.

### 8.2 Benchmark 2: 3-Qubit GHZ State
- **Qubits**: 3
- **Gates**: H, CNOT (x2)
- **Execution Time**: 0.20 ms
- **Result**: Complete correlation (\(c_0 == c_1 == c_2\)) verified.

### 8.3 Benchmark 3: Optimizer Performance
We compiled a circuit with redundant gates:
\[\text{Circuit} = H \cdot H \cdot X \cdot X \cdot RX(\pi/4) \cdot RX(\pi/4)\]
- **Initial Gate Count**: 6 (Depth = 6)
- **Optimized Gate Count**: 1 (Depth = 1)
- **Optimizer Speed**: 3 rewrites completed in under 0.05 ms.
- **Equivalence Status**: Unitary matrix comparison confirmed the optimized and unoptimized circuits were identical up to global phase.

---

## 9. Discussion & Limitations

While Eigen v1.0 MVP provides a robust compile-and-run pipeline, several bottlenecks exist:
1. **Classical Simulation Bottleneck**: The state-vector simulator stores \(2^N\) amplitudes in a list. For \(N > 25\), memory consumption exceeds typical personal computer resources.
2. **Equivalence Checking Overhead**: Exact matrix comparison scales exponentially. Symbolic circuit representations (e.g. Decision Diagrams) are required to support verification for larger systems.
3. **No Noise Simulation**: The simulator assumes perfect, coherent qubits. Real hardware introduces decoherence and gate errors, which are not modeled in Phase 1.

---

## 10. Future Work

For future versions of the Eigen project, we plan to implement:
- **Symbolic Equivalence Checking**: Replacing matrix comparisons with Algebraic Decision Diagrams.
- **Hardware Export**: Adding code generators to transpile EQIR v1 DAGs directly to OpenQASM 2.0 and IBM Qiskit.
- **Noise Models**: Simulating amplitude damping and phase dephasing channels.

---

## 11. Conclusion

Eigen v1.0 MVP demonstrates that quantum programming can be structured, modular, and formally verified using lightweight classical software. By building the compiler and simulator in pure Python around a Directed Acyclic Graph Intermediate Representation (EQIR v1), we provide an educational and research-oriented platform for quantum compiling and verification.

---

## References

1. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
2. Cross, A. W., Bishop, L. S., Smolin, J. A., & Gambetta, J. M. (2017). *Open Quantum Assembly Language*. arXiv preprint arXiv:1707.03429.
3. Qiskit Contributors. (2023). *Qiskit: An Open-Source Framework for Quantum Computing*.
4. Microsoft. (2018). *The Q# Programming Language Specification*.
