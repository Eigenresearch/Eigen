# Eigen Project Audit

This document presents a technical audit of the Eigen v1.0 MVP language, compiler frontend, DAG-based intermediate representation, and state-vector simulator.

---

## 1. Architectural Strengths

1. **Topological Decoupling via DAG**: Modeling quantum operations in **EQIR v1** as a Directed Acyclic Graph (DAG) decoupled execution from linear notation constraints. This allows for straightforward dependency analysis (calculating circuit depth) and facilitates parallel scheduler mappings.
2. **Modular Compiler Front-End**: The recursive descent parser combined with modular namespace imports (`module`, `import`) allows developers to write scaleable quantum code. The import resolver handles path mappings cleanly, resolving standard libraries and local files recursively.
3. **Strict static type checking**: The `TypeChecker` checks type constraints before running simulations, catching errors such as applying quantum gates to classical bits, which is a common failure mode in Qiskit (which lacks compilation type-safety).
4. **Exact Equivalence Checking**: The exact unitary matrix comparison checker is mathematically robust and phase-invariant (\(U_1 = e^{i\theta} U_2\)), providing a reliable correctness validator for compiler optimization passes.

---

## 2. Architectural Weaknesses and Limitations

1. **State-Vector Simulation Complexity**:
   The quantum state-vector simulator stores and processes the full amplitude state space.
   - **Time Complexity**: Applying a 1-qubit gate on a system of \(N\) qubits requires traversing \(2^N\) states, executing \(2^{N-1}\) complex coordinate transforms. The time complexity per gate is \(O(2^N)\).
   - **Memory Complexity**: The state vector size is \(2^N\) complex numbers.
     - At 20 qubits: \(2^{20} = 1,048,576\) complex values (\(\approx 16\) MB of RAM).
     - At 30 qubits: \(2^{30} \approx 1.07 \times 10^9\) complex values (\(\approx 16\) GB of RAM).
     - At 40 qubits: \(2^{40} \approx 1.1 \times 10^{12}\) complex values (\(\approx 16\) TB of RAM).
     This forms a hard limit for classical simulation.
2. **Equivalence Checking Memory Blowup**:
   Unitary matrix comparisons are restricted to \(N \le 8\) qubits.
   - The matrix size scales as \(2^N \times 2^N = 2^{2N}\).
   - Comparing 10-qubit circuits requires comparing matrices of size \(1024 \times 1024\) (\(2^{20} \approx 10^6\) entries).
   - Comparing 20-qubit circuits requires \(2^{40} \approx 10^{12}\) entries (16 TB of RAM), which is impossible for classical machines.
   - Therefore, the current equivalence checker does not scale to large systems.
3. **No Noise / Decoherence Modeling**:
   The simulator performs ideal, coherent unitary calculations. Real NISQ quantum devices suffer from state decay (relaxation time \(T_1\)), phase dephasing (\(T_2\)), and thermal noise. The lack of open system density matrix simulation limits Eigen's usefulness for simulating realistic hardware.
4. **Dynamic branching execution issues**:
   In our DAG model, conditional branches (`if cbit == 1`) are handled by placing classical conditions directly onto gate nodes. If a block is conditionally skipped at runtime, the execution engine skips the corresponding nodes. While simple, this does not support dynamic nested loop structures or complex runtime branching natively in the DAG graph.

---

## 3. Simulator Bottlenecks

- **Pure Python loops**: Applying single-qubit gates requires running a `for` loop in Python over \(2^N\) items. Because Python is an interpreted language, these loops suffer from high CPU overhead compared to compiled C++/Rust array slicing or GPU-accelerated tensor contractions.
- **Copying array matrices**: Copying the state vector to columns during unitary matrix construction requires repeatedly re-allocating memory, leading to garbage collection overhead.

---

## 4. Security Considerations

- **Import path traversal**: The `ImportResolver` maps dotted module names to paths (e.g. `import quantum.bell` maps to `quantum/bell.eig`). Because it joins paths, an attacker could attempt path traversal attacks using dot-dot structures (e.g. `import ..dangerous.exploit`).
- **Input code injection**: There is no sandboxing. If future versions add system commands or file writing directives, the compiler front-end must check path permissions to prevent remote code execution.

---

## 5. Future Architectural Improvements

1. **Compile Simulator to C/C++ or Rust**: Porting the simulator matrix transformations to a compiled C/C++ library (e.g. via PyO3 for Rust or C-types) would speed up execution times by \(100\times\) and allow leveraging multi-core parallel processing (OpenMP) or GPU scaling (CUDA).
2. **Algebraic Decision Diagrams**: Transition the equivalence checker from raw matrix comparisons to Quantum Decision Diagrams (QDDs). QDDs compress regular gate structures into compact graph topologies, enabling the comparison of circuits with up to 50+ qubits in milliseconds.
3. **Transpiler backends**: Implement exporters to target OpenQASM 2.0/3.0. This allows Eigen to compile down to standard formats that can be executed directly on real quantum hardware (IBM Q, Rigetti, etc.).
