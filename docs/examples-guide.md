# Eigen Examples Guide

This guide describes how to run and verify the quantum and hybrid classical-quantum algorithms provided in the `examples/` directory.

## 1. Running Examples

The Eigen command-line interface provides two main paths for running code: the **Eigen VM** (supporting classical-quantum hybrid operations) and the **Qiskit Transpiler Backend** (transpiling to Python Qiskit Aer code).

### 1.1 Running on Eigen VM (Recommended for Hybrid Code)
To run a program natively on the VM (which supports recursion, structs, maps, loops, and exceptions):
```bash
python src/main.py run examples/phase2_demo.eig --vm
```
To enable step-by-step tracing of the quantum state:
```bash
python src/main.py run examples/phase2_demo.eig --vm --trace
```

### 1.2 Transpiling to Qiskit
To transpile an Eigen program into a Qiskit Aer script:
```bash
python src/main.py run examples/phase2_demo.eig --backend qiskit
```
If the program contains unsupported features (like structs or recursion), the transpiler outputs a safe warning report and comments out the unsupported constructs in the final Python code instead of crashing.

---

## 2. Examples Breakdown

### 2.1 Phase 2 Demo (`examples/phase2_demo.eig`)
Showcases the features of the Eigen 2.3 language:
- Classical recursion (factorial)
- Struct declaration, initialization, and field mutation
- Quantum Bell state preparation
- Measurement and cross-cbit assertion check

### 2.2 Bell State (`examples/bell.eig`)
Creates the entangled state:
\[| \Phi^+ \rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}\]
Measuring either qubit collapses both to identical classical outcomes. The program verifies this using an assertion:
```eigen
measure q0 -> c0
measure q1 -> c1
assert c0 == c1
```

### 2.3 Optimization Demo (`examples/opt_demo.eig`)
Contains redundant gates and sequential rotations:
```eigen
H q0
H q0   # cancels out
X q0
X q0   # cancels out
RX q0, PI/4
RX q0, PI/4  # merges into RX q0, PI/2
```
Running it with `--optimize` shows that the optimizer reduces the gate count from 6 to 1, while preserving the final quantum state vector.

---

## 3. Runtime Guarantees and Backend Compatibility Matrix

### Runtime Guarantees
All classical-quantum hybrid examples in the `examples/` folder are guaranteed to run successfully on the **Eigen VM**.

### Compatibility Matrix for Examples
| Example | Target: Eigen VM | Target: Qiskit Backend | Target: EQIR v1.1 DAG |
| --- | --- | --- | --- |
| `bell.eig` | `FULL` | `FULL` | `FULL` |
| `opt_demo.eig` | `FULL` | `FULL` | `FULL` |
| `phase2_demo.eig` | `FULL` | `PARTIAL` (Quantum only) | `NONE` (due to Structs/Recursion) |
