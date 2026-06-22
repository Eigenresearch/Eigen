# Eigen Examples Guide

This guide describes how to run and verify the quantum algorithms provided in the `examples/` directory.

## 1. Running Examples

To execute an Eigen program, use the `run` command on the CLI:
```bash
python src/main.py run examples/bell.eig
```

### Step Tracing (`--trace`)
To inspect the quantum state vector step-by-step during execution, add the `--trace` flag:
```bash
python src/main.py run examples/bell.eig --trace
```

### Optimization (`--optimize`)
To enable graph-based optimizations (gate cancellations and rotation merging) before execution, add the `--optimize` flag:
```bash
python src/main.py run examples/bell.eig --optimize
```

---

## 2. Examples Breakdown

### 2.1 Bell State (`examples/bell.eig`)
Creates the entangled state:
\[| \Phi^+ \rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}\]
Measuring either qubit collapses both to identical classical outcomes. The program verifies this using an assertion:
```eigen
measure q0 -> c0
measure q1 -> c1
assert c0 == c1
```

### 2.2 GHZ State (`examples/ghz.eig`)
Generates a 3-qubit entangled Greenberger–Horne–Zeilinger state:
\[| \text{GHZ} \rangle = \frac{|000\rangle + |111\rangle}{\sqrt{2}}\]
The program measures all three qubits and asserts that their outcomes match.

### 2.3 Conditional Execution (`examples/cond_execution.eig`)
Demonstrates classical feed-forward control:
1. Qubit `q0` is prepared in state \(|1\rangle\) and measured.
2. The runtime reads the classical outcome.
3. If the outcome is `1`, a rotation is applied to `q1`.
4. Measurements verify the conditional path succeeded.

### 2.4 Optimization Demo (`examples/opt_demo.eig`)
Contains redundant gates:
```eigen
H q0
H q0   # cancels out
X q0
X q0   # cancels out
RX q0, PI/4
RX q0, PI/4  # merges into RX q0, PI/2
```
Running it with `--optimize` shows that the optimizer reduces the gate count from 6 to 1, while preserving the final quantum state vector.
