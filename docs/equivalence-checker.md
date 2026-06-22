# Equivalence Checker Specification

This document details the mathematical algorithms and implementation constraints of the Eigen Equivalence Checker.

## 1. Mathematical Foundation (Unitary Comparison)

A quantum circuit on \(N\) qubits can be represented as a unitary matrix \(U\) of size \(2^N \times 2^N\). Two quantum circuits are physically equivalent if their unitary representations \(U_1\) and \(U_2\) are equal up to a global phase factor \(e^{i\theta}\):
\[U_1 = e^{i\theta} U_2 \quad \text{where } \theta \in \mathbb{R}\]

A global phase has no physical consequences because the probability of any measurement outcome is identical:
\[P(x) = |\langle x | U_1 | \psi \rangle|^2 = |\langle x | e^{i\theta} U_2 | \psi \rangle|^2 = |e^{i\theta}|^2 |\langle x | U_2 | \psi \rangle|^2 = |\langle x | U_2 | \psi \rangle|^2\]

---

## 2. Multi-Qubit Matrix Representation (Kronecker Products)

For a system of \(N\) qubits, the state space is the tensor product of individual qubit spaces:
\[\mathcal{H} = \mathbb{C}^2 \otimes \mathbb{C}^2 \otimes \dots \otimes \mathbb{C}^2\]

When a single-qubit gate \(A\) is applied to the \(k\)-th qubit (0-indexed from the right), the global operator acting on the entire system is the Kronecker product:
\[U_{\text{global}} = I^{\otimes (N - 1 - k)} \otimes A \otimes I^{\otimes k}\]
where \(I\) is the \(2 \times 2\) identity matrix.

For a 2-qubit gate like CNOT with control qubit \(c\) and target qubit \(t\):
- If \(c < t\), the global matrix represents a controlled operation:
  \[P_0 \otimes I + P_1 \otimes X\]
  extended via Kronecker products over all \(N\) qubits.

### Unitary Matrix Reconstruction Algorithm
Rather than computing Kronecker products explicitly (which is computationally expensive), Eigen generates the unitary matrix column-by-column by running basis states through the simulator:
1. Let \(I\) be the identity matrix of size \(2^N \times 2^N\). The \(j\)-th column of \(I\) represents the computational basis state \(|j\rangle\) where the \(j\)-th amplitude is 1.0, and all other amplitudes are 0.0.
2. For each column index \(j \in [0, 2^N-1]\):
   - We initialize the state-vector simulator to state \(|j\rangle\).
   - We walk the EQIR graph in topological order, executing all gate nodes.
   - The resulting state vector \(|\psi_j\rangle\) forms the \(j\)-th column of the unitary matrix \(U\).

---

## 3. Global Phase Identification

To check if \(U_1 = e^{i\theta} U_2\):
1. Locate the entry in \(U_2\) with the largest magnitude. Let this be at index \((r, c)\), with value \(v_2 = U_2[r][c]\). This ensures numerical stability by avoiding division by zero or small values.
2. Retrieve the corresponding value in \(U_1\), denoted \(v_1 = U_1[r][c]\).
3. Compute the phase ratio:
   \[g = \frac{v_1}{v_2}\]
4. Verify that \(g\) represents a valid phase factor (i.e. its magnitude is close to 1):
   \[||g| - 1.0| < \epsilon \quad (\text{typically } \epsilon = 10^{-5})\]
5. Check if all other elements are identical up to this factor:
   \[|U_1[x][y] - g \cdot U_2[x][y]| < \epsilon \quad \forall (x, y) \in [0, 2^N-1] \times [0, 2^N-1]\]

If all checks pass, the circuits are equivalent.

---

## 4. Complexity and Qubit Limitations

The cost of computing and storing the unitary matrix scales exponentially with the number of qubits:
- **Matrix Size**: \(2^N \times 2^N = 2^{2N}\) complex numbers.
- **Memory Cost**:
  - For \(N = 8\): \(256 \times 256 \approx 65,536\) complex numbers (approx. 1 MB).
  - For \(N = 10\): \(1024 \times 1024 \approx 1,048,576\) complex numbers (approx. 16 MB).
  - For \(N = 20\): \(1048576 \times 1048576 \approx 1.1 \times 10^{12}\) complex numbers (approx. 16 TB).

### Strict Limit Policy
To prevent exponential memory blowup on classical machines, the Phase 1 Equivalence Checker enforces a hard limit of **\(N \le 8\) qubits**. Any comparison requiring more than 8 qubits immediately exits with a validation error.
