# Eigen Optimizer Specification

This document mathematically describes the optimization passes performed by the Eigen compiler on the EQIR v1.1 DAG.

## 1. Redundancy Elimination (Self-Inverse Gates)

Many quantum gates are self-inverse (unitary operators that are their own inverse, meaning \(U^\dagger = U\), and thus \(U^2 = I\), where \(I\) is the identity operator). 

In Eigen, the self-inverse gates supported for cancellation are:
- Hadamard (\(H\)): \(H^2 = I\)
- Pauli-X (\(X\)): \(X^2 = I\)
- Pauli-Y (\(Y\)): \(Y^2 = I\)
- Pauli-Z (\(Z\)): \(Z^2 = I\)

### Mathematical Rule
If two identical self-inverse gates \(U\) are applied consecutively on the same qubit wire \(q\) with no intervening operations, they simplify to the identity operator:
\[|\psi'\rangle = U \cdot U |\psi\rangle = U^2 |\psi\rangle = I |\psi\rangle = |\psi\rangle\]

### DAG Implementation
1. The optimizer walks the topologically sorted nodes of the EQIR graph.
2. For each node \(A\) representing a gate \(U \in \{H, X, Y, Z\}\) on qubit \(q\):
   - It checks if \(A\) has a child node \(B\) that is the immediate next operation on qubit \(q\).
   - It verifies that \(B\) is a gate of the same type \(U\) on qubit \(q\) and shares the exact same classical condition as \(A\) (or both are unconditional).
3. If a match is found:
   - The optimizer bypasses both nodes by connecting all parent nodes of \(A\) directly to all child nodes of \(B\).
   - Nodes \(A\) and \(B\) are removed from the graph.

---

## 2. Rotation Merging (Gate Fusion)

Rotation gates represent continuous rotations in the Bloch sphere about a specific axis. Consecutive rotations about the same axis are additive.

In Eigen, the rotation gates supported for merging are:
- \(RX(\theta)\): Rotation about the X-axis
- \(RY(\theta)\): Rotation about the Y-axis
- \(RZ(\theta)\): Rotation about the Z-axis

### Mathematical Rule
The composition of two rotations about the same axis is equivalent to a single rotation by the sum of their angles:
\[R_k(\theta_2) \cdot R_k(\theta_1) = R_k(\theta_1 + \theta_2) \quad \text{for } k \in \{X, Y, Z\}\]

For example, for rotations about the X-axis:
\[RX(\theta_1) = \begin{pmatrix} \cos(\theta_1/2) & -i\sin(\theta_1/2) \\ -i\sin(\theta_1/2) & \cos(\theta_1/2) \end{pmatrix}\]
Multiplying two such matrices yields:
\[RX(\theta_2) \cdot RX(\theta_1) = RX(\theta_1 + \theta_2)\]

### DAG Implementation
1. For each node \(A\) representing a rotation gate \(R_k(\theta_1)\) on qubit \(q\):
   - It finds the next operation on qubit \(q\), denoted \(B\).
   - It checks if \(B\) is a rotation gate of the same type \(R_k(\theta_2)\) on qubit \(q\) with the same classical condition.
2. If matched:
   - The optimizer calculates the merged angle:
     \[\theta_{\text{merged}} = (\theta_1 + \theta_2) \pmod{2\pi}\]
   - It updates the angle parameter of node \(A\) to \(\theta_{\text{merged}}\).
   - It bypasses node \(B\) by connecting \(A\) directly to the child nodes of \(B\), and deletes \(B\) from the graph.

---

## 3. Scope Limits and Runtime Guarantees

### Runtime Guarantees
Optimization passes are guaranteed to preserve physical and state-vector equivalence up to a global phase (\(U_1 = e^{i\theta} U_2\)) for any input circuit.

### Backend Compatibility
The optimizer runs exclusively on the **EQIR v1.1 DAG** representation. It does not optimize classical VM instructions or dynamic control paths (like recursive loops or try-catch blocks) that operate outside the static quantum circuit graph.

| Phase / Feature | Optimizer Support | Eigen VM Executable | Qiskit Exportable |
| --- | --- | --- | --- |
| Gate Cancellation | `FULL` | `FULL` | `FULL` |
| Rotation Merging | `FULL` | `FULL` | `FULL` |
| Classical Code Optimization | `NONE` | `NONE` | `NONE` |
