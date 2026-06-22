# EQIR v1 Specification

This document defines the structure and behavior of **EQIR v1** (Eigen Quantum Intermediate Representation v1), a graph-based representation for quantum circuits.

## 1. Graph Data Structure (DAG)

Unlike traditional intermediate representations that represent circuits as linear arrays of instructions, EQIR v1 models the program as a Directed Acyclic Graph (DAG).
- **Nodes (\(V\))**: Represent operations:
  - `ALLOC`: Resource allocation (qubit initialization).
  - `GATE`: Unitary quantum gates (H, X, RX, CNOT, etc.).
  - `MEASURE`: Quantum measurement and collapse.
  - `TRACE`, `PRINT`, `ASSERT`: Classical debugging operations.
- **Edges (\(E\))**: Represent dependencies. A directed edge \(A \to B\) exists if operation \(B\) depends on the result of operation \(A\).

### Wire Dependency Construction
Edges are created dynamically during compilation by tracking the "last writer" of each resource:
1. **Qubit Wires**: For each qubit name, the compiler maintains a pointer to the last node that touched it. When a new gate node is created:
   - For every qubit in its target list, the compiler retrieves the qubit's last active node, creates a directed edge from that node to the new gate node, and updates the last active node pointer to the new gate node.
2. **Classical Wires**: Similar dependency tracking is used for classical bits (`cbit`). A node that reads or writes to a `cbit` (e.g. `measure q0 -> c0` or `if c0 == 1`) depends on the last operation that wrote to `c0`.
3. **Barrier Nodes (`TRACE`)**: Debug directives like `TRACE` print the state of the entire system. Therefore, `TRACE` acts as a full barrier: it depends on the last operations of *all* currently allocated qubits, preventing reordering across the trace directive.

---

## 2. Inlining and Subroutine Expansion

EQIR v1 does not support nested function calls at runtime. Instead, the `EQIRConverter` performs inline expansion of all `qfunc` calls during DAG construction:

1. When a `QFuncCallNode` is encountered, the compiler looks up the targeted `qfunc` declaration.
2. It maps the local parameter names of the `qfunc` to the actual qubit and classical bit arguments passed by the caller.
3. It recursively compiles the body of the `qfunc` using this parameter mapping.
4. The generated operation nodes are inserted directly into the global DAG, with dependency wires connecting them seamlessly to the surrounding operations.

---

## 3. Circuit Depth Analysis

The depth of a quantum circuit corresponds to the length of the longest path of sequential, dependent quantum operations. In EQIR v1, depth is calculated mathematically using dynamic programming on the DAG:

Let \(D(n)\) be the depth at node \(n \in V\):
- For a node \(n\) that is not a quantum operation (e.g. `ALLOC` or classical debug statements), its weight is \(W(n) = 0\).
- For a quantum gate or measurement node, its weight is \(W(n) = 1\).
- The depth at node \(n\) is calculated as:
  \[D(n) = W(n) + \max_{p \in \text{parents}(n)} D(p)\]
- The overall circuit depth is the maximum depth across all sink nodes in the DAG:
  \[\text{Depth} = \max_{s \in \text{sinks}(V)} D(s)\]

This topological analysis calculates the critical path of the circuit, indicating the minimum number of time-steps required to execute the circuit in parallel.
