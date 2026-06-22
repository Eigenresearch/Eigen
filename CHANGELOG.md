# Changelog

All notable changes to the Eigen programming language project will be documented in this file.

## [1.0.0] - 2026-06-22

Initial release of Eigen v1.0 MVP, featuring a modular compiler frontend, a graph-based Intermediate Representation (EQIR v1), an optimizer, a state-vector simulator, and a mathematical equivalence checker.

### Added
- **Compiler Frontend**:
  - `lexer.py` character scanner with column and line tracking.
  - `parser.py` recursive descent parser translating source to AST.
  - `import_resolver.py` supporting dotted module paths (e.g. `import quantum.bell`) and recursive path resolution in `stdlib/` and workspace.
  - `type_checker.py` validating safety checks for `qubit`, `cbit`, `int`, and `float` variables.
- **EQIR v1 (Eigen Quantum Intermediate Representation v1)**:
  - Directed Acyclic Graph (DAG) wire dependency structure in `ir_graph.py`.
  - Inlining of quantum function calls (`qfunc`) into flat DAG representations in `ir_converter.py`.
- **DAG Optimizer**:
  - Redundant self-inverse gate cancellation (`H; H` and `X; X`).
  - Consecutive rotation gate merging (`RX(a); RX(b) -> RX(a+b)`).
- **Quantum State-Vector Simulator**:
  - `simulator.py` supporting standard gates (H, X, Y, Z, S, T), rotation gates (RX, RY, RZ), 2-qubit gates (CNOT, CZ, SWAP), and probabilistic measurement wavefunction collapse.
- **Eigen Runtime**:
  - Topological scheduling of graph nodes.
  - Tracing mode printing complex amplitude states and bit changes.
  - Assertions validation.
- **Equivalence Checker**:
  - Mathematical verification of circuit equivalence up to 8 qubits using exact unitary matrix comparison.
- **Standard Library**:
  - Modules for Bell state, GHZ state, Deutsch algorithm oracles, and Grover diffuser.
- **Test Suite**:
  - 19 unit tests checking all compiler, optimizer, and simulator features.
