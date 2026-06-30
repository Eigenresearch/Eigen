# Welcome to the Eigen Language Documentation

Eigen is a runtime-first, hybrid classical-quantum programming language designed for state-of-the-art performance, stability, and research readiness.

## Core Pillars of Eigen

1. **Runtime-First Execution**: Eigen features a dedicated portable stack virtual machine (VM) with an adaptive Just-In-Time (JIT) compiler.
2. **Zero Python Core**: High-performance compilers, AST parsers, simulators, and optimizers are built natively in Rust.
3. **Advanced Quantum Simulations**: Out-of-the-box support for dense state-vector simulators, sparse state-vector simulators, tensor network Matrix Product States (MPS), and Kraus noise model density matrix simulations.
4. **Mathematical Verification**: Static type check, qubit safety analyzer, and formal equivalence check using ZX-Calculus and graph-based canonical hashes.
5. **Hardware Portability**: Compiles directly to OpenQASM 3.0, Quil, and LLVM/QIR native executable targets.

## Navigation Guide

- **[Language Specification](language-specification.md)**: Deep dive into Eigen's syntax, type system, classical structures, and quantum gate operations.
- **[Architecture](architecture.md)**: Details on the Salsa incremental compiler database, type checker, and overall workspace design.
- **[Compiler Design](compiler-design.md)**: Learn about parser Pratt parsing, SSA-form, LLVM target integration, and AOT compilation.
- **[EQIR Specification](eqir-specification.md)**: Specification of the Eigen Quantum Intermediate Representation (EQIR) directed acyclic graph.
- **[Equivalence Checker](equivalence-checker.md)**: Explanation of equivalence proof strategies using ZX-calculus spider fusion and local complementation.
- **[Optimizer Specification](optimizer-specification.md)**: Peephole circuit optimization, gate merging, and CNOT cancellation.
- **[Runtime Specification](runtime-specification.md)**: VM opcodes, table-driven execution, and JIT caching details.
- **[Standard Library](standard-library.md)**: Pure and native helpers for math, strings, collections, random, and stats.
- **[Examples Guide](examples-guide.md)**: Practical code examples (Grover's algorithm, QFT, Bell state, and quantum teleportation).
