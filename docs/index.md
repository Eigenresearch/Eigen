# Welcome to the Eigen 2.6 «Misery» Language Documentation

Eigen is a runtime-first, hybrid classical-quantum programming language designed for state-of-the-art performance, stability, and research readiness.

## What's New in 2.6 «Misery»

- **Inno Setup Windows Installer** — Full GUI wizard with component selection, PATH, file association
- **Real IBM Device Topologies** — Eagle (127q), Condor (1121q), IonQ all-to-all, Rigetti ring, Google Sycamore
- **Real Heavy-Hex Topology** — `CouplingMap.heavy_hex()` generates genuine IBM heavy-hex
- **Advanced Noise Engine** — `NoiseChannel` ABC, `T1T2NoiseModel` with physical timing, `CrosstalkModel`, `DeviceNoiseProfile`
- **Stabilizer Auto-fallback** — Non-Clifford gates auto-switch to dense simulator instead of crashing
- **Stabilizer Pre-flight Check** — `check_circuit_compatibility()` detects non-Clifford gates
- **MPS Auto Bond Dimension** — Automatic bond dimension increase with truncation error threshold
- **Structured GPU Logging** — `logging` module instead of `print()`
- **Migration Guide** — `MIGRATION.md` for 2.5 → 2.6
- **Configurable Magic Values** — JIT `hot_threshold`, router `lookahead`, MPS `max_bond_dim`

## Core Pillars of Eigen

1. **Runtime-First Execution**: Eigen features a dedicated portable stack virtual machine (VM) with an adaptive Just-In-Time (JIT) compiler that delivers 1.8x faster classical loops than CPython.
2. **Zero Python Core**: High-performance compilers, AST parsers, simulators, and optimizers are built natively in Rust.
3. **Advanced Quantum Simulations**: 6 backends — dense (Rust), sparse, MPS (auto bond dim), stabilizer (1000+ qubits with auto-fallback), density matrix, GPU.
4. **Mathematical Verification**: Static type check, qubit safety analyzer, ZX-calculus formal equivalence checking, and stabilizer compatibility analysis.
5. **Hardware Portability**: Compiles directly to OpenQASM 3.0, Quil, LLVM/QIR native executables, with SABRE routing onto real IBM, IonQ, Rigetti, and Google topologies.
6. **Advanced Noise Modelling**: T1/T2 relaxation with gate timing, crosstalk, device-specific calibration profiles, and composable noise channels.

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

## Additional Resources

- **[README.md](../README.md)**: Full project overview, benchmarks, and comparison with other languages.
- **[LANGUAGE.md](../LANGUAGE.md)**: Complete language specification.
- **[MIGRATION.md](../MIGRATION.md)**: Migration guide for 2.5 → 2.6.
- **[CHANGELOG.md](../CHANGELOG.md)**: Full changelog with all releases.

## Test Results

```
449 tests passed
439 subtests passed
0 failures
```
