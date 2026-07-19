# Welcome to the Eigen 2.7 «Meridian» Language Documentation

Eigen is a runtime-first, hybrid classical-quantum programming language designed for performance, stability, and research workflows.

## What's New in 2.7 «Meridian»

- **Incremental compiler pipeline** — AST, EQIR, EBC, and import caches for unchanged modules
- **VM optimizations** — inline variable cache, frame cache, hot-loop detection, and object pooling
- **GPU acceleration surface** — CuPy/JAX-aware simulator dispatch with deterministic CPU fallback
- **Pulse-level control** — Gaussian, DRAG, and square pulses with scheduled execution
- **Distributed simulation** — MPI state-vector distribution and tensor-network planning
- **Expanded interop** — Python, Rust, C, WASM, LLVM/QIR, OpenQASM 3.0, and Quil targets
- **Research tooling** — tomography, error mitigation, experiment tracking, and seeded reproducibility
- **Developer tooling** — DAP debugging, shell completion, playground, and code migration support

## Core Pillars of Eigen

1. **Runtime-first execution**: A portable stack VM with adaptive JIT compilation.
2. **Native acceleration**: Rust kernels accelerate simulation, routing, and compiler hot paths.
3. **Quantum simulation**: Dense, sparse, MPS, stabilizer, density-matrix, and GPU backends.
4. **Verification**: Static type checking, qubit safety analysis, and circuit equivalence tools.
5. **Hardware portability**: OpenQASM 3.0, Quil, LLVM/QIR, and provider runtime integrations.
6. **Noise modelling**: T1/T2 relaxation, crosstalk, device profiles, and composable channels.

## Navigation Guide

- **[Language Specification](language-specification.md)**: Eigen syntax, types, structures, and quantum operations.
- **[Architecture](architecture.md)**: Compilation, optimization, and execution pipeline.
- **[Compiler Design](compiler-design.md)**: Parsing, SSA, LLVM, and AOT compilation.
- **[EQIR Specification](eqir-specification.md)**: Eigen Quantum Intermediate Representation.
- **[Equivalence Checker](equivalence-checker.md)**: Circuit equivalence and ZX-calculus tooling.
- **[Optimizer Specification](optimizer-specification.md)**: Circuit and bytecode optimization.
- **[Runtime Specification](runtime-specification.md)**: VM opcodes, dispatch, and JIT caching.
- **[Standard Library](standard-library.md)**: Standard modules and native helpers.
- **[Examples Guide](examples-guide.md)**: Practical Eigen programs.

## Additional Resources

- **[README.md](../README.md)**: Project overview, setup, and benchmarks.
- **[LANGUAGE.md](../LANGUAGE.md)**: Authoritative language specification.
- **[MIGRATION.md](../MIGRATION.md)**: Migration guide for 2.6 → 2.7.
- **[CHANGELOG.md](../CHANGELOG.md)**: Release history.

## Test Results

The current test count and platform matrix are reported by CI; run `pytest -q` for local verification.
