# Changelog

All notable changes to the Eigen programming language project will be documented in this file.

## [2.4.0] - 2026-06-29

Release 2.4.0 "Mone" brings a completely decoupled, high-performance compilation and execution pipeline, a zero-copy native Rust frontend, Salsa-inspired incremental compiler caching, JIT v2 optimizations, and standalone LLVM/QIR native executable compilation.

### Added
- **Zero-CPython Standalone AOT Compiler:** Supports compiling to standalone machine binaries (`.exe` on Windows) free of CPython dependencies using a PyO3 feature gate and `cargo build --no-default-features`.
- **Zero-Copy Rust Frontend:** Zero-copy lexer slicing and Pratt-precedence parser implemented in native Rust, yielding a 9.7x compile speedup while instantiating standard mutable Python AST structures.
- **Salsa-Query Caching Database:** Tracks file content hashes (SHA-256) recursively to check intermediate compilation step validity (AST, SSA, EQIR, ZX, EBC) and bypass recompilation.
- **QIR Specification Compliant Emission:** Generates standard opaque pointer declarations and function bindings (`__quantum__rt__qubit_allocate`, etc.) with `--qir` CLI build option.
- **JIT v2 Loop Optimizations:** Implements loop-invariant code motion (LICM), constant folding, trace specialization, and shape/type guards with deoptimization fallbacks.
- **QFT Binary Conformance:** Added Quantum Fourier Transform example in `examples/qft.eig` and verified AOT compilation.

### Fixed
- **MSVC link.exe Compatibility:** Replaced invalid `/DEBUG:NONE` options with `/RELEASE` (and `/OPT:REF` for symbol stripping) and resolved LTO static library linkage mismatches.
- **Windows Path Mount Bug:** Resolved `os.path.relpath` ValueError on Windows when paths cross different drive mounts (e.g. `C:` and `D:`).
- **Stale AST Cache Bug:** Replaced timestamp-based AST caching in `ImportResolver` with file content hashing to handle sub-second writes.
- **Early Dependency Wiping Bug:** Fixed early dependency resets in `QueryDb.execute_query` before cache check evaluation.

## [2.3.0] - 2026-06-23

Release 2.3.0 "Helios" features VM integer opcode dispatch, memory layout optimizations using slots, and advanced simulation backends.

### Added
- **VM Opcode Optimizations:** Flat list-based opcode routing and integer-keyed instruction mapping replacing slow string dictionary lookups.
- **Memory Footprint Optimization:** Implemented `__slots__` layout for `Instruction`, `ActivationFrame`, and `HeapObject` to reduce memory churn.
- **Advanced Simulators:** Added Sparse Simulator and Matrix Product State (MPS) Tensor Network Simulator.
- **GPU Acceleration:** CUDA/ROCm/Metal state vector simulation support.
- **Formal Verification v2:** ZX-Calculus graph reduction engine (spider fusion, local complementation, pivoting, phase gadgets).

## [1.0.0] - 2026-06-22

Initial release of Eigen v1.0 MVP, featuring a modular compiler frontend, a graph-based Intermediate Representation (EQIR v1.1), an optimizer, a state-vector simulator, and a mathematical equivalence checker.
