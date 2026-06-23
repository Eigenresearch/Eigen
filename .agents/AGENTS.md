# Workspace Rules: Eigen 2.3 — Helios Project Guidelines

You are working on Eigen — a runtime-first hybrid classical-quantum programming language.

## Current Goal
Release Eigen 2.3 as a mature, fast, accurate, and research-ready release.

The release must improve:
- runtime speed;
- compiler speed;
- simulation accuracy;
- VM stability;
- honesty of documentation;
- scalability of quantum and hybrid computations;
- backend safety;
- reproducibility;
- developer experience;
- packaging/installability;
- research readiness;
- native integration readiness for Rust/C++.

---

## Core Principles

1. **Runtime-first**: Eigen Runtime remains the source of truth. The language must execute correctly inside its own runtime/VM.
2. **Backend compatibility is secondary**: Qiskit, OpenQASM, IBM, IonQ, Braket, Azure, LLVM, Rust/C++ adapters are compatibility targets and layers, not the source of language semantics.
3. **No silent semantic loss**: If a backend doesn't support a construct, this must be reflected through diagnostics, reports, warnings, strict checks, and documentation.
4. **Production-minded**: The release must improve stability, performance, observability, reproducibility, packaging, and installability.
5. **Research-ready**: Eigen must be suitable for research in language design, quantum compilation, simulation, verification, optimization, and hybrid execution.
6. **Honest scaling**: Exact simulation, equivalence checking, and sparse methods must have explicitly documented limits. Do not promise the impossible.
7. **Backward compatible**: Do not break working programs without extreme necessity. Any breaking changes must be rare and well-documented.

---

## 1. Performance Track

### 1.1 Runtime Speed
Speed up the VM and runtime execution:
- table-driven opcode dispatch;
- cache hot stack/frame references inside execution loop;
- reduce attribute lookup overhead;
- reuse activation frames where safe;
- fast paths for LOAD/STORE/ARITH/JUMP/CALL/RET;
- minimize object churn;
- reduce repeated dictionary lookups;
- optimize hot loops and recursive execution paths;
- improve call stack handling efficiency.

### 1.2 Compiler Speed
Speed up compile pipeline:
- lexer optimization using slicing and minimal temporary strings;
- parser simplification with fewer temporary nodes;
- type checker caching for symbols and repeated subtrees;
- import resolver caching for unchanged modules;
- incremental cache for AST / EQIR / EBC;
- faster module loading and project rebuilds.

### 1.3 Optimizer Speed
Make the optimizer faster and stronger:
- constant folding;
- dead code elimination;
- dead gate elimination;
- gate fusion;
- commutation-aware rewrites;
- peephole optimizations;
- canonicalization passes;
- cleanup pass;
- optimizer regression tests;
- measurable reduction in depth/gate count/runtime cost.

### 1.4 Simulator Speed
Speed up quantum simulation:
- precompute common matrices once;
- in-place updates where safe;
- reduce repeated state copies;
- optimize measurement paths;
- optimize common gate kernels;
- preserve numerical correctness;
- support both exact and approximate/scalable paths depending on circuit structure.

---

## 2. Accuracy Track

### 2.1 Determinism
Add deterministic execution mode:
- fixed seeds;
- reproducible benchmark runs;
- stable ordering in audit/profile output;
- repeatable compiler and simulator results when possible.

### 2.2 Numerical Stability
Improve numerical correctness:
- stable normalization in simulator;
- explicit truncation rules in MPS;
- tracked entanglement entropy;
- explicit truncation error metrics;
- avoid hidden approximations;
- document when approximate methods are used.

### 2.3 Verification Correctness
Improve correctness guarantees:
- fast-reject canonical hashes;
- rewrite-based verification fallback;
- exact equivalence only where it is mathematically justified;
- never claim hash equality alone is a proof of equivalence;
- add verification warnings in docs and CLI output.

---

## 3. Scalability Track

### 3.1 Sparse Simulation
Sparse simulator should be documented honestly:
- it scales with sparsity, not with a fixed qubit promise;
- sparse workloads may scale far beyond dense ones;
- dense superposition-heavy circuits remain exponential;
- sparse simulator is a specialized tool, not a universal solution.

### 3.2 MPS / Tensor Network Scaling
Improve large-system support:
- bond dimension controls;
- Schmidt coefficients tracking;
- Von Neumann entanglement entropy;
- cumulative truncation error;
- support for large low-entanglement systems;
- clear metrics for approximation quality.

### 3.3 Large-Circuit Verification
For bigger circuits:
- use canonicalization as a fast-reject layer;
- then apply rewrite verification;
- then exact unitary comparison only for small circuits;
- add graph-based simplification rules;
- preserve clear fallback behavior.

---

## 4. Quantum Compiler & Verify Track

### 4.1 ZX Calculus Engine
Upgrade ZX support:
- local complementation;
- pivoting;
- bialgebra rules;
- Hopf rules;
- phase gadget fusion;
- spider fusion;
- graph simplification;
- equivalence validation pipeline;
- separate fast-reject and proof/verification stages.

### 4.2 Resource Estimation
Add a resource estimator:
- logical qubits;
- circuit depth;
- T-count;
- T-depth;
- Clifford count;
- CNOT count;
- measurements;
- resource summary for research and hardware planning.

### 4.3 Noise-Aware Simulation
Add noise models:
- bit flip;
- phase flip;
- depolarizing;
- amplitude damping;
- readout error;
- Monte Carlo stochastic execution path;
- seed-controlled reproducibility.

### 4.4 Hardware Routing
Add hardware connectivity mapping:
- coupling map support;
- swap insertion;
- greedy routing;
- basic shortest-path routing;
- topology-aware compilation;
- clear device-specific constraints.

---

## 5. Runtime Stability Track

### 5.1 VM Hardening
Harden runtime behavior:
- guard against invalid opcodes;
- stack underflow protection;
- call-frame integrity checks;
- invalid jump protection;
- structured errors instead of raw Python crashes;
- crash reports for unexpected VM failures.

### 5.2 Bytecode Versioning
Add clear versioning:
- major.minor bytecode format;
- strict compatibility checks;
- clear error on mismatched major versions;
- forward-compatible handling where possible.

### 5.3 Crash Recovery
Add recovery and diagnostics:
- crash logs;
- call stack snapshot;
- opcode/IP information;
- local variable snapshot where safe;
- reproducible failure reports.

---

## 6. Tooling & Dev Experience Track

### 6.1 CLI Expansion
Support and improve CLI subcommands:
`run`, `build`, `exec`, `test`, `fmt`, `doc`, `init`, `bench`, `audit`, `profile`, `doctor`, `publish`, `search`, `install`, `lsp`, `estimate`, `debug`.

### 6.2 Audit Mode
`eigen audit` should:
- scan workspace recursively;
- evaluate imports and declarations;
- compare AST constructs against backend capability matrix;
- support strict mode;
- fail with exit code 1 when unsupported constructs are present in strict mode;
- support research mode output for reproducibility.

### 6.3 Profile Mode
`eigen profile <file>` should report:
- compile time;
- VM time;
- peak memory;
- heap allocations;
- call depth;
- quantum ops;
- optionally noise/routing/estimation stats.

### 6.4 Doctor Mode
`eigen doctor` should verify:
- Python environment;
- packaging tools;
- runtime availability;
- stdlib availability;
- local configuration health;
- compiler/VM smoke tests.

### 6.5 Benchmark Dashboard
`eigen bench --html` should generate:
- compile/runtime charts;
- memory charts;
- slowdown/regression markers;
- per-example performance summaries;
- standalone HTML report.

---

## 7. Documentation Track

### 7.1 Honest Positioning
README must explain:
- what Eigen is and why it exists;
- how it differs from Qiskit, Cirq, PennyLane, OpenQASM 3, Q#, Silq;
- what works in runtime vs. what is only backend compatibility;
- where exact simulation ends and approximate/specialized methods begin.

### 7.2 Performance & Accuracy Docs
Document:
- runtime optimizations;
- simulator boundaries;
- MPS metrics;
- canonical hash limitations;
- noise model behavior;
- routing behavior;
- reproducibility mode;
- benchmark interpretation.

### 7.3 Release Docs
Update README, architecture, runtime spec, optimizer spec, equivalence checker, standard library docs, examples guide, paper, audit, release report, migration notes, and backend compatibility matrix.

---

## 8. Ecosystem Track

### 8.1 Standard Library
Expand stdlib: math, stats, random, collections, io, time, string, linear algebra, optimization, and quantum helpers.

### 8.2 Package Manager
Add package ecosystem support: publish, search, install, lockfile reproducibility, checksum verification, dependency resolution, and registry foundation.

### 8.3 Local Installability
Verify users can run, build, test, bench, doctor, and audit locally after install.

### 8.4 IDE Readiness
Prepare for LSP: diagnostics, hover, go-to-definition, symbol lookup, semantic highlighting, code actions, and future auto-completion.

---

## 9. Native Integration Track (Rust / C++)

### 9.1 Rust Integration
Preferred path for performance-critical modules:
- runtime helpers;
- sparse simulation;
- routing;
- ZX graph operations;
- CLI utilities;
- safe native extension interface.

### 9.2 C++ Integration
Use C++ only where clearly useful: LLVM interoperability, scientific libraries, legacy native libraries, and low-level kernels.

### 9.3 Native Module Policy
Native modules must be optional, not break pure-Python operation, have fallbacks, be testable, and be documented.

---

## 10. Testing Track

Verify runtime capabilities (recursion, structs, exceptions, imports, etc.), backend compatibility execution, simulator correctness (sparse, dense, MPS, noise, routing), verification correctness, and performance benchmarks.

---

## 11. Acceptance Criteria

Eigen 2.3 is ready only if:
1. All tests pass.
2. Performance improves measurably on baseline benchmarks.
3. Runtime is stable under invalid input and corrupted bytecode.
4. Simulator results are correct and honestly documented.
5. Canonical hash is not misrepresented as proof of equivalence.
6. Strict backend mode fails correctly on unsupported constructs.
7. Docs reflect real implementation.
8. Native Rust/C++ integration has a roadmap with FFI boundary.
9. Local installation works.
10. Audit/profile/doctor/bench behave correctly.

---

## 12. Non-Goals
- Do not promise unlimited qubit simulation.
- Do not claim hash equality is mathematical proof of equivalence.
- Do not make backend compatibility override runtime semantics.
- Do not silently change output semantics.
- Do not break the pure runtime path when native modules are absent.
