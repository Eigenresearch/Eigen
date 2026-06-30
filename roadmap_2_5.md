# Eigen 2.5 «Mitz» — Technical Roadmap & Design Specification

> **Motto:** "Rust-First, Zero-Python Core, Industry-Grade Performance"
> **Target:** Address high-priority weak spots (W-1 to W-8) and deliver the remaining M5, M6, M7 milestones to make Eigen a production-grade, research-ready classical-quantum programming language.

---

## 1. Weak Spots Mitigation Plan (W-1 to W-8)

### [x] W-1: Native Multi-Qubit Gates (Toffoli, CSWAP, Fredkin, CP) (Done)
- **Problem:** Missing out-of-the-box 3-qubit gates, forcing manual CNOT decompositions.
- **Solution:** 
  - Added `CCX` (Toffoli), `CSWAP` (Fredkin), and `CP` (Controlled-Phase) to the native Rust simulator kernel (`native/rust/src/simulator.rs`).
  - Updated both Python and native Rust lexers/parsers to support `Toffoli`/`CCX`, `CSWAP`/`Fredkin`, and `CP` tokens.
  - Implemented exact matrix representations, VM, and runtime execution loops.

### [x] W-2: State-Vector Simulator Performance (NumPy/Rust/SIMD) (Done)
- **Problem:** Performance bottleneck when running large state-vector simulations in Python.
- **Solution:**
  - Optimized pure-Python `PythonDenseStatevector` by replacing slow loops with vectorized NumPy array slice updates, resulting in 100x+ speedups for larger systems.
  - Rust simulator (`RustStatevector`) runs by default when native bindings are compiled.

### [x] W-3: Parameterized Controlled Rotations (CRX, CRY, CRZ) (Done)
- **Problem:** Missing `CRX`, `CRY`, `CRZ` gates, causing phase-kickback issues during manual decomposition.
- **Solution:**
  - Added native `apply_crx`, `apply_cry`, and `apply_crz` methods to `RustStatevector` and `RustSparseSimulator` in `native/rust/src/simulator.rs`.
  - Added the corresponding gate definitions to the Python and Rust lexer, parser, and code generators.

### [x] W-4: Noise Channels and Decoherence (Done)
- **Problem:** Only ideal unitary simulation is supported.
- **Solution:**
  - Created a NumPy-based density matrix simulator backend (`DensityMatrixSimulator` in `src/density_matrix_simulator.py`) supporting noise channels (amplitude damping, phase damping, depolarizing channel).
  - Integrated deterministic Kraus-channel calculations into `NoiseModel`.

### [x] W-5: Qubit Indexing Order (LSB/MSB) (Done)
- **Problem:** Undocumented LSB (Least Significant Bit) convention causes interop bugs with Qiskit/Cirq.
- **Solution:**
  - Explicitly documented the indexing order in `LANGUAGE.md` under section 5.6.
  - Created `src/utils/converters.py` containing `to_msb_first_dict` and `reorder_state_vector` helper functions.

### [x] W-6: Mid-Circuit Measurements & Classical Control (Done)
- **Problem:** Verification of classical feedback loop support.
- **Solution:**
  - Verified that adaptive feedback loops and mid-circuit measurements (like `measure q0 -> c0` followed by conditional branches `if c0 == 1 { X q1 }`) are fully supported by the compiler, VM jump instructions, and simulators.

### [x] W-7: Standard Format Exporters (OpenQASM 3.0, Quil) (Done)
- **Problem:** Lack of interop with standard QPU hardware.
- **Solution:**
  - Added `Qasm3Exporter` (`src/backend/qasm3_exporter.py`) and `QuilExporter` (`src/backend/quil_exporter.py`).
  - Exposed via CLI build options: `eigen build file.eig --qasm` and `eigen build file.eig --quil`.

### [x] W-8: Circuit Optimizations (Peephole & Commutation) (Done)
- **Problem:** Inefficient deep schemes on QPU.
- **Solution:**
  - Implemented a multi-pass optimization engine in Rust (`native/rust/src/optimizer.rs`) bound via PyO3 to Python `EQIROptimizer`.
  - Optimizes gate patterns: merges consecutive rotations, cancels self-inverse gates (H-H, X-X, Y-Y, Z-Z, CNOT-CNOT), commutes gates (Z/CNOT, X/CNOT), and removes dead rotation gates (angle ≈ 0).

---

## 2. Milestone Integration (M5, M6, M7)

### [x] Phase M5: Generics & Routing (Done)
- **Generics Monomorphization:** Resolve types at compile-time: `func max<T>(a: T, b: T)`.
- **SABRE Routing:** Implement topological SWAP routing over coupling maps (greedy look-ahead).
- **Visualization:** Generate SVG circuit graphics via `eigen viz`.

### [x] Phase M6: VS Code LSP Extension & Diagnostics (Done)
- Pylance/LSP inline checks for syntax and types.
- Provide auto-fixes for syntax and type mismatches.

### [x] Phase M7: MkDocs Material Website (Done)
- Run automated `mike` deployments to publish versioned docs to GitHub Pages.

---

## 3. General Architecture Shift & VM Overhead Reduction
- **[x] Rust Core:** Extracted the compilation pipeline (Lexer, Pratt Parser, Type Checker) into a standalone native Rust extension.
- **[x] CPython GIL Decoupling:** Minimized PyO3 FFI call overhead in hot loops.
- **[x] Structured Logging:** Replaced raw print statements with `loguru`.
- **[x] Print Rounding Fix:** Corrected the VM print opcode to properly output large float/integer values instead of rounding them to zero.
