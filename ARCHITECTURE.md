# Eigen Language — Architecture

This document describes the actual architecture of the Eigen codebase as
implemented in `src/`. It is intended to be accurate: it describes what the
code does, not what it should do.

---

## 1. Overview

Eigen is a domain-specific, hybrid classical-quantum programming language with
a compiled runtime. A `.eig` source file passes through a multi-stage
pipeline (Lexer → Parser → AST → Type Checker → IR → EBC → VM) and is
executed by a stack-based bytecode interpreter that dispatches quantum
operations to one of several simulator backends (dense, sparse, MPS,
stabilizer, density matrix). An optional Rust native extension
(`eigen_native`) accelerates hot paths (lexing, parsing, EQIR optimization,
SVD, SABRE scoring, sparse simulation). When the extension is absent, every
code path falls back to a pure-Python implementation, so Eigen runs without
compiling Rust.

The runtime is hybrid: classical bytecode and quantum operations share the
same VM (`src/backend/vm.py`) and the same instruction stream. Quantum gates
are emitted as `Q_GATE` opcodes; the VM forwards them to the active
`QuantumSimulator`.

---

## 2. Compiler Pipeline

```
Source (.eig)
   │
   ▼
Lexer (src/frontend/lexer.py)            — pure-Python tokenizer
   │                                       (or eigen_native.parse_native for AST)
   ▼
Parser (src/frontend/parser.py)          — Pratt parser → AST
   │                                       parser_recovery.py for multi-error mode
   ▼
AST (src/frontend/ast.py)                 — ProgramNode, FuncDeclNode, GateNode,
   │                                       MatchNode, StringInterpolationNode, ...
   ▼
ImportResolver (src/semantic/import_resolver.py)
   │                                       — LazyModuleLoader + ImportCache
   ▼
TypeChecker (src/semantic/type_checker.py)
   │                                       — Monomorphizer runs after type check
   ▼
MLIR Dialect (src/ir/mlir_dialect.py)    — ASTToMLIRConverter → MLIRModule
   │                                       MLIRToEQIRConverter → EQIRGraph
   ▼
EQIR Graph (src/ir/ir_graph.py)           — nodes: ALLOC, GATE, MEASURE,
   │                                       TRACE, PRINT, ASSERT
   ▼
EQIROptimizer (src/ir/optimizer.py)      — 7-pass worklist rewriter
   │                                       (or eigen_native.optimize_eqir_native)
   ▼
PassManager (src/ir/pass_manager.py)     — optional per-pass statistics
   │
   ▼
EBC Compiler (src/backend/ebc_compiler.py) — AST → list[Instruction] bytecode
   │                                       with peephole + constant folding
   ▼
Bytecode (src/backend/bytecode.py)        — Opcode + Instruction + version check
   │
   ▼
EigenVM (src/backend/vm.py)              — stack-based interpreter
```

### Key files

| Stage | File | Notes |
|-------|------|-------|
| Lexer | `src/frontend/lexer.py` | `KEYWORDS_MAP` is class-level; recognizes gates, types, operators, escape sequences. |
| Parser | `src/frontend/parser.py` | Pratt-style precedence climbing; `parse_statement` is the dispatch root. Handles `qfunc`, `func`, `async func`, `struct`, `enum`, `trait`, `impl`, `type`, `match`, `try/catch/finally`, `parallel/task`, noise declarations, all gate families (1-, 2-, 3-qubit, controlled rotations). |
| AST | `src/frontend/ast.py` | `NATIVE_AVAILABLE` flag enables the Rust fast path in `Parser.parse`. |
| Type Checker | `src/semantic/type_checker.py` | Resolves types and stdlib calls; `Monomorphizer` runs afterwards. |
| Import Resolver | `src/semantic/import_resolver.py` | File-hash-based `ImportCache` and `LazyModuleLoader` detect cycles. |
| MLIR Dialect | `src/ir/mlir_dialect.py` | `ASTToMLIRConverter` produces `MLIRModule` of `MLIRFunction` / `MLIRBlock` / `MLIROp`; `MLIRToEQIRConverter` lowers to EQIR. `convert_function` has an `_inlining_stack` guard against recursive qfunc inlining (emits `RECURSIVE_CALL` placeholder). |
| EQIR Graph | `src/ir/ir_graph.py` | DAG of `EQIRNode` with `parents`/`children` sets; dependency edges derived from qubit/cbit writers. `to_dict`/`from_dict` round-trip is used by the Rust optimizer. |
| EQIR Optimizer | `src/ir/optimizer.py` | 7 rewrites (see §3). Uses deterministic `min(worklist)` popping for byte-stable output. |
| Pass Manager | `src/ir/pass_manager.py` | `OptimizationPass` / `PassManager` / `PassReport`; tracks per-pass `PassStats`. |
| EBC Compiler | `src/backend/ebc_compiler.py` | Lowers AST to `Instruction` list; `peephole=True` folds constant conditions and emits super-instructions (`LOAD_CONST_STORE`, `LOAD_VAR_LOAD_CONST_ADD`, ...). |
| Bytecode | `src/backend/bytecode.py` | 60+ opcodes; `OPCODE_LIST` defines stable ordering; `UnsupportedBytecodeVersionError` for major-version mismatch. |
| VM | `src/backend/vm.py` | Stack-based interpreter (see §4). |

### Incremental compilation

`src/compiler.py` exposes `parse`, `resolve_imports`, `type_check`, `to_eqir`
queries that route through a `QueryDb` (`src/compiler_db.py`) keyed by file
hash. A module-level `_incremental_cache` (`IncrementalCache` from
`compiler_optimizations.py`) caches AST/EQIR/EBC by SHA-256 content hash.
`get_project_hash` walks the import graph so a header change invalidates all
dependents.

---

## 3. EQIR Optimizer Passes

`EQIROptimizer.optimize` runs a worklist loop over `graph.nodes` applying the
following rewrites (in priority order). When `eigen_native` is available, the
whole pass is delegated to `eigen_native.optimize_eqir_native(dict_data)`,
which performs the same rewrites with deterministic lexicographic tie-breaking.

| # | Rule | Effect |
|---|------|--------|
| 1 | Self-inverse cancellation | Adjacent `H H`, `X X`, `Y Y`, `Z Z` on the same qubit cancel. |
| 2 | Rotation merging | Adjacent `RX/RX`, `RY/RY`, `RZ/RZ` angles are summed modulo `2π`. |
| 3 | Dead gate elimination | A rotation whose angle is `≈0` is bypassed. |
| 4 | Peephole `H→X/Z→H` | `H X H` → `Z`, `H Z H` → `X`. |
| 5 | Peephole `S→S→Z`, `T→T→S` | Adjacent identical `S` or `T` collapse to the next Clifford up. |
| 6 | Commutation (Z-CNOT-Z) | `Z q0 → CNOT q0,q1 → Z q0` collapses to just the CNOT. |
| 7 | Commutation (X-CNOT-X) | `X q1 → CNOT q0,q1 → X q1` collapses to just the CNOT. |

The worklist is processed with `min(worklist)` for determinism; `children`
and `parents` are iterated in `sorted(..., key=lambda c: c.id)` order so the
optimized graph is byte-identical across runs regardless of `PYTHONHASHSEED`.

---

## 4. VM Architecture

`EigenVM` (`src/backend/vm.py`) is a stack-based bytecode interpreter.

### State

- `instructions: list[Instruction]` and `ip: int` — program counter.
- `operand_stack` — initially pre-allocated 256 slots, cleared, grown
  dynamically. Bounded by `max_operand_stack_depth` (default `1 << 20`).
- `call_stack: list[ActivationFrame]` — one frame per function call; each
  frame has `locals`, `try_stack`, `return_address`, `current_line`,
  `func_name`.
- `globals: dict` — top-level variables.
- `heap: dict[int, HeapObject]` — structs, maps, arrays, strings; protected
  by `heap_lock`. `VMRef` is the on-stack reference type.
- `try_stack` — exception-handler stack used by `PUSH_TRY`/`POP_TRY`/`THROW`.

### Dispatch modes

`EigenVM(dispatch_mode='fast'|'table')`:

- `'fast'` (default): an inline `if/elif` chain for the top-20 opcodes, with
  a fall-through to a dispatch table for the remainder.
- `'table'`: pure table-driven dispatch, `self.dispatch_table[op](arg)`.

Both modes use the **same** handler methods (`op_halt`, `op_load_const`,
`op_call`, `op_q_gate`, ...) so behavior is identical.

### Hardening

- `RLock`-protected `execute()` body for thread safety.
- `max_instruction_count` — halts runaway loops.
- `instruction_timeout_s` — wall-clock deadline checked in the main loop.
- `deterministic=True` rejects non-deterministic opcodes and pins RNG state.
- `max_operand_stack_depth` — guards against stack-overflow exploits.

### JIT

`JITCompiler` (`src/jit/jit_compiler.py`) is per-VM (state is not shared
across `EigenVM` instances). It:

- Maintains an LRU cache keyed by `get_function_hash(instructions_segment)`.
- Uses a sandboxed `exec` globals dict (`_build_sandbox_globals`) that strips
  `type`/`getattr`/`hasattr`/`isinstance` to prevent MRO-based escapes.
- Emits register-based Python `while` loops for tight backward-jump patterns
  via `native_codegen.py` and `trace_compiler.py`.
- Falls back to the interpreter on a shape guard failure (`jit_deopts += 1`).

JIT is enabled when `opt_level >= 3`. The `HotLoopDetector` tracks
backward-branch frequency; once a branch exceeds the threshold it triggers
JIT compilation.

### Optimization components (§1.1)

- `InlineCache` — monomorphic variable-lookup cache integrated into
  `lookup_var`.
- `FrameCache` — caches `frame.locals` reference, invalidated on `CALL`/`RET`.
- `HotLoopDetector` — backward-branch frequency tracking.
- `ObjectPool` — reusable list pool for `ALLOC_ARRAY`.

### Quantum integration

`Q_ALLOC`, `Q_GATE`, `Q_MEASURE`, `Q_TRACE`, `Q_NOISE` opcodes forward to
`self.simulator` (a `QuantumSimulator`). Noise is applied between gates
through `self.noise_model` (a `NoiseModel` or `NoisePipeline`).

### Async / parallel

- `ASYNC_CALL`, `AWAIT`, `YIELD_TASK` opcodes drive the `async_scheduler`.
- `SPAWN`/`JOIN` opcodes provide CSP-style parallelism via a thread pool.
- `execute_parallel` spawns fresh per-shot VMs (no shared mutable state).

---

## 5. Simulator Backends

`QuantumSimulator` (`src/simulator.py`) is the dispatch surface. It
instantiates a `StateBackend` implementation and forwards gate calls.

| Backend | File | Representation | Cost per gate | Max qubits |
|---------|------|----------------|---------------|-----------|
| Dense (Rust) | `simulator.py:RustStatevectorWrapper` | `2^n` complex vector | `O(2^n)` | ~25 (hard cap) |
| Dense (Python) | `simulator.py:PythonDenseStatevector` | NumPy `2^n` complex | `O(2^n)` | ~25 |
| Sparse | `sparse_simulator.py:SparseQuantumSimulator` | Dict of non-zero amplitudes | `O(|non-zero|)` | ~100+ |
| MPS | `tensor_network/mps.py:MPSSimulator` | Tensor train, bond dim χ (default 64) | `O(χ²)` | ~100+ |
| Stabilizer | `stabilizer_simulator.py:StabilizerSimulator` | Clifford tableau (CHP) | `O(n²)` | ~1000+ |
| Density Matrix | `density_matrix_simulator.py:DensityMatrixSimulator` | `2^n × 2^n` matrix | `O(4^n)` | ~12 |
| GPU | `simulator_optimizations.py:GPUAccelerationSurface` | CuPy/JAX arrays | `O(2^n)` | ~24 |

### Dense state-vector

`PythonDenseStatevector` uses LSB-first indexing with three LRU caches
(`_index_cache`, `_index_cache_2q`, `_index_cache_3q`, max 1024 entries each)
to avoid recomputing the bit-permutation indices for each gate. Buffers
`_buf0`/`_buf1` are reused across `apply_1qubit_gate` calls. `allocate_qubit`
doubles the state vector and clears the caches; a hard `MemoryError` is
raised above 25 qubits.

The `RustStatevectorWrapper` delegates to `eigen_native.RustStatevector`,
which exposes `apply_h/x/y/z/s/t/rx/ry/rz/cnot/cz/swap/ccx/cswap/cp/crx/
cry/crz` and `measure(k, r)`.

### Sparse

`SparseQuantumSimulator` stores amplitudes in a dict keyed by bitstring and
prunes entries below `1e-12` every 100 gates (`_prune_interval`). When
`eigen_native.RustSparseSimulator` is present it is used instead, with
`get_state_list`/`set_state_list` for serialization.

### MPS

`MPSSimulator` represents the state as a chain of rank-3 tensors
`(left_bond, 2, right_bond)`. Default `max_bond_dim = 64`,
`max_truncation_error = 1e-4`. `auto_bond_dim=True` doubles the bond
dimension whenever the discarded weight of an SVD exceeds the tolerance.
SVD uses `eigen_native.compute_svd_native` when available, else
`numpy.linalg.svd`. Truncation error is accumulated in a
`TruncationAccumulator` (`numerical_stability.py`).

### Stabilizer

`StabilizerSimulator` implements the Aaronson-Gottesman CHP algorithm. State
is a `(2n) × (2n+1)` binary tableau: rows `0..n-1` are destabilizer
generators, rows `n..2n-1` are stabilizer generators; the last column is the
phase bit `r`. Supports only the Clifford group:
`{H, S, SDG, X, Y, Z, CNOT, CZ, SWAP, I, SX}`.
Non-Clifford gates (`T, TDG, RX, RY, RZ, CCX, CSWAP, CP, CR*, U1/U2/U3`)
raise `NonCliffordGateError(ValueError)`. `QuantumSimulator(sim_type=
'stabilizer')` catches the error and falls back to dense with a warning.

### Density matrix

`DensityMatrixSimulator` stores the full `2^n × 2^n` density matrix ρ.
Unitary gates apply `ρ → U ρ U†`; noise channels apply `ρ → Σ_k K_k ρ K_k†`
via `_apply_channel`. Trace is renormalized every 100 gates. Cached standard
gates (`H/X/Y/Z/S/T/I2/P0/P1`) avoid per-call allocation. Supports CCX via a
double-controlled operator and CSWAP via three CNOTs.

### GPU

`GPUAccelerationSurface` lazily imports CuPy or JAX. When available and
`n > 8`, `apply_1qubit_gate` delegates to `_gpu_accel.apply_gate_gpu`.

### Backend selection

`backend/sim_selector.py` performs automatic backend selection: circuits
containing only Clifford gates → stabilizer; low entanglement (bond dim
estimation) → MPS; otherwise dense.

---

## 6. ZX-Calculus Module

`src/zx/` provides a ZX-graph representation and a Clifford-tableau-based
equivalence checker.

| File | Contents |
|------|----------|
| `zx_graph.py` | `ZXVertex` (`type ∈ {Z, X, H, Boundary}`, `phase` in units of π), `ZXGraph` with `hadamard_edges` set. |
| `zx_equivalence.py` | `Pauli` tableau tracker; `is_clifford_circuit`; full Clifford-circuit equivalence check via Pauli tracking. |
| `spider_fusion.py` | Spider-fusion rewrite: adjacent same-type spiders merge phases. |
| `local_complementation.py` | Local complementation rewrite on graph vertices. |
| `pivoting.py` | Pivot rewrite on interior Pauli pairs. |
| `exceptions.py` | `IndeterminateEquivalenceError` for circuits the checker cannot decide. |

The ZX module is consumed by `src/equivalence.py` to provide circuit
equivalence checks and by `src/compilation_research.py` for ZX-based
simplification (alongside phase-polynomial optimization, Solovay-Kitaev, and
CNOT synthesis).

---

## 7. Noise Models

`src/noise/` provides composable noise channels.

| File | Contents |
|------|----------|
| `noise_channel.py` | Abstract `NoiseChannel` base; concrete `BitFlip`, `PhaseFlip`, `Depolarizing`, `AmplitudeDamping`, `PhaseDamping`, `ReadoutErrorChannel`, `ReadoutError` (2×2 confusion matrix), and `NoisePipeline` (sequential composition). |
| `noise_model.py` | `NoiseModel` — single-channel model used by the VM. Branches between density-matrix (exact, Kraus-based) and state-vector (stochastic, single-shot sampling) dispatch. |
| `t1t2_model.py` | `T1T2NoiseModel` — physically motivated amplitude + phase damping with per-gate durations from `GATE_TIMES`. Computes `gamma = 1 - exp(-t/T1)` and `lambda = 1 - exp(-t · (1/T2 - 1/(2·T1)))`. |
| `crosstalk_model.py` | `CrosstalkModel` — correlated two-qubit errors (`XX/YY/ZZ/IX/...`) plus spectator-qubit errors via a `CouplingMap`. |
| `device_profile.py` | `DeviceNoiseProfile` — IBM / IonQ calibration profiles (T1/T2 averages, single/two-qubit error rates, readout error, crosstalk). Ships with `ibm_sherbrooke`, `ibm_brisbane`, `ibm_kyiv`, `ibm_osaka` defaults. |

See [NOISE_MODELS.md](NOISE_MODELS.md) for parameter reference and the
stochastic-vs-exact distinction.

---

## 8. Routing and Hardware Compilation

`src/routing/router.py` maps logical qubit operations onto a hardware
`CouplingMap` by inserting `SWAP` gates.

### Coupling maps

`CouplingMap(edges)` stores bidirectional edges as `(min, max)` tuples and
maintains an adjacency dict. Pre-built factories:

- `linear(n)` — chain `0-1-2-...-(n-1)`.
- `grid(rows, cols)` — 2D square lattice.
- `heavy_hex(d)` — IBM's heavy-hex topology with anchor qubits.
- `ibm_eagle()` — 127-qubit IBM Eagle topology (Sherbrooke/Brisbane/Kyiv).
- `ibm_condor()` — large-scale heavy-hex (1121 qubits, `heavy_hex(24)`).
- `ionq_alltoall(n)` — fully-connected IonQ trapped-ion topology.
- `rigetti_ring(n)` — ring with wrap-around edge.
- `google_sycamore(rows=9, cols=6)` — Google Sycamore 2D grid.

`shortest_path` uses BFS (or `eigen_native.fast_shortest_path` when available);
`distance(q1, q2)` returns `len(path) - 1` or `float('inf')`.

### Routers

Three router implementations all expose `.route(circuit_ops, logical_qubits)
-> RoutedCircuit`:

| Router | Strategy |
|--------|----------|
| `BasicSwapRouter` | For each non-adjacent 2-qubit gate, insert SWAPs along the shortest path between the two physical qubits. Deterministic, optimal for single-gate routes. |
| `GreedyRouter` | For each non-adjacent gate, evaluate every coupling-map edge as a candidate SWAP and pick the one minimizing the look-ahead distance (`DEFAULT_LOOKAHEAD = 5` operations). |
| `SabreRouter` | Structure-Aware Bidirectional Router. Maintains a `front_layer` (executable gates) and `extended_layer` (next-up gates). Scores each candidate SWAP as `front_score + lookahead_weight · extended_score`. Uses `eigen_native.fast_sabre_swap_score` when available for deterministic lexicographic tie-breaking. Iteration-capped at `len(ops) × num_qubits × 20`. |

`route_eqir_graph(graph, coupling_map, router_type)` extracts ALLOC and GATE
nodes from an `EQIRGraph` in topological order and dispatches to one of the
routers (`'basic'`, `'greedy'`, `'sabre'`).

### RoutedCircuit

`RoutedCircuit` records `operations` (tuples of `(gate_name, physical_qubits,
args)`), `initial_mapping`, `final_mapping`, and `swap_count`. `summary()`
returns the dict form used by tests and audit logs.

### Hardware targets

The library ships three canonical topologies that match real devices:

- **IBM Eagle** (127 qubits, heavy-hex) — `CouplingMap.ibm_eagle()`.
- **IonQ all-to-all** (n qubits) — `CouplingMap.ionq_alltoall(n)`. No SWAPs
  are ever needed because every pair is adjacent.
- **Rigetti ring** (n qubits) — `CouplingMap.rigetti_ring(n)`. Worst-case
  distance `⌊n/2⌋`; routing is O(n) per long-range gate.

These are exercised in `tests/test_routing.py`.

---

## 9. Key Design Decisions

- **Hybrid VM** rather than a separate quantum simulator driver: classical
  bytecode and quantum opcodes share one instruction stream and one stack,
  simplifying conditional quantum gates (`Q_GATE` carries a `condition`
  tuple referencing a classical cbit).
- **LSB-first qubit ordering** throughout the dense simulator, the density
  matrix, and the MPS path — a single convention avoids index rewrites.
- **EQIR as the optimization substrate** rather than the AST: gates are
  first-class graph nodes with `parents`/`children` sets, so rewrites are
  local and the worklist terminates.
- **Determinism by construction** in the optimizer: `min(worklist)` popping,
  `sorted(children, key=lambda c: c.id)` iteration, and lexicographic
  tie-breaking in `fast_sabre_swap_score` make routing output byte-stable
  w.r.t. `PYTHONHASHSEED`.
- **Rust as a hot-path accelerator only**: every Rust entry point
  (`parse_native`, `optimize_eqir_native`, `fast_shortest_path`,
  `fast_sabre_swap_score`, `compute_svd_native`, `RustStatevector`,
  `RustSparseSimulator`) has a pure-Python fallback. The language runs
  without the native extension.
- **Sandboxed JIT**: the `exec` globals dict strips introspection builtins
  (`type`, `getattr`, `hasattr`, `isinstance`) to prevent MRO-based escapes
  (defense in depth — see `_build_sandbox_globals` in
  `src/jit/jit_compiler.py`).
- **Per-VM JIT state**: the LRU cache and execution counts live on the
  `EigenVM` instance, not at module level, so two programs with identical
  bytecode do not share a compiled artifact.
- **Bytecode versioning** (`BYTECODE_VERSION_MAJOR.MINOR`): older bytecode
  is loadable; a higher major version raises `UnsupportedBytecodeVersionError`
  with a message that references the version mismatch.
- **Stabilizer non-Clifford handling**: `NonCliffordGateError(ValueError)`
  is raised by the bare `StabilizerSimulator`; `QuantumSimulator(sim_type=
  'stabilizer')` catches it and falls back to dense with a warning.
- **T2 ≤ 2·T1 constraint** enforced in both `NoiseModel` and
  `T1T2NoiseModel` constructors (clamps and emits a `warnings.warn`).
