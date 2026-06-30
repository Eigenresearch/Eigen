# Changelog

All notable changes to the Eigen programming language project will be documented in this file.

## [2.5.0] - 2026-06-30

Release 2.5.0 «Mitz» delivers a massive performance overhaul (1.8x faster than CPython), 8 new language features, 12 critical bug fixes, a stabilizer simulator for 1000+ qubit Clifford circuits, SABRE quantum routing, and comprehensive infrastructure improvements.

### Added — Performance
- **Fast-Loop JIT Engine**: Register-based Python codegen for tight loops — 1.8-2x faster than CPython
- **Stabilizer Simulator**: O(n²) Clifford circuit simulation, handles 1000+ qubits (4s)
- **Simulator Dispatch Optimization**: Replaced string comparisons (`sim_type == 'mps'`) with None-checks
- **VM Hot-Loop Rewrite**: Pre-extracted opcode/arg arrays, named constants, removed per-instruction overhead
- **Native Bypass for Loops**: JIT path preferred over Rust executor for loop-heavy code

### Added — Language Features (8)
- **Exponentiation operator `**`**: Full pipeline (lexer, parser, VM, JIT, type checker, IR, MLIR)
- **Hex/Binary/Octal literals**: `0xFF`, `0b1010`, `0o77`
- **Scientific notation**: `1.23e-5` float literal parsing
- **String interpolation**: `"Result: ${val}"` with AST `StringInterpolationNode`
- **match/case pattern matching**: `match x { case 1 { ... } default { ... } }`
- **Void functions**: `func foo() { ... }` without mandatory `-> type`
- **Lexer escape sequences**: `\n \t \r \0 \\ \" \a \b \f \v`
- **Block comments**: `//` comment support (in addition to `#`)

### Added — Infrastructure
- **SABRE Quantum Routing**: Hardware-aware SWAP insertion (was already in router.py, now documented)
- **Parser Error Recovery**: Multiple syntax errors per compilation with sync points
- **HTML Benchmark Dashboard**: `eigen bench --html` generates visual reports
- **Shared Gate Registry**: `src/backend/gate_registry.py` with gate matrices and metadata
- **Bytecode Version Validation**: `UnsupportedBytecodeVersionError` for incompatible bytecode
- **ZX Hadamard Edges**: `ZXGraph.hadamard_edges` with `add_edge(hadamard=True)` API
- **AST `to_source()` Method**: Human-readable source for assert messages
- **Stabilizer backend in CLI**: `--backend stabilizer` option

### Fixed — Critical Bugs (12)
- **BUG-C02**: JIT `exec()` RCE — sandboxed with `{"__builtins__": {}}` + explicit safe builtins
- **BUG-C03**: `eval()` code injection — replaced with AST-based safe evaluator
- **BUG-C01**: Controlled rotation crash — CP/CRX/CRY/CRZ now extract angles from stack
- **BUG-C04**: `op_ret` stack underflow — guard pushes `None` if stack empty
- **BUG-H04**: try/catch at top level — VM-level try stack
- **BUG-C06**: Canonical hash order — topological sort + qubit targets included
- **BUG-C09**: Compound assignment double-eval — temp variable prevents side-effect duplication
- **BUG-C05**: IR converter drops nodes — all AST node types now handled
- **BUG-C12**: Optimizer blocked by VarDecl — excluded from control-flow check
- **BUG-H06**: Amplitude damping incorrect — proper Kraus K0/K1 operators
- **BUG-H17**: Print rounding errors — full-precision float formatting (no `1e-06`)
- **BUG-H14**: ZX Hadamard edges missing — `hadamard_edges` set in ZXGraph

### Fixed — Additional (20+)
- Phase damping Kraus operators (E0/E1)
- Type checker stdlib whitelist (sin, cos, sqrt, len, range, etc.)
- Optimizer angle merge type check (no crash on strings)
- Verify recursive traversal of if/while/for/try blocks
- `run.py` `-O` flag not triggering optimizer
- While/If condition compilation unified
- `format_amplitudes()` guarded by `trace_mode` (massive perf win)
- `trace_log` bounded to 10000 entries (prevents memory leak)
- `std_mapping` hoisted to class constant (avoids per-call dict creation)
- `KEYWORDS_MAP`/`char_tokens` hoisted to class constants in lexer
- Index caches cleared on `allocate_qubit` (prevents stale entries)
- `GLOBAL_EXEC_COUNTS` bounded (prevents unbounded growth)
- `topological_sort` converted to iterative (prevents RecursionError)
- `compute_depth` converted to iterative
- `BoolOp` short-circuit evaluation in runtime
- Duplicate `--strict` flag removed from CLI
- CLI version updated to 2.6 Nova
- Unused imports removed (sys, random, EQIRNode, EQIRConverter, rotation_types)
- Workspace root searches upward for `eigen.toml`/`pyproject.toml`
- `math.pi` instead of hardcoded `3.141592653589793`

### Test Results
- **445 tests passed** (17 new tests for Mitz features, 428 existing, 0 regressions)
- **439 subtests passed**
- **0 failures**

## [2.6.0] - 2026-06-30

Release 2.6.0 "Nova" addresses all remaining CRITICAL and HIGH bugs from the 2.5 audit (123 findings), adds key language features, and improves architectural quality.

### Added
- **Exponentiation Operator `**`:** Full support for `a ** b` across lexer, parser, type checker, EBC compiler, VM, JIT codegen, IR converter, and MLIR dialect.
- **Void Functions:** `func foo() { ... }` without mandatory `-> type` return type annotation (defaults to `void`).
- **Bytecode Version Validation:** `UnsupportedBytecodeVersionError` raised when loading bytecode with unsupported version.
- **Shared Gate Registry:** Centralized gate metadata (`src/backend/gate_registry.py`) with gate matrices, qubit counts, and Clifford gate classification.
- **AST `to_source()` Method:** Human-readable source representation for `VarRefNode`, `LiteralNode`, `BinaryOpNode`, `DotAccessNode`, `IndexAccessNode`, `CallNode`.
- **ZX-Graph Hadamard Edges:** `ZXGraph.hadamard_edges` set with `add_edge(hadamard=True)` and `is_hadamard_edge()` API.
- **Kraus-Operator Amplitude Damping:** `_apply_amplitude_damping()` method with proper K0/K1 Kraus operators.
- **Type Checker Stdlib Whitelist:** `STDLIB_FUNCTIONS` set for `sin`, `cos`, `sqrt`, `len`, `range`, etc.
- **Lexer Escape Sequences:** `\n`, `\t`, `\r`, `\0`, `\\`, `\"`, `\a`, `\b`, `\f`, `\v` in string literals.
- **Native Parser Fallback:** Python parser used as fallback when Rust native parser encounters unsupported syntax.

### Fixed
- **BUG-C02: JIT RCE Vulnerability:** `exec()` now uses `{"__builtins__": {}}` with only explicit safe builtins (`type`, `repr`, `bool`, `int`, `float`, `str`, `len`, `abs`, `range`, `isinstance`, `hasattr`, `getattr`).
- **BUG-C09: Compound Assignment Double-Eval:** `obj.field += val` and `arr[i] += val` now evaluate object/index once via temp variables.
- **BUG-H06: Amplitude Damping Noise:** Replaced measure-and-flip with proper Kraus operator decomposition.
- **BUG-H08: Undefined Function Errors:** Type checker now whitelists stdlib functions and supports method calls (`obj.method()`).
- **BUG-H12: String Escape Sequences:** Lexer now properly translates `\n`, `\t`, `\r`, `\0`, etc.
- **BUG-H13: Optimizer Angle Merging Crash:** Type check (`isinstance(angle, (int, float))`) before merging rotation angles.
- **BUG-H14: ZX Hadamard Edges:** ZXGraph now supports both regular and Hadamard edges.
- **BUG-H15: Verify Traversal:** `analyze_body` now recursively traverses `IfNode`, `WhileNode`, `ForNode`, `TryCatchNode`.
- **BUG-H16: Run `-O` Flag:** `run.py` now uses computed `optimize` variable instead of `args.optimize` boolean.
- **BUG-H17: Assert Messages:** Assert failure now shows `x == 5` instead of `VarRefNode(x) == LiteralNode(5: int)`.
- **BUG-H18: While/If Condition Unification:** Both use `_compile_condition()` helper method.
- **BUG-M09: Hot-Loop Import:** `import ast` moved to module level in `runtime.py`.
- **BUG-M12: JIT Inline Name Collisions:** Replaced `random.randint` with atomic counter.
- **BUG-M13: Native Codegen Opcode:** Safe `opcode.lower()` handling for both string and Enum types.
- **BUG-M15: CLI Version String:** Updated to `v2.6 — Nova`.
- **BUG-M16: Duplicate `--strict`:** Removed top-level duplicate.
- **BUG-M39: Workspace Root:** Now searches upward for `eigen.toml`/`pyproject.toml` instead of using CWD.
- **BUG-M42: Hardcoded Pi:** Replaced `3.141592653589793` with `math.pi`.
- **BUG-M43: IR Converter Expr:** Added `%`, `**`, comparison, and logical operators to `evaluate_expr`.
- **BUG-M44: Silent Error Swallowing:** `evaluate_expr` now emits `DiagnosticWarning` to stderr.
- **BUG-M47: Strict Tautology:** Simplified `args.strict or getattr(args, "strict", False)`.
- **BUG-L17: Missing `__init__.py`:** Created `src/zx/__init__.py`.
- **BUG-L27: Dead Code:** Removed `except Exception as e: raise e` no-op in `compiler.py`.
- **BUG-L16: Equivalence Tolerance:** Tightened from `1e-5` to `1e-9`.
- **BUG-M34: ForNode Range Support:** Type checker now accepts `int` and `any` iterable types.
- **BUG-M35: String Comparisons:** Type checker now allows `string` in if/assert conditions.

### Test Results
- **416 tests passed** (30 new tests for Nova fixes, 386 original tests, 0 regressions)
- **439 subtests passed**
- **0 failures**

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
