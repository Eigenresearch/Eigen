# Migration Guide: Eigen 2.6 «Misery» → 2.7 «Meridian»

This guide covers all breaking changes, new features, and migration steps for upgrading from Eigen 2.6 to 2.7.

---

## Breaking Changes

### 1. Bytecode Version Format

**2.6:** `BYTECODE_VERSION = 1` (single int)

**2.7:** `BYTECODE_VERSION_MAJOR = 1`, `BYTECODE_VERSION_MINOR = 0` (major.minor format)

The `validate_bytecode_version()` function now uses `check_bytecode_compatibility()` which returns a `CompatibilityStatus`:
- `EXACT` — same version, fully compatible
- `FORWARD_MINOR` — same major, higher minor (loadable, unknown opcodes may fail at runtime)
- `BACKWARD` — older version, always loadable
- `INCOMPATIBLE_FUTURE` — higher major, raises `UnsupportedBytecodeVersionError`

**Migration:** Existing serialized bytecode (version=1) continues to work — `1` is parsed as `(1, 0)` which matches the supported version exactly.

### 2. VM Variable Lookup — Inline Cache

The `lookup_var()` method now uses an `InlineCache` internally. This is transparent for API consumers but changes performance characteristics:

- First lookup of each variable name populates the cache (slightly slower)
- Subsequent lookups hit the cache (faster)
- If a variable's scope changes (frame → globals), the cache is invalidated automatically

**Migration:** No action needed. The cache is internal and auto-managed.

### 3. VM STORE_VAR — FrameCache

`STORE_VAR` now uses `FrameCache` which caches the current frame's `locals` dict reference. The cache is invalidated on `CALL` and `RET`.

**Migration:** No action needed.

### 4. MLIR Recursion Guard

`MLIRToEQIRConverter.convert_function()` now has a recursion guard (`_inlining_stack`). Self-recursive and mutually-recursive qfuncs emit a `RECURSIVE_CALL` placeholder gate instead of infinite inlining.

**Migration:** If your code relied on recursive qfunc inlining producing a specific graph structure, check that the `RECURSIVE_CALL` placeholder is handled by downstream passes.

### 5. test_aot Subprocess Timeout

The subprocess timeout for AOT compilation increased from 60s to 120s. `test_aot_seed_determinism` now catches `subprocess.TimeoutExpired` and skips gracefully.

**Migration:** No action needed — tests that previously timed out will now either pass or skip.

---

## New Features in 2.7

### VM Optimizations

| Feature | Module | Integration Point |
|---------|--------|-------------------|
| InlineCache | `vm_optimizations.py` | `vm.py:lookup_var()` |
| FrameCache | `vm_optimizations.py` | `vm.py:STORE_VAR` |
| HotLoopDetector | `vm_optimizations.py` | `vm.py:JMP/JIF` |
| ObjectPool | `vm_optimizations.py` | `vm.py:op_alloc_array` |

### Compiler Optimizations

| Feature | Module | Integration Point |
|---------|--------|-------------------|
| IncrementalCache | `compiler_optimizations.py` | `compiler.py:compile_to_eqir()` |
| ImportCache | `compiler_optimizations.py` | `import_resolver.py:resolve_module_file()` |
| LazyModuleLoader | `compiler_optimizations.py` | `import_resolver.py:parse_file()` |
| ParallelCompiler | `parallel_compiler.py` | `compiler.py:compile_multiple_parallel()` |

### Simulator Optimizations

| Feature | Module | Integration Point |
|---------|--------|-------------------|
| GPUAccelerationSurface | `simulator_optimizations.py` | `simulator.py:apply_1qubit_gate()` |
| apply_gate_inplace | `simulator_optimizations.py` | Available as utility function |
| optimize_measurement_order | `simulator_optimizations.py` | `simulator.py:measure_multiple()` |
| PulseSchedule | `pulse_control.py` | `simulator.py:get_pulse_schedule()` |

### FFI

| Target | Module | Output |
|--------|--------|--------|
| Python | `ffi.py:PythonFFIBindingEmitter` | Working `ctypes` bindings with auto library load |
| Rust | `ffi.py:RustFFIEmitter` | Compilable `#[no_mangle] extern "C"` (no `unimplemented!()`) |
| C | `ffi.py:CHeaderEmitter` | Portable C99 header with include guards |
| WASM | `ffi.py:WASMModule` | WebAssembly text format (`.wat`) |

### Debugging

`DebugSession` in `debugger/dap_server.py` provides:
- `set_breakpoint(source, line)` / `clear_breakpoint(source, line)`
- `step_into()` / `step_over(depth)` / `step_out(depth)` / `continue_execution()`
- `handle_dap_request(request)` — full DAP protocol handler
- Integrated into `EigenVM` via `enable_debug()` / `set_breakpoint()`

### CLI

| Feature | Command |
|---------|---------|
| Auto-completion | `eigen completions --shell bash/zsh/fish/powershell` |
| Playground | `EigenPlayground.repl_loop()` |
| Code migration | `CodeMigrator.migrate(source)` / `migrate_file(path)` |

### Documentation

- `generate_tutorial("markdown"|"html")` — 6-step getting-started tutorial
- `generate_video_tutorial_index()` — video tutorial catalog
- `generate_browser_playground()` — self-contained HTML playground

### Research Tools

- **Quantum tomography** — `state_tomography()`, `process_tomography()` with chi matrix
- **Error mitigation** — `zero_noise_extrapolation()`, `probabilistic_error_cancellation()`, `m3_measurement_mitigation()`
- **Compilation research** — `PhasePolynomial`, `ZXSimplifier`, `solovay_kitaev()`, `synthesize_cnot_circuit()`, `best_layout()`, `schedule_circuit()`
- **Seed management** — `SeedManager`, `GlobalSeedManager` with per-component SHA-256 derivation
- **Experiment tracking** — `ExperimentTracker` with JSON/LaTeX export
- **Project scalability** — `WorkspaceManifest`, `PackageGraph`, `render_dot_dag()`, `render_ascii_layers()`
- **Mutation testing** — `mutmut` config and `MutationTestResult` parser

### Testing

| Addition | Count |
|----------|-------|
| Test files added | 15+ |
| New tests | ~700 |
| Total tests | 2410 |
| Hypothesis tests | 12 |
| Property-based tests | 19 |

---

## Dependency Changes

### New Optional Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `hypothesis` | Property-based testing | `pip install hypothesis` |
| `mutmut` | Mutation testing | `pip install mutmut` |
| `mpi4py` | MPI distributed simulation | `pip install mpi4py` |
| `pymupdf` | PDF page counting (paper build) | `pip install pymupdf` |

None of these are required — Eigen 2.7 runs without them. Missing optional dependencies trigger graceful fallbacks.

---

## Migration Checklist

- [ ] Update `pyproject.toml` version to `2.7.0`
- [ ] Run `pytest tests/ -q` — all 2410 tests should pass
- [ ] If using serialized bytecode: verify `validate_bytecode_version()` still accepts your files
- [ ] If using recursive qfuncs: check for `RECURSIVE_CALL` placeholder gates in EQIR output
- [ ] If using AOT: subprocess timeout is now 120s (was 60s)
- [ ] Install optional deps if needed: `pip install hypothesis mutmut mpi4py`
- [ ] Review new CLI commands: `completions`, `playground`, `migrate`
- [ ] Update CI to install optional deps for full test coverage

---

## FAQ

**Q: Do I need to recompile my Rust native extension?**
A: No — the Rust extension API is unchanged. `eigen_native` from 2.6 works with 2.7.

**Q: Will my 2.6 bytecode files work?**
A: Yes — bytecode version 1 is parsed as (1, 0) and is fully compatible.

**Q: Do the VM optimizations change execution results?**
A: No — InlineCache, FrameCache, and ObjectPool are performance-only. They do not change semantics.

**Q: Is the paper.pdf included in the git repo?**
A: No — paper.pdf is 1.5 MB and is excluded from git. It's available on [Google Drive](https://drive.google.com/file/d/11rhrJ0xqsZDynLpQujr6TrQ8kS77zLjq/view?usp=sharing).

---

*Eigen Research / Eigen Labs — July 2026*
