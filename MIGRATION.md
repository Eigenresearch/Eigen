# Migration Guide: Eigen 2.5 (Mitz) to 2.6 (Misery)

This document guides you through breaking changes and migration steps when upgrading from Eigen 2.5 to 2.6.

## Version Synchronization

All version identifiers are now unified to `2.6.0`:

| Component | 2.5 (old) | 2.6 (new) |
|---|---|---|
| `pyproject.toml` | `2.5.0` | `2.6.0` |
| CLI version string | `v2.6 — Nova` | `v2.6 — Misery` |
| `Cargo.toml` (native) | `2.4.0` | `2.6.0` |
| CHANGELOG | `[2.6.0]` | `[2.6.0]` (synchronized) |

## Breaking Changes

### 1. Stabilizer Simulator — Non-Clifford Gate Handling

**Before (2.5):** Calling `T()`, `RX()`, `RY()`, `RZ()`, `CCX()`, `CSWAP()`, `CP()`, `CRX()`, `CRY()`, `CRZ()` on a `StabilizerSimulator` raised a raw `ValueError` with no recovery path.

**After (2.6):**
- A `NonCliffordGateError` is raised (subclass of `ValueError` for backward compatibility).
- A pre-flight circuit analysis (`check_circuit_compatibility()`) is available to detect non-Clifford gates before execution.
- When using `QuantumSimulator(sim_type='stabilizer')`, non-Clifford gates trigger an automatic fallback to the dense state-vector simulator with a warning.

**Migration:**
```python
# Old code that would crash:
sim = QuantumSimulator(sim_type='stabilizer')
sim.T('q0')  # ValueError in 2.5

# New behavior in 2.6:
sim = QuantumSimulator(sim_type='stabilizer')
sim.T('q0')  # Auto-fallback to dense, prints warning

# To check compatibility explicitly:
from src.stabilizer_simulator import StabilizerSimulator, CLIFFORD_GATES
gates = [('T', ['q0'], []), ('CNOT', ['q0', 'q1'], [])]
incompatible = StabilizerSimulator.check_circuit_compatibility(gates)
if incompatible:
    print(f"Non-Clifford gates detected: {incompatible}")
```

### 2. `heavy_hex()` Coupling Map — Real IBM Topology

**Before (2.5):** `CouplingMap.heavy_hex(n)` returned `CouplingMap.grid(n, n)` — a simple grid, not an actual heavy-hex topology.

**After (2.6):** `CouplingMap.heavy_hex(d)` generates a genuine IBM heavy-hex topology with `d` qubits per side, featuring the characteristic staggered pattern with "heavy" hexagonal cells.

**Migration:**
- If your code relied on `heavy_hex(n)` producing an `n*n` grid, use `CouplingMap.grid(n, n)` explicitly instead.
- If you want a specific IBM device topology, use `CouplingMap.ibm_eagle()` for the 127-qubit Eagle processor.

### 3. Noise Model — Extended API

**Before (2.5):** `NoiseModel` only supported `noise_type` string parameter with 5 basic models.

**After (2.6):** New `NoiseChannel` abstract class and composition pipeline:
- `T1T2NoiseModel` for physical T1/T2 relaxation with circuit timing.
- `CrosstalkModel` for two-qubit correlated errors.
- `DeviceNoiseProfile.from_ibm(backend_name)` for loading real hardware calibration data.
- Custom Kraus channels via user-defined `NoiseChannel` subclasses.

**Migration:** The original `NoiseModel` API is preserved for backward compatibility. New models are opt-in.

### 4. GPU Engine — Structured Logging

**Before (2.5):** `GPUEngine` used `print()` statements for diagnostics.

**After (2.6):** Uses Python `logging` module with a dedicated `eigen.gpu` logger. Configure verbosity via standard logging configuration.

**Migration:**
```python
import logging
logging.getLogger('eigen.gpu').setLevel(logging.DEBUG)  # Enable debug output
```

### 5. MPS Simulator — Auto Bond Dimension

**Before (2.5):** `MPSSimulator(max_bond_dim=32)` with a hardcoded default and no truncation error handling.

**After (2.6):**
- Default `max_bond_dim` increased to 64.
- `auto_bond_dim` mode automatically increases bond dimension when truncation error exceeds a threshold.
- Warnings emitted when simulation accuracy may be degraded.
- Configurable via `max_truncation_error` parameter.

### 6. Coverage Configuration

**Before (2.5):** Coverage `omit` list excluded critical components (simulator, runtime, compiler, VM, IR).

**After (2.6):** Critical components are now included in coverage. The `fail_under` threshold remains at 60%.

## New Features

- **Inno Setup Windows Installer** with GUI wizard, component selection, PATH management, and `.eig` file association.
- **Real IBM device topologies**: `CouplingMap.ibm_eagle()`, `CouplingMap.ibm_condor()`, `CouplingMap.ionq_alltoall()`, `CouplingMap.rigetti_ring()`, `CouplingMap.google_sycamore()`.
- **Advanced Noise Engine**: `NoiseChannel` abstract class, `T1T2NoiseModel`, `CrosstalkModel`, `DeviceNoiseProfile`.
- **Equivalence Checker Documentation**: Canonical hash is documented as necessary-but-not-sufficient for equivalence.

## Deprecation Warnings

The following will be removed in a future release:
- Direct `print()` usage in GPU engine (use `logging` module instead).
- `NoiseModel` string-based `noise_type` parameter (prefer `NoiseChannel` composition).

## Checklist

- [ ] Update any code that catches `ValueError` from stabilizer non-Clifford gates to also handle `NonCliffordGateError`.
- [ ] Replace `CouplingMap.heavy_hex(n)` with `CouplingMap.grid(n, n)` if you relied on grid behavior.
- [ ] Configure `eigen.gpu` logger if you depended on `print()` output from GPU engine.
- [ ] Review MPS `max_bond_dim` settings if you relied on the old default of 32.
