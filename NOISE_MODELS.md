# Eigen Noise Models

Eigen ships a composable noise framework in `src/noise/`. This document
describes the available channels, their parameters, and the difference
between **stochastic** (state-vector) and **exact** (density-matrix) noise
application.

---

## 1. Modules

| Module | File | Purpose |
|--------|------|---------|
| `NoiseModel` | `src/noise/noise_model.py` | Single-channel noise model used by the VM. Branches between stochastic and exact dispatch. |
| `NoiseChannel` family | `src/noise/noise_channel.py` | Abstract base + concrete channels (BitFlip, PhaseFlip, Depolarizing, AmplitudeDamping, PhaseDamping, ReadoutError) and `NoisePipeline` composition. |
| `T1T2NoiseModel` | `src/noise/t1t2_model.py` | Physically motivated T1/T2 relaxation with per-gate durations. |
| `CrosstalkModel` | `src/noise/crosstalk_model.py` | Correlated two-qubit + spectator errors. |
| `DeviceNoiseProfile` | `src/noise/device_profile.py` | IBM / IonQ calibration profiles. |

---

## 2. Channels

### BitFlipChannel

```python
from src.noise.noise_channel import BitFlipChannel

BitFlipChannel(prob: float, rng=None, seed=None)
```

With probability `prob`, applies `X` to the target qubit. Stochastic channel
(single Pauli draw); on a density-matrix simulator it applies the analytic
`{√(1-p)·I, √p·X}` Kraus set instead.

### PhaseFlipChannel

```python
PhaseFlipChannel(prob: float, rng=None, seed=None)
```

With probability `prob`, applies `Z`. Analytic Kraus set:
`{√(1-p)·I, √p·Z}`.

### DepolarizingChannel

```python
DepolarizingChannel(prob: float, rng=None, seed=None)
```

With probability `prob`, applies a uniformly random Pauli from `{X, Y, Z}`.
Analytic Kraus set:
`{√(1-p)·I, √(p/3)·X, √(p/3)·Y, √(p/3)·Z}`.

### AmplitudeDampingChannel

```python
AmplitudeDampingChannel(gamma: float, rng=None, seed=None)
```

T1 relaxation. Kraus operators:

```
K0 = [[1,          0          ],
      [0, √(1 - γ)]]
K1 = [[0,   √γ],
      [0,   0 ]]
```

`gamma = 1 - exp(-t / T1)` where `t` is the gate duration.

On a state-vector simulator without `apply_kraus_channel`, the channel
stochastically applies `K0` (no jump) or `K1` (jump) with probability `γ`
for the jump branch.

### PhaseDampingChannel

```python
PhaseDampingChannel(lambda_val: float, rng=None, seed=None)
```

T2 dephasing. Kraus operators:

```
K0 = [[1,          0          ],
      [0, √(1 - λ)]]
K1 = [[0,          0          ],
      [0,   √λ     ]]
```

`lambda = 1 - exp(-t · (1/T2 - 1/(2·T1)))`.

### ReadoutErrorChannel / ReadoutError

Two readout-error facilities exist:

```python
ReadoutErrorChannel(prob: float, rng=None, seed=None)
```

Symmetric bit-flip at measurement time: with probability `prob`, the observed
outcome is flipped (`1 - outcome`). The channel itself is a no-op at gate
time — it only acts through `apply_readout_noise(outcome)`.

```python
ReadoutError(confusion_matrix, rng=None, seed=None)
```

A 2×2 stochastic confusion matrix `matrix[true] = [p(0|true), p(1|true)]`
where each row sums to 1. `apply(outcome, rng=None)` samples the observed
bit from the row `matrix[outcome]`. When supplied to `NoiseModel(
readout_error=ReadoutError(...))`, the VM consults it on every measurement.

---

## 3. Parameters

| Symbol | Meaning | Range | Where used |
|--------|---------|-------|------------|
| `prob` | Per-gate error probability | `[0, 1]` | `BitFlipChannel`, `PhaseFlipChannel`, `DepolarizingChannel`, `ReadoutErrorChannel` |
| `gamma` | Amplitude-damping strength | `[0, 1]` | `AmplitudeDampingChannel`, `NoiseModel(noise_type='amplitude_damping')` |
| `lambda` (constructor `lambda_val`) | Phase-damping strength | `[0, 1]` | `PhaseDampingChannel`, `NoiseModel(noise_type='phase_damping')` |
| `T1` | Energy relaxation time (μs) | `> 0` | `T1T2NoiseModel`, `DeviceNoiseProfile` |
| `T2` | Dephasing time (μs) | `> 0` and `≤ 2·T1` | `T1T2NoiseModel`, `DeviceNoiseProfile` |

### T1 / T2 mapping to γ and λ

`T1T2NoiseModel` derives the per-gate damping parameters from gate duration:

```python
gamma   = 1 - exp(-t / T1)
lambda  = 1 - exp(-t * (1/T2 - 1/(2*T1)))
```

If `1/T2 - 1/(2·T1) ≤ 0` (which happens when `T2` is very close to `2·T1`),
the rate falls back to `1/T2`. Gate durations (in μs) come from
`GATE_TIMES`:

| Gate class | Duration (μs) |
|------------|----------------|
| Single-qubit (`H, X, Y, Z, S, T, RX, RY, RZ`) | 0.035 |
| Two-qubit (`CNOT, CZ, CP, CR*`) | 0.300 |
| `SWAP` | 0.600 |
| Three-qubit (`CCX, CSWAP`) | 0.900 |
| `measure` | 1.000 |
| `I` | 0.0 |

Defaults: `T1 = 80.0 μs`, `T2 = 50.0 μs`.

### Physical constraint: T2 ≤ 2·T1

Both `NoiseModel.__init__` (when `t1` and `t2` are provided) and
`T1T2NoiseModel.__init__` enforce:

```python
if t2 > 2.0 * t1:
    warnings.warn("T2 exceeds 2*T1; clamping T2 to 2*T1.")
    t2 = 2.0 * t1
```

This is a physical constraint: pure dephasing cannot be slower than twice
the energy-relaxation time. Negative or zero T1/T2 raises `ValueError`.

---

## 4. Stochastic vs Exact Noise

Eigen supports two fundamentally different ways of applying a noise channel,
selected automatically based on the active simulator:

### Stochastic (state-vector)

Used when the simulator's state is a single state-vector (dense, sparse,
MPS, stabilizer). The channel is applied as a **single random sample**:

- `BitFlip`: with probability `prob`, apply `X`.
- `PhaseFlip`: with probability `prob`, apply `Z`.
- `Depolarizing`: with probability `prob`, apply one of `X`/`Y`/`Z` chosen
  uniformly.
- `AmplitudeDamping`: with probability `γ`, apply `K1` (the jump); otherwise
  apply `K0`.
- `PhaseDamping`: with probability `λ` (computed from the analytic
  `p = 1 - √(1 - λ)`), apply `Z`; otherwise apply `K0`.

This is the **trajectory** picture: each shot produces one realization of
the noise. Averaging over many shots converges to the exact channel but a
single shot is *not* the same as the exact evolution. This is dramatically
faster (one state-vector per shot, no matrix algebra) and is the only
practical option for large circuits.

### Exact (density-matrix)

Used when the simulator's state is a density matrix (`DensityMatrixSimulator`
or when `NoiseModel` detects `simulator.sim_type == 'density_matrix'` or
`simulator.density_sim is True`). The channel is applied via its **Kraus
operator sum**:

```
ρ → Σ_k  K_k ρ K_k†
```

This is the **ensemble** picture: a single application of the channel yields
the exact post-noise density matrix. No sampling, no trajectories. Slower
per shot (`O(4^n)` for the density matrix) but exact.

### Detection logic

`NoiseModel.apply_gate_noise` decides as follows:

```python
if getattr(simulator, 'sim_type', None) == 'density_matrix' \
   or (hasattr(simulator, 'density_sim') and simulator.density_sim):
    # Exact path: apply_{bit_flip,phase_flip,depolarizing,
    #                 amplitude_damping,phase_damping}_noise
    ...
else:
    # Stochastic path: draw r ~ U(0,1); apply X/Z/Y/K0/K1 if r < prob
    ...
```

For amplitude/phase damping specifically, the state-vector path first checks
for `simulator.apply_kraus_channel` (used by MPS and any backend that
supports Kraus application on a state-vector); if absent, it falls back to
single-shot `apply_1qubit_gate` sampling.

### Practical guidance

| Goal | Backend | Noise style |
|------|---------|-------------|
| Fast noisy sampling (many shots, large circuits) | `dense` / `sparse` / `mps` | Stochastic |
| Exact expectation values from a single run | `density_matrix` | Exact |
| Comparison with hardware calibration data | `DeviceNoiseProfile` (uses density matrix under the hood) | Exact |
| Stabilizer circuits (Clifford only) | `stabilizer` | Stochastic (Pauli channels only) |

---

## 5. Composing channels: NoisePipeline

`NoisePipeline` chains multiple `NoiseChannel` instances in order:

```python
from src.noise.noise_channel import (
    NoisePipeline, AmplitudeDampingChannel, PhaseDampingChannel,
    ReadoutErrorChannel,
)
from src.noise.t1t2_model import T1T2NoiseModel

pipeline = NoisePipeline(seed=42)
pipeline.add_channel(AmplitudeDampingChannel(gamma=0.01))
pipeline.add_channel(PhaseDampingChannel(lambda_val=0.02))
pipeline.add_channel(ReadoutErrorChannel(prob=0.005))

# T1T2NoiseModel is itself a NoisePipeline subclass
phys = T1T2NoiseModel(t1=120.0, t2=80.0)
phys.add_channel(ReadoutErrorChannel(prob=0.01))
```

`apply_gate_noise(simulator, qubit_name)` runs every channel's
`apply_to_qubit` in order; `apply_readout_noise(outcome)` runs every
`ReadoutErrorChannel` in order.

---

## 6. Examples

### Python API

```python
from src.backend.vm import EigenVM
from src.noise.noise_model import NoiseModel
from src.noise.noise_channel import ReadoutError

# 1% depolarizing noise with a 2% readout confusion matrix
readout = ReadoutError([[0.98, 0.02], [0.02, 0.98]])
nm = NoiseModel(noise_type='depolarizing', noise_prob=0.01, readout_error=readout, seed=42)

vm = EigenVM(noise_model=nm, sim_type='dense', seed=42)
# ... vm.execute(instructions) ...
```

### T1/T2 model with physical gate times

```python
from src.noise.t1t2_model import T1T2NoiseModel

nm = T1T2NoiseModel(t1=120.0, t2=80.0, seed=42)
# A 300 ns CNOT contributes gamma = 1 - exp(-0.3/120) ≈ 0.00249
# A 35 ns H        contributes gamma = 1 - exp(-0.035/120) ≈ 0.000292
```

### Device profile

```python
from src.noise.device_profile import DeviceNoiseProfile

profile = DeviceNoiseProfile.from_ibm('ibm_sherbrooke')
# t1_avg=120, t2_avg=80, single_qubit_error=3e-4, two_qubit_error=5e-3,
# readout_error=1e-2, crosstalk_prob=8e-4
```

### Stochastic vs exact on the same circuit

```python
from src.backend.vm import EigenVM
from src.noise.noise_model import NoiseModel

# Stochastic — each shot is a single trajectory
vm_stoch = EigenVM(
    noise_model=NoiseModel(noise_type='amplitude_damping',
                            noise_prob=0.05, seed=1),
    sim_type='dense', seed=1)

# Exact — a single run gives the post-channel density matrix
vm_exact = EigenVM(
    noise_model=NoiseModel(noise_type='amplitude_damping',
                            noise_prob=0.05, seed=1),
    sim_type='density_matrix', seed=1)
```

Run the same circuit through both; averaging ~1000 stochastic shots
reproduces the exact density matrix to within sampling error.

---

## 7. Limitations

- **State-vector stochastic noise is non-deterministic per shot.** Use a
  fixed `seed` for reproducible runs; otherwise average over many shots to
  converge to the channel.
- **Stabilizer backend** supports only Pauli noise (bit-flip, phase-flip,
  depolarizing). Amplitude/phase damping are non-Clifford and fall back to
  the dense backend when used through `QuantumSimulator`.
- **T2 > 2·T1 is clamped** with a `warnings.warn`; the constructor does not
  raise. Inspect `model.t2` after construction to see the clamped value.
- **`ReadoutErrorChannel` only acts at measurement time** via
  `apply_readout_noise`. It does not modify the quantum state.
- **`NoiseModel.apply_gate_noise` is called per-qubit**, not per-gate, so a
  2-qubit gate triggers two noise applications (one per qubit). Use
  `CrosstalkModel.apply_two_qubit_noise` for correlated errors.
