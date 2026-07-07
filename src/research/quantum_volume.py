"""
P3 §12.1 — Quantum Volume estimation (surface-level).

Roadmap (`sol.md` "12.1 Quantum Research Tools"):
    - [ ] **Quantum volume estimation** — метрика производительности QPU

The quantum volume (QV) protocol (Cross et al. "Validating quantum
computers using randomized model circuits", Phys. Rev. A 2019) is:

  1. For each trial:
     a. Generate a depth-width model circuit on `width` qubits
        (depth == width — a "square random circuit").
     b. Apply to the |0...0> computational basis state.
     c. Sample shot outcomes. Heavy outputs are the bitstrings whose
        ideal probability exceeds the median.
     d. Count the fraction of shots landing in the heavy set.
  2. The trial is "successful" when the heavy-output ratio exceeds
     2/3 with 95% confidence (i.e., (ratio - 2/3) σ >= some z). For
     deterministic surface-level implementation, we report the ratio
     and the confidence-bound check directly.

This module exposes:
  * `random_quantum_volume_circuit(width, rng)` → list of (gate, *args)
    tuples that can be applied to a QuantumSimulator instance.
  * `apply_circuit(sim, circuit)` — apply a generated circuit to a
    QuantumSimulator instance.
  * `heavy_output_set(width, sim)` → set of int bitstrings (encoded as
    integer indices) representing the heavy set for the current
    simulator state.
  * `quantum_volume(width, trials=10, shots=1000, rng=None,
    sim_seed=None)` — run the full protocol and return an
    `EstimateResult` dataclass with width, mean_ratio, num_success,
    num_trials, heavy_count_each, and a `succeed: bool` indicating
    whether the QV >= width threshold is met cross-trial.

Surface-level: no noise model is applied — for this codebase's
purpose, noise channels are an orthogonal §12.1 sub-feature ("Error
mitigation techniques: ZNE, PEC, M3"). The QV estimate here measures
the IDEAL heavy-output ratio, which is ~0.93 for a Haar-random
circuit at depth == width (Cross et al.). We surface the API so
downstream work can layer noise + mitigation without restructuring.
"""
from __future__ import annotations

import dataclasses
import math
import random
import statistics
import typing

# Import lazily inside functions to keep this module importable in
# environments where the simulator chain has heavy side effects. Most
# callers already depend on `src.simulator`, though — and importing
# at top-level is fine; QuantumSimulator is lightweight on init.


# ---------------------------------------------------------- circuit gen

# A "gate step" is a tuple: (gate_name, *args). The args are positional
# arguments to the corresponding `QuantumSimulator.<gate_name>` method.
GateStep = typing.Tuple[str, typing.Any]


def random_quantum_volume_circuit(width: int,
                                  rng: random.Random,
                                  ) -> typing.List[GateStep]:
    """Generate a "square" model circuit of depth == width on
    `width` qubits. Each layer consists of:
      1. Random 1-qubit gates {X, Y, Z, H, S, T, RX, RY, RZ} on each
         qubit (rotation gates sampled with theta ∈ [0, 2π)).
      2. A near-neighbor entangling layer: pair up consecutive qubits
         (offset alternated per layer) and apply CNOT.

    The shape (depth == width) follows the QV convention from Cross
    et al. 2019. Returns a list of `(gate_name, *target_args)` steps
    suitable for `apply_circuit(sim, ...)`.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    qubit_names = [f"q{i}" for i in range(width)]
    steps: typing.List[GateStep] = []
    one_q_gates = ["X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ"]

    # Depth == width by convention.
    for layer in range(width):
        # Layer of random single-qubit gates.
        for i in range(width):
            gate = rng.choice(one_q_gates)
            if gate in ("RX", "RY", "RZ"):
                theta = rng.uniform(0.0, 2.0 * math.pi)
                steps.append((gate, qubit_names[i], theta))
            else:
                steps.append((gate, qubit_names[i]))
        # Entangling layer: pair up consecutive qubits with offset
        # alternating per layer so every qubit is entangled with both
        # of its neighbors over two layers.
        offset = layer % 2
        i = offset
        while i + 1 < width:
            control = qubit_names[i]
            target = qubit_names[i + 1]
            steps.append(("CNOT", control, target))
            i += 2
    return steps


def apply_circuit(sim, circuit: typing.Iterable[GateStep]) -> None:
    """Apply a generated circuit (list of (gate, *args) steps) to a
    QuantumSimulator instance.

    The function dispatches via `getattr(sim, gate_name)(*args)`; this
    is intentionally permissive so callers may pass mocked
    simulators that expose only the gate methods used by the
    generated circuits.
    """
    for step in circuit:
        gate = step[0]
        args = step[1:]
        method = getattr(sim, gate)
        method(*args)


def _allocate(width: int, sim) -> None:
    for i in range(width):
        sim.allocate_qubit(f"q{i}")


def _state_index_probabilities(sim) -> typing.List[float]:
    """Compute the squared magnitude of every amplitude in the
    QuantumSimulator's state vector. The order matches the
    qubit-index mapping defined in `simulator.qubit_map`.
    """
    state = sim.get_state_vector()
    return [abs(amp) ** 2 for amp in state]


def heavy_output_set(sim) -> typing.Set[int]:
    """Return the set of heavy-output bitstring indices (as
    integer indices into the state vector) for the current simulator
    state, per Cross et al. 2019. Heavy outputs are those whose
    ideal probability exceeds the median.
    """
    probs = _state_index_probabilities(sim)
    if not probs:
        return set()
    median = statistics.median(probs)
    return {i for i, p in enumerate(probs) if p > median}


def sample_outcome(sim) -> int:
    """Sample a single shot outcome by collapsing via the
    backend-agnostic Monte Carlo in QuantumSimulator's RNG.

    We DON'T use `sim.measure(q)` because that mutates state on each
    call (and re-running the model circuit per shot is wasteful).
    Instead we draw a single integer index from the current
    probability distribution.
    """
    probs = _state_index_probabilities(sim)
    if not probs:
        return 0
    # Trim floating-point residue and renormalize to a clean sum-1.
    total = sum(probs)
    if total <= 0:
        # All amplitudes zero — degenerate; pick zero.
        return 0
    probs = [p / total for p in probs]
    r = sim.rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return len(probs) - 1


# -------------------------------------------------------------- runner

@dataclasses.dataclass
class EstimateResult:
    width: int
    trials: int
    shots: int
    num_success: int
    heavy_ratio_per_trial: typing.List[float]
    mean_ratio: float
    succeed: bool  # cross-trial confidence: ratio > 2/3 with z >= 2.

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def quantum_volume(width: int, *, trials: int = 10, shots: int = 1000,
                   rng: typing.Optional[random.Random] = None,
                   sim_seed: typing.Optional[int] = None,
                   ) -> EstimateResult:
    """Run the QV protocol for `trials` independent square circuits
    of width `width`. Returns an `EstimateResult`.

    Per the Cross et al. definition, each trial:
      * Generates a fresh random circuit of depth == width.
      * Builds a fresh QuantumSimulator and applies the circuit.
      * Samples `shots` shot outcomes.
      * Counts the fraction of outcomes that land in the heavy set.

    The overall QV >= width threshold is met when the heavy
    output probability exceeds 2/3 with at least 2σ confidence
    across all trials. We compute:
        mu = mean of per-trial ratios
        sigma = standard deviation across trials
        z = (mu - 2/3) / sigma
        succeed = z > 2
    """
    if rng is None:
        rng = random.Random()
    if width < 1:
        raise ValueError("width must be >= 1")
    # Local import here so importing `src.research.quantum_volume`
    # doesn't transitively require every simulator backend to be
    # loadable.
    from src.simulator import QuantumSimulator
    ratios: typing.List[float] = []
    for t in range(trials):
        sim = QuantumSimulator(sim_type="dense", seed=sim_seed)
        _allocate(width, sim)
        circuit = random_quantum_volume_circuit(width, rng)
        apply_circuit(sim, circuit)
        heavy = heavy_output_set(sim)
        successes = 0
        for _ in range(shots):
            outcome = sample_outcome(sim)
            if outcome in heavy:
                successes += 1
        ratios.append(successes / shots)
    mu = statistics.fmean(ratios) if ratios else 0.0
    sigma = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
    if sigma > 0:
        z = (mu - 2.0 / 3.0) / sigma
    elif mu > 2.0 / 3.0:
        z = float("inf")
    else:
        z = 0.0
    num_success = sum(1 for r in ratios if r > 2.0 / 3.0)
    return EstimateResult(
        width=width,
        trials=trials,
        shots=shots,
        num_success=num_success,
        heavy_ratio_per_trial=ratios,
        mean_ratio=mu,
        succeed=z >= 2.0,
    )
