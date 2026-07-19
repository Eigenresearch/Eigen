"""T1/T2 relaxation noise model with physical circuit timing.

Models amplitude damping (T1) and phase damping (T2) based on
gate execution times. This is a physically motivated noise model
that accounts for the duration of each gate operation.
"""
import math
import warnings
from src.noise.noise_channel import (
    AmplitudeDampingChannel,
    PhaseDampingChannel,
    NoisePipeline,
)

GATE_TIMES = {
    'I': 0.0,
    'X': 0.035,
    'Y': 0.035,
    'Z': 0.035,
    'H': 0.035,
    'S': 0.035,
    'T': 0.035,
    'RX': 0.035,
    'RY': 0.035,
    'RZ': 0.035,
    'CNOT': 0.300,
    'CZ': 0.300,
    'SWAP': 0.600,
    'CCX': 0.900,
    'CSWAP': 0.900,
    'CP': 0.300,
    'CRX': 0.300,
    'CRY': 0.300,
    'CRZ': 0.300,
    'measure': 1.0,
}

DEFAULT_T1 = 80.0
DEFAULT_T2 = 50.0


class T1T2NoiseModel(NoisePipeline):
    """T1/T2 relaxation noise model with physical timing.

    Applies amplitude damping (T1) and phase damping (T2) based on
    the duration of each gate. Longer gates accumulate more decoherence.

    Args:
        t1: T1 relaxation time in microseconds.
        t2: T2 dephasing time in microseconds.
        gate_times: Optional dict overriding default gate durations (in us).
        rng: Random number generator.
        seed: Seed for RNG if rng not provided.
    """

    def __init__(self, t1: float = DEFAULT_T1, t2: float = DEFAULT_T2,
                 gate_times: dict = None, rng=None, seed=None):
        super().__init__(rng=rng, seed=seed)
        if t1 <= 0:
            raise ValueError(f"T1 must be positive, got {t1}")
        if t2 <= 0:
            raise ValueError(f"T2 must be positive, got {t2}")
        if t2 > 2.0 * t1:
            warnings.warn(
                f"T2 ({t2}) exceeds 2*T1 ({2.0 * t1}); clamping T2 to 2*T1. "
                f"This is a physical constraint — T2 cannot exceed 2*T1.",
                stacklevel=2,
            )
            t2 = 2.0 * t1
        self.t1 = t1
        self.t2 = t2
        self.gate_times = dict(GATE_TIMES)
        if gate_times:
            self.gate_times.update(gate_times)

        self._gamma_cache = {}

    def _compute_gamma(self, duration: float) -> float:
        """Compute amplitude damping gamma from T1 and gate duration."""
        if duration <= 0:
            return 0.0
        gamma = 1.0 - math.exp(-duration / self.t1)
        return gamma

    def _compute_lambda(self, duration: float) -> float:
        """Compute phase damping lambda from T2 and gate duration."""
        if duration <= 0:
            return 0.0
        rate = 1.0 / self.t2 - 1.0 / (2.0 * self.t1)
        if rate <= 0:
            rate = 1.0 / self.t2
        lam = 1.0 - math.exp(-duration * rate)
        return lam

    def apply_gate_noise(self, simulator, qubit_name: str, gate_name: str = 'I', **kwargs):
        """Apply T1/T2 noise based on gate duration.

        Args:
            simulator: The quantum simulator instance.
            qubit_name: Name of the qubit.
            gate_name: Name of the gate being applied (determines duration).
        """
        duration = self.gate_times.get(gate_name, 0.035)
        gamma = self._compute_gamma(duration)
        lam = self._compute_lambda(duration)

        if gamma > 0:
            amp_channel = AmplitudeDampingChannel(gamma, rng=self.rng)
            amp_channel.apply_to_qubit(simulator, qubit_name)

        if lam > 0:
            phase_channel = PhaseDampingChannel(lam, rng=self.rng)
            phase_channel.apply_to_qubit(simulator, qubit_name)

    def apply_two_qubit_noise(self, simulator, q1: str, q2: str,
                               gate_name: str = 'CNOT', **kwargs):
        """Apply T1/T2 noise to both qubits of a two-qubit gate."""
        self.apply_gate_noise(simulator, q1, gate_name)
        self.apply_gate_noise(simulator, q2, gate_name)

    def get_gate_duration(self, gate_name: str) -> float:
        """Get the duration of a gate in microseconds."""
        return self.gate_times.get(gate_name, 0.035)
