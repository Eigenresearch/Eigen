"""Crosstalk noise model for two-qubit correlated errors.

Models spatially correlated errors that occur when two-qubit gates
induce errors on neighbouring qubits. This is critical for modelling
real superconducting quantum processors where microwave pulses for
one gate can affect nearby qubits.
"""
from src.noise.noise_channel import NoisePipeline


class CrosstalkModel(NoisePipeline):
    """Crosstalk noise model for correlated two-qubit errors.

    When a two-qubit gate is applied, this model introduces correlated
    errors on both qubits with a configurable probability. Additionally,
    spectator qubits (neighbours of the gate qubits) can experience
    single-qubit errors at a reduced rate.

    Args:
        crosstalk_prob: Probability of correlated error on gate qubits.
        spectator_prob: Probability of error on spectator qubits.
        spectator_distance: Maximum graph distance for spectator effect.
        coupling_map: Optional CouplingMap for spectator detection.
        rng: Random number generator.
        seed: Seed for RNG.
    """

    def __init__(self, crosstalk_prob: float = 0.001,
                 spectator_prob: float = 0.0001,
                 spectator_distance: int = 1,
                 coupling_map=None,
                 rng=None, seed=None):
        super().__init__(rng=rng, seed=seed)
        self.crosstalk_prob = crosstalk_prob
        self.spectator_prob = spectator_prob
        self.spectator_distance = spectator_distance
        self.coupling_map = coupling_map

    def apply_gate_noise(self, simulator, qubit_name: str, **kwargs):
        gate_name = kwargs.get('gate_name', 'I')
        if gate_name in ('CNOT', 'CZ', 'SWAP', 'CP', 'CRX', 'CRY', 'CRZ'):
            if self.rng.random() < self.crosstalk_prob:
                error_type = self.rng.choice(['X', 'Y', 'Z'])
                getattr(simulator, error_type)(qubit_name)

    def apply_two_qubit_noise(self, simulator, q1: str, q2: str, **kwargs):
        if self.crosstalk_prob <= 0.0:
            return

        if self.rng.random() < self.crosstalk_prob:
            error = self.rng.choice(['XX', 'YY', 'ZZ', 'IX', 'XI', 'IZ', 'ZI'])
            for i, gate in enumerate(error):
                if gate != 'I':
                    target = q1 if i == 0 else q2
                    getattr(simulator, gate)(target)

        if self.coupling_map and self.spectator_prob > 0:
            self._apply_spectator_errors(simulator, q1, q2)

    def _apply_spectator_errors(self, simulator, q1: str, q2: str):
        """Apply errors to spectator qubits near the gate qubits."""
        all_qubits = getattr(simulator, 'qubit_map', {})
        if not all_qubits:
            return

        gate_indices = set()
        for name, idx in all_qubits.items():
            if name in (q1, q2):
                gate_indices.add(idx)

        for name, idx in all_qubits.items():
            if name in (q1, q2):
                continue
            is_spectator = False
            for gi in gate_indices:
                dist = self._qubit_distance(gi, idx)
                if dist is not None and dist <= self.spectator_distance:
                    is_spectator = True
                    break
            if is_spectator and self.rng.random() < self.spectator_prob:
                error_type = self.rng.choice(['X', 'Y', 'Z'])
                getattr(simulator, error_type)(name)

    def _qubit_distance(self, q1_idx: int, q2_idx: int):
        """Compute distance between qubits using coupling map."""
        if self.coupling_map is None:
            return None
        return self.coupling_map.distance(q1_idx, q2_idx)
