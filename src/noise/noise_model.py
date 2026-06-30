import random
import math

class NoiseModel:
    def __init__(self, noise_type: str = None, noise_prob: float = 0.0, rng=None):
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        self.rng = rng if rng is not None else random.Random()

    def apply_gate_noise(self, simulator, qubit_name: str):
        if not self.noise_type or self.noise_prob <= 0.0:
            return

        if self.noise_type == 'readout_error':
            return

        if getattr(simulator, 'sim_type', None) == 'density_matrix' or (hasattr(simulator, 'density_sim') and simulator.density_sim):
            density_sim = getattr(simulator, 'density_sim', simulator)
            if self.noise_type == 'bit_flip':
                density_sim.apply_bit_flip_noise(qubit_name, self.noise_prob)
            elif self.noise_type == 'phase_flip':
                density_sim.apply_phase_flip_noise(qubit_name, self.noise_prob)
            elif self.noise_type == 'depolarizing':
                density_sim.apply_depolarizing_noise(qubit_name, self.noise_prob)
            elif self.noise_type == 'amplitude_damping':
                density_sim.apply_amplitude_damping_noise(qubit_name, self.noise_prob)
            elif self.noise_type == 'phase_damping':
                density_sim.apply_phase_damping_noise(qubit_name, self.noise_prob)
            return

        r = self.rng.random()
        if r < self.noise_prob:
            if self.noise_type == 'bit_flip':
                simulator.X(qubit_name)
            elif self.noise_type == 'phase_flip':
                simulator.Z(qubit_name)
            elif self.noise_type == 'depolarizing':
                ch = self.rng.choice(['X', 'Y', 'Z'])
                if ch == 'X':
                    simulator.X(qubit_name)
                elif ch == 'Y':
                    simulator.Y(qubit_name)
                elif ch == 'Z':
                    simulator.Z(qubit_name)
            elif self.noise_type == 'amplitude_damping':
                self._apply_amplitude_damping(simulator, qubit_name, self.noise_prob)
            elif self.noise_type == 'phase_damping':
                self._apply_phase_damping(simulator, qubit_name, self.noise_prob)

    def apply_readout_noise(self, outcome: int) -> int:
        if self.noise_type == 'readout_error' and self.noise_prob > 0.0:
            r = self.rng.random()
            if r < self.noise_prob:
                return 1 - outcome
        return outcome

    def _apply_amplitude_damping(self, simulator, qubit_name: str, gamma: float):
        k0 = [[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]]
        k1 = [[0.0, math.sqrt(gamma)], [0.0, 0.0]]
        if hasattr(simulator, 'apply_kraus_channel'):
            simulator.apply_kraus_channel(qubit_name, [k0, k1])
        elif hasattr(simulator, 'apply_1qubit_gate'):
            r = self.rng.random()
            if r < gamma:
                outcome = simulator.measure(qubit_name)
                if outcome == 1:
                    simulator.X(qubit_name)
            else:
                simulator.apply_1qubit_gate(qubit_name, k0)

    def _apply_phase_damping(self, simulator, qubit_name: str, lambda_val: float):
        p = 1.0 - math.sqrt(1.0 - lambda_val)
        k0 = [[1.0, 0.0], [0.0, math.sqrt(1.0 - lambda_val)]]
        k1 = [[0.0, 0.0], [0.0, math.sqrt(lambda_val)]]
        if hasattr(simulator, 'apply_kraus_channel'):
            simulator.apply_kraus_channel(qubit_name, [k0, k1])
        elif hasattr(simulator, 'apply_1qubit_gate'):
            r = self.rng.random()
            if r < p:
                simulator.Z(qubit_name)
            else:
                simulator.apply_1qubit_gate(qubit_name, k0)
