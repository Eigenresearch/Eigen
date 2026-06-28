import random

class NoiseModel:
    def __init__(self, noise_type: str = None, noise_prob: float = 0.0, rng=None):
        # noise_type: 'depolarizing', 'bit_flip', 'phase_flip', 'amplitude_damping', 'readout_error'
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        self.rng = rng if rng is not None else random.Random()

    def apply_gate_noise(self, simulator, qubit_name: str):
        if not self.noise_type or self.noise_prob <= 0.0:
            return
            
        if self.noise_type == 'readout_error':
            return  # Readout error is only applied at measurement time
            
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
                # Amplitude damping decays state to |0>.
                # We project the qubit to |0> using a measure-and-conditional-flip cycle.
                outcome = simulator.measure(qubit_name)
                if outcome == 1:
                    simulator.X(qubit_name)

    def apply_readout_noise(self, outcome: int) -> int:
        if self.noise_type == 'readout_error' and self.noise_prob > 0.0:
            r = self.rng.random()
            if r < self.noise_prob:
                return 1 - outcome
        return outcome
