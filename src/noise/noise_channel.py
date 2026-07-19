"""Abstract noise channel and composition pipeline.

Provides a composable noise framework where multiple NoiseChannel
instances can be chained together to model complex noise processes.
"""
import math
import random
from abc import ABC, abstractmethod


class NoiseChannel(ABC):
    """Abstract base class for a quantum noise channel.

    A NoiseChannel represents a single noise process (e.g. bit-flip,
    amplitude damping, T1 relaxation). Multiple channels can be
    composed into a NoisePipeline to model realistic composite noise.
    """

    def __init__(self, rng=None, seed=None):
        self.rng = rng if rng is not None else random.Random(seed)

    @abstractmethod
    def apply_to_qubit(self, simulator, qubit_name: str, **kwargs):
        """Apply this noise channel to a single qubit.

        Args:
            simulator: The quantum simulator instance.
            qubit_name: Name of the qubit to apply noise to.
            **kwargs: Additional parameters (e.g. gate_time, gate_type).
        """
        pass

    @abstractmethod
    def apply_to_pair(self, simulator, q1: str, q2: str, **kwargs):
        """Apply this noise channel to a qubit pair (for correlated noise).

        Args:
            simulator: The quantum simulator instance.
            q1: Name of the first qubit.
            q2: Name of the second qubit.
            **kwargs: Additional parameters.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this noise channel."""
        pass


class BitFlipChannel(NoiseChannel):
    """Stochastic X-error bit-flip noise channel."""

    def __init__(self, prob: float, rng=None, seed=None):
        super().__init__(rng, seed)
        self.prob = prob

    @property
    def name(self) -> str:
        return "bit_flip"

    def apply_to_qubit(self, simulator, qubit_name: str, **kwargs):
        if self.prob <= 0.0:
            return
        if self.rng.random() < self.prob:
            simulator.X(qubit_name)

    def apply_to_pair(self, simulator, q1: str, q2: str, **kwargs):
        self.apply_to_qubit(simulator, q1)
        self.apply_to_qubit(simulator, q2)


class PhaseFlipChannel(NoiseChannel):
    """Stochastic Z-error phase-flip noise channel."""

    def __init__(self, prob: float, rng=None, seed=None):
        super().__init__(rng, seed)
        self.prob = prob

    @property
    def name(self) -> str:
        return "phase_flip"

    def apply_to_qubit(self, simulator, qubit_name: str, **kwargs):
        if self.prob <= 0.0:
            return
        if self.rng.random() < self.prob:
            simulator.Z(qubit_name)

    def apply_to_pair(self, simulator, q1: str, q2: str, **kwargs):
        self.apply_to_qubit(simulator, q1)
        self.apply_to_qubit(simulator, q2)


class DepolarizingChannel(NoiseChannel):
    """Depolarizing noise channel: random X, Y, or Z error."""

    def __init__(self, prob: float, rng=None, seed=None):
        super().__init__(rng, seed)
        self.prob = prob

    @property
    def name(self) -> str:
        return "depolarizing"

    def apply_to_qubit(self, simulator, qubit_name: str, **kwargs):
        if self.prob <= 0.0:
            return
        if self.rng.random() < self.prob:
            ch = self.rng.choice(['X', 'Y', 'Z'])
            getattr(simulator, ch)(qubit_name)

    def apply_to_pair(self, simulator, q1: str, q2: str, **kwargs):
        self.apply_to_qubit(simulator, q1)
        self.apply_to_qubit(simulator, q2)


class AmplitudeDampingChannel(NoiseChannel):
    """Amplitude damping (T1 relaxation) noise channel using Kraus operators."""

    def __init__(self, gamma: float, rng=None, seed=None):
        super().__init__(rng, seed)
        self.gamma = gamma

    @property
    def name(self) -> str:
        return "amplitude_damping"

    def apply_to_qubit(self, simulator, qubit_name: str, **kwargs):
        if self.gamma <= 0.0:
            return
        k0 = [[1.0, 0.0], [0.0, math.sqrt(1.0 - self.gamma)]]
        k1 = [[0.0, math.sqrt(self.gamma)], [0.0, 0.0]]
        if hasattr(simulator, 'apply_kraus_channel'):
            simulator.apply_kraus_channel(qubit_name, [k0, k1])
        elif hasattr(simulator, 'apply_1qubit_gate'):
            r = self.rng.random()
            if r < self.gamma:
                simulator.apply_1qubit_gate(qubit_name, k1)
            else:
                simulator.apply_1qubit_gate(qubit_name, k0)

    def apply_to_pair(self, simulator, q1: str, q2: str, **kwargs):
        self.apply_to_qubit(simulator, q1)
        self.apply_to_qubit(simulator, q2)


class PhaseDampingChannel(NoiseChannel):
    """Phase damping (T2 dephasing) noise channel using Kraus operators."""

    def __init__(self, lambda_val: float, rng=None, seed=None):
        super().__init__(rng, seed)
        self.lambda_val = lambda_val

    @property
    def name(self) -> str:
        return "phase_damping"

    def apply_to_qubit(self, simulator, qubit_name: str, **kwargs):
        if self.lambda_val <= 0.0:
            return
        k0 = [[1.0, 0.0], [0.0, math.sqrt(1.0 - self.lambda_val)]]
        k1 = [[0.0, 0.0], [0.0, math.sqrt(self.lambda_val)]]
        if hasattr(simulator, 'apply_kraus_channel'):
            simulator.apply_kraus_channel(qubit_name, [k0, k1])
        elif hasattr(simulator, 'apply_1qubit_gate'):
            r = self.rng.random()
            if r < self.lambda_val:
                simulator.apply_1qubit_gate(qubit_name, k1)
            else:
                simulator.apply_1qubit_gate(qubit_name, k0)

    def apply_to_pair(self, simulator, q1: str, q2: str, **kwargs):
        self.apply_to_qubit(simulator, q1)
        self.apply_to_qubit(simulator, q2)


class ReadoutErrorChannel(NoiseChannel):
    """Readout (measurement) error channel — symmetric bit-flip at measurement."""

    def __init__(self, prob: float, rng=None, seed=None):
        super().__init__(rng, seed)
        self.prob = prob

    @property
    def name(self) -> str:
        return "readout_error"

    def apply_to_qubit(self, simulator, qubit_name: str, **kwargs):
        pass

    def apply_to_pair(self, simulator, q1: str, q2: str, **kwargs):
        pass

    def apply_readout_noise(self, outcome: int) -> int:
        if self.prob > 0.0 and self.rng.random() < self.prob:
            return 1 - outcome
        return outcome


class ReadoutError:
    """Confusion-matrix-based readout error channel.

    The confusion matrix is indexed as ``matrix[true_outcome]`` and yields
    ``[p(0|true), p(1|true)]``. For a single qubit it is therefore a 2x2
    stochastic matrix::

        [[p(0|0), p(1|0)],
         [p(0|1), p(1|1)]]

    Each row must sum to 1. ``apply`` samples the observed outcome given a
    noiseless ``outcome``.
    """

    def __init__(self, confusion_matrix, rng=None, seed=None):
        matrix = [list(row) for row in confusion_matrix]
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError(
                "confusion_matrix must be 2x2 for a single-qubit readout")
        for row in matrix:
            total = sum(row)
            if total < 0.0:
                raise ValueError("confusion_matrix rows must be non-negative")
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"confusion_matrix rows must be stochastic (sum to 1), "
                    f"got row sum {total}")
        self.matrix = matrix
        self.rng = rng if rng is not None else random.Random(seed)

    def apply(self, outcome: int, rng=None) -> int:
        r = rng if rng is not None else self.rng
        return r.choices([0, 1], weights=self.matrix[outcome])[0]


class NoisePipeline:
    """Composition pipeline for multiple noise channels.

    Channels are applied in order. This allows modelling composite
    noise (e.g. T1 relaxation + crosstalk + readout error).
    """

    def __init__(self, channels: list[NoiseChannel] = None, rng=None, seed=None):
        self.channels = channels if channels is not None else []
        self.rng = rng if rng is not None else random.Random(seed)

    def add_channel(self, channel: NoiseChannel):
        self.channels.append(channel)
        return self

    def apply_gate_noise(self, simulator, qubit_name: str, **kwargs):
        for channel in self.channels:
            channel.apply_to_qubit(simulator, qubit_name, **kwargs)

    def apply_two_qubit_noise(self, simulator, q1: str, q2: str, **kwargs):
        for channel in self.channels:
            channel.apply_to_pair(simulator, q1, q2, **kwargs)

    def apply_readout_noise(self, outcome: int) -> int:
        result = outcome
        for channel in self.channels:
            if isinstance(channel, ReadoutErrorChannel):
                result = channel.apply_readout_noise(result)
        return result

    @property
    def channel_names(self) -> list[str]:
        return [ch.name for ch in self.channels]
