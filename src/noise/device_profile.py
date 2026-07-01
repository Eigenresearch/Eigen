"""Device noise profile for loading real hardware calibration data.

Supports importing noise parameters from IBM Quantum and IonQ backends.
When real calibration data is not available, uses typical published values.
"""
import json
import math
import random
from src.noise.noise_channel import (
    NoiseChannel,
    NoisePipeline,
    BitFlipChannel,
    PhaseFlipChannel,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
    ReadoutErrorChannel,
)
from src.noise.t1t2_model import T1T2NoiseModel
from src.noise.crosstalk_model import CrosstalkModel


IBM_DEVICE_PROFILES = {
    'ibm_sherbrooke': {
        'num_qubits': 127,
        't1_avg': 120.0,
        't2_avg': 80.0,
        'single_qubit_error': 0.0003,
        'two_qubit_error': 0.005,
        'readout_error': 0.01,
        'crosstalk_prob': 0.0008,
    },
    'ibm_brisbane': {
        'num_qubits': 127,
        't1_avg': 100.0,
        't2_avg': 60.0,
        'single_qubit_error': 0.0004,
        'two_qubit_error': 0.008,
        'readout_error': 0.015,
        'crosstalk_prob': 0.001,
    },
    'ibm_kyiv': {
        'num_qubits': 127,
        't1_avg': 110.0,
        't2_avg': 70.0,
        'single_qubit_error': 0.0003,
        'two_qubit_error': 0.006,
        'readout_error': 0.012,
        'crosstalk_prob': 0.0009,
    },
    'ibm_osaka': {
        'num_qubits': 127,
        't1_avg': 90.0,
        't2_avg': 50.0,
        'single_qubit_error': 0.0005,
        'two_qubit_error': 0.01,
        'readout_error': 0.02,
        'crosstalk_prob': 0.0012,
    },
}

IONQ_DEVICE_PROFILES = {
    'ionq_harmony': {
        'num_qubits': 11,
        't1_avg': 10000.0,
        't2_avg': 1000.0,
        'single_qubit_error': 0.0003,
        'two_qubit_error': 0.002,
        'readout_error': 0.005,
        'crosstalk_prob': 0.0001,
    },
    'ionq_aria': {
        'num_qubits': 25,
        't1_avg': 10000.0,
        't2_avg': 2000.0,
        'single_qubit_error': 0.0002,
        'two_qubit_error': 0.001,
        'readout_error': 0.003,
        'crosstalk_prob': 0.00005,
    },
    'ionq_forte': {
        'num_qubits': 36,
        't1_avg': 10000.0,
        't2_avg': 5000.0,
        'single_qubit_error': 0.0001,
        'two_qubit_error': 0.0005,
        'readout_error': 0.002,
        'crosstalk_prob': 0.00003,
    },
}


class DeviceNoiseProfile:
    """Noise profile loaded from real hardware calibration data.

    Combines T1/T2 relaxation, gate-specific depolarizing errors,
    readout errors, and crosstalk into a single NoisePipeline.

    Usage:
        profile = DeviceNoiseProfile.from_ibm('ibm_sherbrooke')
        pipeline = profile.build_pipeline()
        pipeline.apply_gate_noise(simulator, 'q0', gate_name='H')
    """

    def __init__(self, backend_name: str, num_qubits: int,
                 t1_avg: float, t2_avg: float,
                 single_qubit_error: float, two_qubit_error: float,
                 readout_error: float, crosstalk_prob: float,
                 gate_specific_rates: dict = None,
                 coupling_map=None, rng=None, seed=None):
        self.backend_name = backend_name
        self.num_qubits = num_qubits
        self.t1_avg = t1_avg
        self.t2_avg = t2_avg
        self.single_qubit_error = single_qubit_error
        self.two_qubit_error = two_qubit_error
        self.readout_error = readout_error
        self.crosstalk_prob = crosstalk_prob
        self.gate_specific_rates = gate_specific_rates or {}
        self.coupling_map = coupling_map
        self.rng = rng if rng is not None else random.Random(seed)

    @staticmethod
    def from_ibm(backend_name: str, coupling_map=None, rng=None, seed=None) -> 'DeviceNoiseProfile':
        """Load noise profile from IBM Quantum backend name.

        Args:
            backend_name: IBM backend name (e.g. 'ibm_sherbrooke').
            coupling_map: Optional hardware coupling map.
            rng: Random number generator.

        Returns:
            DeviceNoiseProfile configured for the specified backend.

        Raises:
            ValueError: If the backend name is not recognized.
        """
        name = backend_name.lower()
        if name not in IBM_DEVICE_PROFILES:
            known = ', '.join(sorted(IBM_DEVICE_PROFILES.keys()))
            raise ValueError(
                f"Unknown IBM backend '{backend_name}'. "
                f"Known backends: {known}"
            )
        params = IBM_DEVICE_PROFILES[name]
        return DeviceNoiseProfile(
            backend_name=backend_name,
            coupling_map=coupling_map,
            rng=rng,
            seed=seed,
            **params,
        )

    @staticmethod
    def from_ionq(backend_name: str, coupling_map=None, rng=None, seed=None) -> 'DeviceNoiseProfile':
        """Load noise profile from IonQ backend name.

        Args:
            backend_name: IonQ backend name (e.g. 'ionq_harmony').

        Returns:
            DeviceNoiseProfile configured for the specified backend.
        """
        name = backend_name.lower()
        if name not in IONQ_DEVICE_PROFILES:
            known = ', '.join(sorted(IONQ_DEVICE_PROFILES.keys()))
            raise ValueError(
                f"Unknown IonQ backend '{backend_name}'. "
                f"Known backends: {known}"
            )
        params = IONQ_DEVICE_PROFILES[name]
        return DeviceNoiseProfile(
            backend_name=backend_name,
            coupling_map=coupling_map,
            rng=rng,
            seed=seed,
            **params,
        )

    @staticmethod
    def from_json(filepath: str, coupling_map=None, rng=None, seed=None) -> 'DeviceNoiseProfile':
        """Load noise profile from a JSON calibration file.

        The JSON file should have the following structure:
        {
            "backend_name": "ibm_sherbrooke",
            "num_qubits": 127,
            "t1_avg": 120.0,
            "t2_avg": 80.0,
            "single_qubit_error": 0.0003,
            "two_qubit_error": 0.005,
            "readout_error": 0.01,
            "crosstalk_prob": 0.0008,
            "gate_specific_rates": {"CNOT_0_1": 0.004}
        }
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return DeviceNoiseProfile(
            coupling_map=coupling_map,
            rng=rng,
            seed=seed,
            **data,
        )

    def build_pipeline(self) -> NoisePipeline:
        """Build a NoisePipeline from this device profile.

        The pipeline includes:
        - T1/T2 relaxation (amplitude + phase damping)
        - Gate-specific depolarizing errors
        - Readout errors
        - Crosstalk
        """
        pipeline = NoisePipeline(rng=self.rng)

        t1t2 = T1T2NoiseModel(
            t1=self.t1_avg,
            t2=self.t2_avg,
            rng=self.rng,
        )
        pipeline.add_channel(_T1T2ChannelAdapter(t1t2))

        depol_1q = DepolarizingChannel(self.single_qubit_error, rng=self.rng)
        pipeline.add_channel(depol_1q)

        readout = ReadoutErrorChannel(self.readout_error, rng=self.rng)
        pipeline.add_channel(readout)

        crosstalk = CrosstalkModel(
            crosstalk_prob=self.crosstalk_prob,
            coupling_map=self.coupling_map,
            rng=self.rng,
        )
        pipeline.add_channel(_CrosstalkChannelAdapter(crosstalk))

        return pipeline

    def summary(self) -> dict:
        """Return a summary dict of the noise profile."""
        return {
            'backend_name': self.backend_name,
            'num_qubits': self.num_qubits,
            't1_avg_us': self.t1_avg,
            't2_avg_us': self.t2_avg,
            'single_qubit_error': self.single_qubit_error,
            'two_qubit_error': self.two_qubit_error,
            'readout_error': self.readout_error,
            'crosstalk_prob': self.crosstalk_prob,
        }


class _T1T2ChannelAdapter(NoiseChannel):
    """Adapter to make T1T2NoiseModel compatible with NoiseChannel interface."""

    def __init__(self, t1t2_model: T1T2NoiseModel):
        self._model = t1t2_model
        super().__init__(rng=t1t2_model.rng)

    @property
    def name(self) -> str:
        return "t1t2_relaxation"

    def apply_to_qubit(self, simulator, qubit_name: str, **kwargs):
        gate_name = kwargs.get('gate_name', 'I')
        self._model.apply_gate_noise(simulator, qubit_name, gate_name=gate_name)

    def apply_to_pair(self, simulator, q1: str, q2: str, **kwargs):
        gate_name = kwargs.get('gate_name', 'CNOT')
        self._model.apply_two_qubit_noise(simulator, q1, q2, gate_name=gate_name)


class _CrosstalkChannelAdapter(NoiseChannel):
    """Adapter to make CrosstalkModel compatible with NoiseChannel interface."""

    def __init__(self, crosstalk_model: CrosstalkModel):
        self._model = crosstalk_model
        super().__init__(rng=crosstalk_model.rng)

    @property
    def name(self) -> str:
        return "crosstalk"

    def apply_to_qubit(self, simulator, qubit_name: str, **kwargs):
        self._model.apply_gate_noise(simulator, qubit_name, **kwargs)

    def apply_to_pair(self, simulator, q1: str, q2: str, **kwargs):
        self._model.apply_two_qubit_noise(simulator, q1, q2, **kwargs)
