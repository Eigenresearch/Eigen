"""§3.2 — Pulse-level control: surface module for hardware
pulse specification and execution.

Pulse-level control allows users to define arbitrary microwave
pulses (drive amplitude, frequency, phase, duration) instead of
using the gate abstraction. This is required for:
  - Pulse-level calibration of quantum gates
  - Custom pulse shapes (e.g. DRAG, Gaussian, square)
  - Direct control of cross-resonance drives
"""
from __future__ import annotations

import dataclasses
import typing
import math


@dataclasses.dataclass
class PulseShape:
    """Base pulse shape specification."""
    name: str
    duration_ns: float
    amplitude: float  # normalized [0, 1]
    frequency_hz: float = 0.0
    phase_rad: float = 0.0

    def samples(self, sample_rate_ghz: float = 1.0) -> list[float]:
        """Generate discrete samples of this pulse."""
        n = int(self.duration_ns * sample_rate_ghz)
        return [self._sample_at(t / sample_rate_ghz, sample_rate_ghz)
                for t in range(n)]

    def _sample_at(self, t_ns: float, sr: float) -> float:
        return self.amplitude  # base = square pulse


@dataclasses.dataclass
class GaussianPulse(PulseShape):
    """Gaussian-enveloped pulse."""
    sigma_ns: float = 10.0

    def _sample_at(self, t_ns: float, sr: float) -> float:
        center = self.duration_ns / 2.0
        env = math.exp(-0.5 * ((t_ns - center) / self.sigma_ns) ** 2)
        return self.amplitude * env * math.cos(
            2 * math.pi * self.frequency_hz * t_ns * 1e-9
            + self.phase_rad)


@dataclasses.dataclass
class DRAGPulse(PulseShape):
    """DRAG (Derivative Removal by Adiabatic Gate) pulse."""
    sigma_ns: float = 10.0
    beta: float = 0.5

    def _sample_at(self, t_ns: float, sr: float) -> float:
        center = self.duration_ns / 2.0
        gauss = math.exp(-0.5 * ((t_ns - center) / self.sigma_ns) ** 2)
        deriv = -(t_ns - center) / (self.sigma_ns ** 2) * gauss
        i_part = self.amplitude * gauss
        q_part = self.amplitude * self.beta * deriv
        return i_part * math.cos(
            2 * math.pi * self.frequency_hz * t_ns * 1e-9
            + self.phase_rad) - q_part * math.sin(
            2 * math.pi * self.frequency_hz * t_ns * 1e-9
            + self.phase_rad)


@dataclasses.dataclass
class SquarePulse(PulseShape):
    """Square (constant-amplitude) pulse."""

    def _sample_at(self, t_ns: float, sr: float) -> float:
        return self.amplitude * math.cos(
            2 * math.pi * self.frequency_hz * t_ns * 1e-9
            + self.phase_rad)


@dataclasses.dataclass
class PulseInstruction:
    """A single pulse instruction on a specific channel."""
    channel: str
    pulse: PulseShape
    start_time_ns: float = 0.0


class PulseSchedule:
    """A schedule of pulse instructions for a quantum experiment."""

    def __init__(self):
        self.instructions: list[PulseInstruction] = []
        self.channels: set[str] = set()

    def add(self, channel: str, pulse: PulseShape,
            start_time_ns: float = 0.0):
        self.instructions.append(
            PulseInstruction(channel, pulse, start_time_ns))
        self.channels.add(channel)
        self.instructions.sort(key=lambda i: i.start_time_ns)

    @property
    def duration_ns(self) -> float:
        if not self.instructions:
            return 0.0
        return max(i.start_time_ns + i.pulse.duration_ns
                    for i in self.instructions)

    def to_dict(self) -> dict:
        return {
            "duration_ns": self.duration_ns,
            "channels": sorted(self.channels),
            "instructions": [
                {
                    "channel": i.channel,
                    "start_time_ns": i.start_time_ns,
                    "pulse": {
                        "name": i.pulse.name,
                        "duration_ns": i.pulse.duration_ns,
                        "amplitude": i.pulse.amplitude,
                        "frequency_hz": i.pulse.frequency_hz,
                        "phase_rad": i.pulse.phase_rad,
                    },
                }
                for i in self.instructions
            ],
        }


def gate_to_pulse(gate_name: str, qubit_freq_hz: float = 5e9
                   ) -> PulseShape | None:
    """Convert a gate name to a canonical pulse shape (surface)."""
    if gate_name == "X":
        return SquarePulse(name="X", duration_ns=20.0,
                             amplitude=1.0, frequency_hz=qubit_freq_hz)
    elif gate_name == "Y":
        return SquarePulse(name="Y", duration_ns=20.0,
                             amplitude=1.0, frequency_hz=qubit_freq_hz,
                             phase_rad=math.pi / 2)
    elif gate_name == "H":
        return GaussianPulse(name="H", duration_ns=40.0,
                               amplitude=0.5, frequency_hz=qubit_freq_hz,
                               sigma_ns=10.0)
    elif gate_name == "T":
        return SquarePulse(name="T", duration_ns=10.0,
                             amplitude=0.5, frequency_hz=qubit_freq_hz,
                             phase_rad=math.pi / 4)
    return None
