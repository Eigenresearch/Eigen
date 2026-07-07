import math
import cmath
from dataclasses import dataclass
from typing import Optional as _Opt

SINGLE_QUBIT_GATES = {"H", "X", "Y", "Z", "S", "T"}
TWO_QUBIT_GATES = {"CNOT", "CZ", "SWAP"}
THREE_QUBIT_GATES = {"CCX", "CSWAP"}
ROTATION_GATES = {"RX", "RY", "RZ"}
CONTROLLED_ROTATION_GATES = {"CP", "CRX", "CRY", "CRZ"}
ANGLE_GATES = ROTATION_GATES | CONTROLLED_ROTATION_GATES
ALL_GATES = SINGLE_QUBIT_GATES | TWO_QUBIT_GATES | THREE_QUBIT_GATES | ANGLE_GATES

GATE_QUBIT_COUNT = {
    "H": 1, "X": 1, "Y": 1, "Z": 1, "S": 1, "T": 1,
    "RX": 1, "RY": 1, "RZ": 1,
    "CNOT": 2, "CZ": 2, "SWAP": 2,
    "CP": 2, "CRX": 2, "CRY": 2, "CRZ": 2,
    "CCX": 3, "CSWAP": 3,
}

GATE_TAKS_ANGLE = {
    "RX": True, "RY": True, "RZ": True,
    "CP": True, "CRX": True, "CRY": True, "CRZ": True,
}

CLIFFORD_GATES = {"H", "X", "Y", "Z", "S", "CNOT", "CZ", "SWAP"}

_GATE_MATRIX_CACHE = {}

def get_gate_matrix(gate_name: str, theta: float = None) -> list[list[complex]]:
    if gate_name in _GATE_MATRIX_CACHE and theta is None:
        return _GATE_MATRIX_CACHE[gate_name]
    inv_sqrt2 = 0.7071067811865475
    matrices = {
        "H": [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]],
        "X": [[0.0, 1.0], [1.0, 0.0]],
        "Y": [[0.0, -1j], [1j, 0.0]],
        "Z": [[1.0, 0.0], [0.0, -1.0]],
        "S": [[1.0, 0.0], [0.0, 1j]],
        "T": [[1.0, 0.0], [0.0, inv_sqrt2 + inv_sqrt2 * 1j]],
        "CNOT": [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
        "CZ": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]],
        "SWAP": [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],
    }
    if gate_name in matrices:
        if theta is None:
            _GATE_MATRIX_CACHE[gate_name] = matrices[gate_name]
        return matrices[gate_name]
    if gate_name == "RX":
        c, s = math.cos(theta/2), math.sin(theta/2)
        return [[c, -1j*s], [-1j*s, c]]
    if gate_name == "RY":
        c, s = math.cos(theta/2), math.sin(theta/2)
        return [[c, -s], [s, c]]
    if gate_name == "RZ":
        e0 = cmath.exp(-1j*theta/2)
        e1 = cmath.exp(1j*theta/2)
        return [[e0, 0], [0, e1]]
    if gate_name == "CP":
        return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,cmath.exp(1j*theta)]]
    if gate_name == "CRX":
        c, s = math.cos(theta/2), math.sin(theta/2)
        return [[1,0,0,0],[0,1,0,0],[0,0,c,-1j*s],[0,0,-1j*s,c]]
    if gate_name == "CRY":
        c, s = math.cos(theta/2), math.sin(theta/2)
        return [[1,0,0,0],[0,1,0,0],[0,0,c,-s],[0,0,s,c]]
    if gate_name == "CRZ":
        e0 = cmath.exp(-1j*theta/2)
        e1 = cmath.exp(1j*theta/2)
        return [[1,0,0,0],[0,1,0,0],[0,0,e0,0],[0,0,0,e1]]
    if gate_name == "CCX":
        return [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]]
    if gate_name == "CSWAP":
        return [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1]]
    return None


# === Audit §5: Shared gate registry for the four exporter backends ======
#
# Previously each of the four hardware-backend exporter modules
# (`ibm_backend.py`, `ionq_backend.py`, `azure_backend.py`,
# `braket_backend.py`) carried its own large `if/elif` chain dispatching by
# gate name. Adding a new gate required touching all four files, and the
# audit specifically called out two cases of this duplication causing bugs:
#  - Azure's exporter had not been updated for SWAP, silently dropping it.
#  - The IBM exporter was pinned to OpenQASM 2.0 while Qiskit has moved
#    to 3.0 as its primary interchange format.
#
# The unification here places *one* canonical, per-exporter encoding table in
# this module. `get_gate_spec(gate_name)` returns a `GateSpec`
# with metadata for the four exporters. Each exporter's `export()` becomes
# a uniform ~10-line dispatch loop, reducing the chance of new gates being
# added in some exporters but forgotten in others.

@dataclass(frozen=True)
class GateSpec:
    """Per-exporter encoding metadata for a single Eigen gate."""
    qubit_count: int
    takes_angle: bool
    is_clifford: bool
    # OpenQASM 2.0/3.0 stdgates name (e.g. "h", "rz", "cx", "swap"). Controlled
    # rotations in 3.0 use the `ctrl @ <base>(angle)` form, which is also a
    # valid OpenQASM 2.0 qelib1.inc gate name for the legacy `crx/cry/crz/cp`.
    qasm_name: str
    # OpenQASM 3.0-only spelling if it differs from `qasm_name`; otherwise
    # `None`. Used by exporters that prefer the modern 3.0 form (`ctrl @ ...`).
    qasm3_rendering: _Opt[str] = None
    # Azure QIR callee function (without the
    # `__quantum__qis__` prefix or `__body` suffix - the exporter builds the
    # full `__quantum__qis__<func>__body` identifier).
    qir_func: str = ""
    # Braket `Circuit` method name (e.g. "h", "cnot", "rz").
    braket_method: str = ""
    # IonQ JSON "gate" field value (e.g. "h", "cnot", "swap").
    ionq_gate: str = ""

    def qir_callee(self) -> str:
        """Full QIR function identifier for the Azure exporter."""
        return f"__quantum__qis__{self.qir_func}__body"


_GATE_SPEC_TABLE: dict[str, GateSpec] = {
    "H":     GateSpec(1, False, True,  "h",    None,            "h",            "h",    "h"),
    "X":     GateSpec(1, False, True,  "x",    None,            "x",            "x",    "x"),
    "Y":     GateSpec(1, False, True,  "y",    None,            "y",            "y",    "y"),
    "Z":     GateSpec(1, False, True,  "z",    None,            "z",            "z",    "z"),
    "S":     GateSpec(1, False, True,  "s",    None,            "s",            "s",    "s"),
    "T":     GateSpec(1, False, False, "t",    None,            "t",            "t",    "t"),
    "RX":    GateSpec(1, True,  False, "rx",   None,            "rx",           "rx",   "rx"),
    "RY":    GateSpec(1, True,  False, "ry",   None,            "ry",           "ry",   "ry"),
    "RZ":    GateSpec(1, True,  False, "rz",   None,            "rz",           "rz",   "rz"),
    "CNOT":  GateSpec(2, False, True,  "cx",   None,            "cnot",         "cnot", "cnot"),
    "CZ":    GateSpec(2, False, True,  "cz",   None,            "cz",           "cz",   "cz"),
    "SWAP":  GateSpec(2, False, True,  "swap", None,            "swap",         "swap", "swap"),
    "CCX":   GateSpec(3, False, False, "ccx",  None,            "ccx",          "ccx",  "ccx"),
    "CSWAP": GateSpec(3, False, False, "cswap",None,            "cswap",        "cswap", "cswap"),
    "CP":    GateSpec(2, True,  False, "cp",   "ctrl @ phase",  "cp",           "phase", "cp"),
    "CRX":   GateSpec(2, True,  False, "crx",  "ctrl @ rx",     "crx",          "rx",   "crx"),
    "CRY":   GateSpec(2, True,  False, "cry",  "ctrl @ ry",     "cry",          "ry",   "cry"),
    "CRZ":   GateSpec(2, True,  False, "crz",  "ctrl @ rz",     "crz",          "rz",   "crz"),
}


def get_gate_spec(gate_name: str) -> _Opt[GateSpec]:
    """Return the GateSpec for `gate_name` (case-insensitive) or None."""
    if not gate_name:
        return None
    return _GATE_SPEC_TABLE.get(gate_name.upper())


def all_registered_gates() -> list[str]:
    """Return the list of all gate names registered with the shared
    registry, sorted for deterministic iteration. Used by tests."""
    return sorted(_GATE_SPEC_TABLE.keys())
