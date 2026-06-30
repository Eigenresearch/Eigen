import math
import cmath

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
