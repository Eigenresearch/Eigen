from enum import Enum
from dataclasses import dataclass

class CapabilityLevel(Enum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"

@dataclass
class UnsupportedOp:
    kind: str
    source_span: tuple[int, int] | None
    pretty_repr: str
    reason: str

class BackendCapabilities:
    def __init__(self, **kwargs):
        self.supports_quantum_gates = kwargs.get("supports_quantum_gates", CapabilityLevel.NONE)
        self.supports_measurements = kwargs.get("supports_measurements", CapabilityLevel.NONE)
        self.supports_classical_functions = kwargs.get("supports_classical_functions", CapabilityLevel.NONE)
        self.supports_recursion = kwargs.get("supports_recursion", CapabilityLevel.NONE)
        self.supports_arrays = kwargs.get("supports_arrays", CapabilityLevel.NONE)
        self.supports_maps = kwargs.get("supports_maps", CapabilityLevel.NONE)
        self.supports_structs = kwargs.get("supports_structs", CapabilityLevel.NONE)
        self.supports_exceptions = kwargs.get("supports_exceptions", CapabilityLevel.NONE)
        self.supports_imports = kwargs.get("supports_imports", CapabilityLevel.NONE)
        self.supports_loops = kwargs.get("supports_loops", CapabilityLevel.NONE)
        self.supports_field_access = kwargs.get("supports_field_access", CapabilityLevel.NONE)
        self.supports_index_access = kwargs.get("supports_index_access", CapabilityLevel.NONE)

    def to_dict(self) -> dict:
        return {
            "supports_quantum_gates": self.supports_quantum_gates.value,
            "supports_measurements": self.supports_measurements.value,
            "supports_classical_functions": self.supports_classical_functions.value,
            "supports_recursion": self.supports_recursion.value,
            "supports_arrays": self.supports_arrays.value,
            "supports_maps": self.supports_maps.value,
            "supports_structs": self.supports_structs.value,
            "supports_exceptions": self.supports_exceptions.value,
            "supports_imports": self.supports_imports.value,
            "supports_loops": self.supports_loops.value,
            "supports_field_access": self.supports_field_access.value,
            "supports_index_access": self.supports_index_access.value,
        }

def get_backend_capabilities(backend_name: str) -> BackendCapabilities:
    if backend_name == "runtime":
        return BackendCapabilities(
            supports_quantum_gates=CapabilityLevel.FULL,
            supports_measurements=CapabilityLevel.FULL,
            supports_classical_functions=CapabilityLevel.FULL,
            supports_recursion=CapabilityLevel.FULL,
            supports_arrays=CapabilityLevel.FULL,
            supports_maps=CapabilityLevel.FULL,
            supports_structs=CapabilityLevel.FULL,
            supports_exceptions=CapabilityLevel.FULL,
            supports_imports=CapabilityLevel.FULL,
            supports_loops=CapabilityLevel.FULL,
            supports_field_access=CapabilityLevel.FULL,
            supports_index_access=CapabilityLevel.FULL,
        )
    elif backend_name in ("qiskit", "ibmq"):
        return BackendCapabilities(
            supports_quantum_gates=CapabilityLevel.FULL,
            supports_measurements=CapabilityLevel.FULL,
            supports_classical_functions=CapabilityLevel.NONE,
            supports_recursion=CapabilityLevel.NONE,
            supports_arrays=CapabilityLevel.PARTIAL,
            supports_maps=CapabilityLevel.NONE,
            supports_structs=CapabilityLevel.NONE,
            supports_exceptions=CapabilityLevel.NONE,
            supports_imports=CapabilityLevel.PARTIAL,
            supports_loops=CapabilityLevel.NONE,
            supports_field_access=CapabilityLevel.NONE,
            supports_index_access=CapabilityLevel.NONE,
        )
    else:
        return BackendCapabilities()
