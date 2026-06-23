from enum import Enum
from dataclasses import dataclass

class CapabilityLevel(Enum):
    SUPPORTED = "supported"
    EMULATED = "emulated"
    UNSUPPORTED = "unsupported"

@dataclass
class UnsupportedOp:
    kind: str
    source_span: tuple[int, int] | None
    pretty_repr: str
    reason: str

class BackendCapabilities:
    def __init__(self, **kwargs):
        self.supports_quantum_gates = kwargs.get("supports_quantum_gates", CapabilityLevel.UNSUPPORTED)
        self.supports_measurements = kwargs.get("supports_measurements", CapabilityLevel.UNSUPPORTED)
        self.supports_classical_functions = kwargs.get("supports_classical_functions", CapabilityLevel.UNSUPPORTED)
        self.supports_recursion = kwargs.get("supports_recursion", CapabilityLevel.UNSUPPORTED)
        self.supports_arrays = kwargs.get("supports_arrays", CapabilityLevel.UNSUPPORTED)
        self.supports_maps = kwargs.get("supports_maps", CapabilityLevel.UNSUPPORTED)
        self.supports_structs = kwargs.get("supports_structs", CapabilityLevel.UNSUPPORTED)
        self.supports_exceptions = kwargs.get("supports_exceptions", CapabilityLevel.UNSUPPORTED)
        self.supports_imports = kwargs.get("supports_imports", CapabilityLevel.UNSUPPORTED)
        self.supports_loops = kwargs.get("supports_loops", CapabilityLevel.UNSUPPORTED)
        self.supports_field_access = kwargs.get("supports_field_access", CapabilityLevel.UNSUPPORTED)
        self.supports_index_access = kwargs.get("supports_index_access", CapabilityLevel.UNSUPPORTED)

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
            supports_quantum_gates=CapabilityLevel.SUPPORTED,
            supports_measurements=CapabilityLevel.SUPPORTED,
            supports_classical_functions=CapabilityLevel.SUPPORTED,
            supports_recursion=CapabilityLevel.SUPPORTED,
            supports_arrays=CapabilityLevel.SUPPORTED,
            supports_maps=CapabilityLevel.SUPPORTED,
            supports_structs=CapabilityLevel.SUPPORTED,
            supports_exceptions=CapabilityLevel.SUPPORTED,
            supports_imports=CapabilityLevel.SUPPORTED,
            supports_loops=CapabilityLevel.SUPPORTED,
            supports_field_access=CapabilityLevel.SUPPORTED,
            supports_index_access=CapabilityLevel.SUPPORTED,
        )
    elif backend_name in ("qiskit", "ibmq"):
        return BackendCapabilities(
            supports_quantum_gates=CapabilityLevel.SUPPORTED,
            supports_measurements=CapabilityLevel.SUPPORTED,
            supports_classical_functions=CapabilityLevel.UNSUPPORTED,
            supports_recursion=CapabilityLevel.UNSUPPORTED,
            supports_arrays=CapabilityLevel.EMULATED,
            supports_maps=CapabilityLevel.UNSUPPORTED,
            supports_structs=CapabilityLevel.EMULATED,
            supports_exceptions=CapabilityLevel.UNSUPPORTED,
            supports_imports=CapabilityLevel.EMULATED,
            supports_loops=CapabilityLevel.UNSUPPORTED,
            supports_field_access=CapabilityLevel.EMULATED,
            supports_index_access=CapabilityLevel.EMULATED,
        )
    else:
        return BackendCapabilities()
