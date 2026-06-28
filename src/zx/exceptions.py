class IndeterminateEquivalenceError(Exception):
    """Raised when the equivalence checking cannot be verified conclusively (e.g. for N > 16 qubits)."""
    pass
