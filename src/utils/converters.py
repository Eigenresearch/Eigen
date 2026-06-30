# Qubit Indexing and State Vector Convention Converters

def to_msb_first_dict(amplitudes: dict[str, complex]) -> dict[str, complex]:
    """
    Converts amplitude dictionary keys from LSB-first bitstring (where the first-created
    qubit q0 corresponds to the rightmost bit: 'qN...q1q0') to MSB-first bitstring
    (where q0 corresponds to the leftmost bit: 'q0q1...qN').
    
    Example:
        {'01': 1.0} (q1=0, q0=1) -> {'10': 1.0}
    """
    converted = {}
    for bitstring, amp in amplitudes.items():
        converted[bitstring[::-1]] = amp
    return converted

def reorder_state_vector(
    state_vector: list[complex], 
    num_qubits: int, 
    source_convention: str = "lsb", 
    target_convention: str = "msb"
) -> list[complex]:
    """
    Reorders elements of a state vector between LSB-first (Little-Endian) and
    MSB-first (Big-Endian) qubit index conventions.
    
    In 'lsb' convention, qubit q0 corresponds to the least significant bit (bit 0).
    In 'msb' convention, qubit q0 corresponds to the most significant bit (bit N-1).
    """
    if source_convention == target_convention:
        return list(state_vector)
        
    n = num_qubits
    reordered = [0.0j] * len(state_vector)
    
    for idx, amp in enumerate(state_vector):
        # Reverse the bits of the index
        reversed_idx = 0
        for i in range(n):
            bit = (idx >> i) & 1
            if bit:
                reversed_idx |= (1 << (n - 1 - i))
        reordered[reversed_idx] = amp
        
    return reordered
