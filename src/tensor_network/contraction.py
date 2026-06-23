# Contraction helper functions for tensor network simulations
import numpy as np
from src.tensor_network.tensor import Tensor

def contract_network(tensors: list[Tensor]) -> Tensor:
    if not tensors:
        return None
    curr = tensors[0]
    for other in tensors[1:]:
        # Find shared labels
        shared = list(set(curr.labels).intersection(set(other.labels)))
        if shared:
            # Contract over first shared label
            curr = curr.contract(other, shared[0])
        else:
            # Outer product
            new_data = np.kron(curr.data, other.data)
            new_labels = curr.labels + other.labels
            curr = Tensor(new_data, new_labels)
    return curr
