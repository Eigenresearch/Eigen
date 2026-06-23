# Tensor helper classes for MPS and other tensor network representations
import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray, labels: list[str]):
        self.data = data
        self.labels = labels

    def contract(self, other: 'Tensor', label: str) -> 'Tensor':
        # Contract two tensors over a shared label
        idx1 = self.labels.index(label)
        idx2 = other.labels.index(label)
        new_data = np.tensordot(self.data, other.data, axes=(idx1, idx2))
        new_labels = [l for i, l in enumerate(self.labels) if i != idx1] + \
                     [l for i, l in enumerate(other.labels) if i != idx2]
        return Tensor(new_data, new_labels)
