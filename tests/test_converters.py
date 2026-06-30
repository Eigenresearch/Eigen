import unittest
from src.utils.converters import to_msb_first_dict, reorder_state_vector

class TestConverters(unittest.TestCase):
    def test_to_msb_first_dict(self):
        # LSB format: q1 q0
        # '01' (q1=0, q0=1) -> '10' in MSB (q0=1, q1=0)
        amplitudes = {'01': 1.0, '10': 0.5j}
        converted = to_msb_first_dict(amplitudes)
        self.assertEqual(converted, {'10': 1.0, '01': 0.5j})

    def test_reorder_state_vector(self):
        # 2 qubits
        # LSB-first index mapping:
        # 0 (00) -> 0 (00)
        # 1 (01) -> 2 (10)
        # 2 (10) -> 1 (01)
        # 3 (11) -> 3 (11)
        state_vec = [1.0, 2.0, 3.0, 4.0]
        reordered = reorder_state_vector(state_vec, 2, source_convention="lsb", target_convention="msb")
        self.assertEqual(reordered, [1.0, 3.0, 2.0, 4.0])

if __name__ == "__main__":
    unittest.main()
