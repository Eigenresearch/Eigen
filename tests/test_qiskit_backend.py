import unittest
from src.ir.ir_graph import EQIRGraph
from src.backend.qiskit_backend import QiskitBackend

class TestQiskitBackend(unittest.TestCase):
    def test_transpile_bell_state(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('ALLOC', targets=['q1'])
        graph.add_operation('GATE', gate_name='H', targets=['q0'])
        graph.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q1'])
        graph.add_operation('MEASURE', targets=['q0'], cbit_name='c0')
        graph.add_operation('MEASURE', targets=['q1'], cbit_name='c1')

        backend = QiskitBackend()
        qiskit_script, report = backend.transpile(graph)
        
        self.assertIn("QuantumCircuit(2, 2)", qiskit_script)
        self.assertIn("qc.h(0)", qiskit_script)
        self.assertIn("qc.cx(0, 1)", qiskit_script)
        self.assertIn("qc.measure(0, 0)", qiskit_script)
        self.assertIn("qc.measure(1, 1)", qiskit_script)
        self.assertIn("AerSimulator()", qiskit_script)

if __name__ == "__main__":
    unittest.main()
