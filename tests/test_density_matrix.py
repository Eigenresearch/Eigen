import unittest
import math
import cmath
from src.simulator import QuantumSimulator

class TestDensityMatrix(unittest.TestCase):
    def test_gates_density_matrix(self):
        sim = QuantumSimulator(sim_type='density_matrix')
        sim.allocate_qubit('q0')
        sim.allocate_qubit('q1')
        
        sim.H('q0')
        sim.CNOT('q0', 'q1')
        
        # Bell state (|00> + |11>)/sqrt(2)
        # Density matrix should be 0.5|00><00| + 0.5|00><11| + 0.5|11><00| + 0.5|11><11|
        amps = sim.get_amplitudes_dict()
        self.assertAlmostEqual(amps['00'], 1.0 / math.sqrt(2))
        self.assertAlmostEqual(amps['11'], 1.0 / math.sqrt(2))

    def test_bit_flip_noise(self):
        # Apply X, then bit flip with 20% probability
        # Effective state: 80% |1>, 20% |0>
        # Diagonal of rho: rho[0,0] = 0.2, rho[1,1] = 0.8
        sim = QuantumSimulator(sim_type='density_matrix')
        sim.allocate_qubit('q0')
        sim.X('q0')
        
        from src.noise.noise_model import NoiseModel
        noise = NoiseModel(noise_type='bit_flip', noise_prob=0.2)
        noise.apply_gate_noise(sim, 'q0')
        
        # In get_amplitudes_dict(), amplitude is sqrt of diagonal element (probability)
        amps = sim.get_amplitudes_dict()
        self.assertAlmostEqual(amps['0'], math.sqrt(0.2))
        self.assertAlmostEqual(amps['1'], math.sqrt(0.8))

    def test_amplitude_damping_noise(self):
        # Prepare in state |1>, apply damping with p=0.3
        # State becomes: 30% |0>, 70% |1>
        sim = QuantumSimulator(sim_type='density_matrix')
        sim.allocate_qubit('q0')
        sim.X('q0')
        
        from src.noise.noise_model import NoiseModel
        noise = NoiseModel(noise_type='amplitude_damping', noise_prob=0.3)
        noise.apply_gate_noise(sim, 'q0')
        
        amps = sim.get_amplitudes_dict()
        self.assertAlmostEqual(amps['0'], math.sqrt(0.3))
        self.assertAlmostEqual(amps['1'], math.sqrt(0.7))

    def test_depolarizing_noise(self):
        # Prepare in state |1>, apply depolarizing with p=0.3
        # E0 = sqrt(0.7) * I, E1,E2,E3 = sqrt(0.1) * X,Y,Z
        # E0: |1> -> |1> with prob 0.7
        # X: |1> -> |0> with prob 0.1
        # Y: |1> -> -i|0> with prob 0.1
        # Z: |1> -> -|1> with prob 0.1
        # Overall probability of |0> = 0.1 + 0.1 = 0.2
        # Overall probability of |1> = 0.7 + 0.1 = 0.8
        sim = QuantumSimulator(sim_type='density_matrix')
        sim.allocate_qubit('q0')
        sim.X('q0')
        
        from src.noise.noise_model import NoiseModel
        noise = NoiseModel(noise_type='depolarizing', noise_prob=0.3)
        noise.apply_gate_noise(sim, 'q0')
        
        amps = sim.get_amplitudes_dict()
        self.assertAlmostEqual(amps['0'], math.sqrt(0.2))
        self.assertAlmostEqual(amps['1'], math.sqrt(0.8))

    def test_cli_density_matrix(self):
        # Verify running via CLI
        import os
        import subprocess
        
        test_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(test_dir)
        python_exe = os.path.join(workspace_root, ".venv", "Scripts", "python.exe")
        main_py = os.path.join(workspace_root, "src", "main.py")
        
        source_file = os.path.join(test_dir, "temp_dm_test.eig")
        with open(source_file, "w", encoding="utf-8") as f:
            f.write("""eigen 2.5
            qubit q0
            H q0
            cbit c0
            measure q0 -> c0
            """)
            
        try:
            result = subprocess.run([
                python_exe, main_py, "run", source_file, "--backend", "density_matrix"
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"CLI run failed: {result.stderr}")
        finally:
            if os.path.exists(source_file):
                os.remove(source_file)

if __name__ == '__main__':
    unittest.main()
