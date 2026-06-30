import time
import math
import cmath
from src.simulator import PythonDenseStatevector

def run_qubit_test():
    print("--- Running Qubit Simulation Test (GHZ-5 State) ---")
    start = time.perf_counter()
    
    dense = PythonDenseStatevector()
    for _ in range(5):
        dense.allocate_qubit()
        
    dense.H(0)
    for i in range(4):
        dense.CNOT(i, i+1)
    
    for i in range(5):
        dense.RX(i, 0.25 * math.pi)
        
    dense_state = dense.get_state_vector()
    dur = time.perf_counter() - start
    print(f"Dense simulation completed in {dur*1000:.3f} ms.")
    return dense_state, dur

def run_accuracy_test():
    print("--- Running Simulation Accuracy Test ---")
    start = time.perf_counter()
    
    dense_rot = PythonDenseStatevector()
    dense_rot.allocate_qubit()
    
    theta = math.pi / 10.0
    for _ in range(10):
        dense_rot.RX(0, theta)
        
    res_state = dense_rot.get_state_vector()
    target = [0.0j, -1j]
    
    fidelity = abs(sum(c1.conjugate() * c2 for c1, c2 in zip(res_state, target)))
    dur = time.perf_counter() - start
    print(f"Accuracy check completed in {dur*1000:.3f} ms. Fidelity = {fidelity:.10f}")
    return fidelity, dur

if __name__ == "__main__":
    state, d_dur = run_qubit_test()
    fid, a_dur = run_accuracy_test()
    
    with open("tests_run_results.md", "w", encoding="utf-8") as f:
        f.write("# Verification Benchmark Results\n\n")
        f.write("## Test 1: 5-Qubit GHZ + Rotation Statevector\n")
        f.write(f"- **Duration:** {d_dur*1000:.3f} ms\n")
        f.write("- **Statevector Snippet (First 4 elements):**\n")
        f.write("```python\n")
        for i in range(min(4, len(state))):
            f.write(f"  |{i:05b}>: {state[i]:.6f}\n")
        f.write("```\n\n")
        f.write("## Test 2: Multi-rotation Gate Accuracy (Fidelity check)\n")
        f.write("- **Goal:** 10 consecutive `RX(pi/10)` rotations vs theoretical target `-i|1>`\n")
        f.write(f"- **Fidelity:** {fid:.10f} (Closer to 1.0000000000 means higher precision)\n")
        f.write(f"- **Fidelity Verification:** {'PASSED' if math.isclose(fid, 1.0, abs_tol=1e-9) else 'FAILED'}\n")
        f.write(f"- **Duration:** {a_dur*1000:.3f} ms\n")
