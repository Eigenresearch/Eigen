# Transpiled from Eigen EQIR v1.1 Graph to Qiskit Script
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ==================================================
# WARNING: Unsupported classical constructs encountered:
# - StructDeclNode (pretty: 'struct Person'): Structs not supported
# - FuncDeclNode (pretty: 'func factorial'): Classical functions not supported
# - CallNode (pretty: 'call'): Classical calls not supported
# - CallNode (pretty: 'call'): Classical calls not supported
# - StructLiteralNode (pretty: 'Person {...}'): Structs not supported
# - DotAccessNode (pretty: '.age'): Field access not supported
# - DotAccessNode (pretty: '.age'): Field access not supported
# ==================================================

# Allocate circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Assert condition: (c0 == c1) == True
# Allocated qubit: q1
# Allocated qubit: q0
qc.h(0)
qc.cx(0, 1)
qc.measure(1, 1)
print('[PRINT DIRECTIVE] cbit c1:', c1 if 'c1' in locals() else 'c1')
qc.measure(0, 0)
print('[PRINT DIRECTIVE] cbit c0:', c0 if 'c0' in locals() else 'c0')
# Assert condition: (__unsupported_DotAccessNode__ == 31) == True
# Eigen warning: unsupported print expression omitted (__unsupported_DotAccessNode__)
# Assert condition: (__unsupported_CallNode__ == 120) == True
# Eigen warning: unsupported print expression omitted (__unsupported_CallNode__)

# Execute the circuit using Qiskit Aer
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
result = simulator.run(compiled_circuit, shots=1024).result()
counts = result.get_counts(qc)
print('Simulation results counts:', counts)