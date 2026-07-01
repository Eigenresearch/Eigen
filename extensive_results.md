# Extensive Benchmark: Native Python vs Eigen VM (Misery 2.6)

Comparison of execution time between native Python execution and Eigen VM (with JIT enabled) across 25 tests (including 11 accuracy/precision checks).

## Performance & Correctness Summary

| Benchmark Test Case | Native Python | Eigen VM | Accuracy / Fidelity |
| --- | --- | --- | --- |
| 01. Fibonacci Recursion | 0.383 ms | 50.462 ms | N/A |
| 02. Arithmetic Loop | 1.051 ms | 85.352 ms | N/A |
| 03. Array Operations | 0.534 ms | 46.927 ms | N/A |
| 04. Exception Try-Catch | 0.696 ms | 35.955 ms | N/A |
| 05. Quantum Teleportation | 0.127 ms | 0.302 ms | PASSED (Deterministic Output Correct) |
| 06. Hadamard Precision | 0.025 ms | 0.076 ms | PASSED (Fidelity: 1.000000) |
| 07. CNOT Entanglement | 0.041 ms | 0.092 ms | PASSED (Fidelity: 1.000000) |
| 08. Phase Rotation | 0.038 ms | 0.106 ms | PASSED (Fidelity: 1.000000) |
| 09. Toffoli Truth Table | 0.054 ms | 0.471 ms | PASSED (Fidelity: 1.000000) |
| 10. CSWAP Exchange | 0.050 ms | 0.109 ms | PASSED (Fidelity: 1.000000) |
| 11. Controlled-Phase CP | 1.333 ms | 0.108 ms | PASSED (Fidelity: 1.000000) |
| 12. QFT-3 Simulation | 0.102 ms | 0.257 ms | PASSED (Fidelity: 1.000000) |
| 13. Bernstein-Vazirani | 0.097 ms | 0.260 ms | PASSED (Deterministic Output Correct) |
| 14. Superposition Balance | 0.074 ms | 0.200 ms | PASSED (Fidelity: 1.000000) |
| 15. State Normalization | 0.032 ms | 0.118 ms | PASSED (Fidelity: 1.000000) |
| 16. Nested Loops | 0.537 ms | 66.575 ms | N/A |
| 17. Recursive Factorial | 0.001 ms | 0.123 ms | N/A |
| 18. Array Sorting | 0.008 ms | 1.148 ms | N/A |
| 19. Struct Operations | 0.176 ms | 11.274 ms | N/A |
| 20. Map Operations | 0.064 ms | 5.979 ms | N/A |
| 21. Bitwise Logic | 0.395 ms | 32.359 ms | N/A |
| 22. Function Call Overhead | 0.223 ms | 22.115 ms | N/A |
| 23. Large Circuit Loops | 4.665 ms | 18.227 ms | N/A |
| 24. Deep Nested If-Else | 0.502 ms | 43.163 ms | N/A |
| 25. Composite Assertion Checks | 0.166 ms | 21.440 ms | N/A |
