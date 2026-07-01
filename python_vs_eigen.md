# Language Benchmark: Native Python vs Eigen VM (Misery 2.6)

Comparison of execution time between native Python execution and Eigen VM (with JIT compilation enabled) across 30 runs.

| Benchmark Test Case | Native Python (ms) | Eigen VM (ms) | Slowdown / Ratio |
| --- | --- | --- | --- |
| Arithmetic Loop | 0.717 ms | 86.193 ms | **120.22x** |
| Array Operations | 0.500 ms | 47.025 ms | **94.03x** |
| Exception Try-Catch | 0.701 ms | 35.690 ms | **50.95x** |
| Fibonacci Recursion | 0.377 ms | 50.012 ms | **132.50x** |
| Quantum Teleportation | 1.069 ms | 0.292 ms | **0.27x** |
