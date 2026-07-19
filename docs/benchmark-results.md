# Language Benchmark: Native Python vs Eigen VM (Meridian 2.7)

Comparison of execution time between native Python execution and Eigen VM (with JIT compilation enabled) across 30 runs.

| Benchmark Test Case | Native Python (ms) | Eigen VM (ms) | Slowdown / Ratio |
| --- | --- | --- | --- |
| Arithmetic Loop | 0.826 ms | 0.951 ms | **1.15x** |
| Array Operations | 0.595 ms | 72.063 ms | **121.08x** |
| Exception Try-Catch | 0.770 ms | 64.145 ms | **83.31x** |
| Fibonacci Recursion | 0.688 ms | 68.161 ms | **99.13x** |
| Quantum Teleportation | 0.187 ms | 0.358 ms | **1.91x** |
