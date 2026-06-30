# Eigen 2.6 «Nova» Execution Benchmarks

Comparison of execution time between Pure Python VM execution (JIT Disabled) and JIT-compiled VM execution (JIT Enabled) across 50 runs.

| Benchmark Test Case | Pure Python VM (ms) | JIT-Enabled VM (ms) | Speedup Factor |
| --- | --- | --- | --- |
| Fibonacci Recursion | 49.930 ms | 50.532 ms | **0.99x** |
| Arithmetic Loop | 86.363 ms | 85.846 ms | **1.01x** |
| Array Operations | 46.939 ms | 46.370 ms | **1.01x** |
| Exception Try-Catch | 33.545 ms | 36.238 ms | **0.93x** |
| Quantum CNOT Circuit | 103.980 ms | 113.239 ms | **0.92x** |
