# Extreme Performance & Correctness: Native Python vs Eigen VM (Nova 2.6)

Detailed verification of 10 highly complex algorithmic and quantum operations, showing execution speeds and exact outputs to verify compiler correctness.

| Test Case | Native Python Time | Eigen VM Time | Python Output | Eigen VM Output |
| --- | --- | --- | --- | --- |
| 01. 16-Qubit Sparse Circuit | 10.309 ms | 0.233 ms | `Amplitudes: |1000000110000001>: 1.0000+0.0000j` | `Amplitudes: |1000000110000001>: 1.0000+0.0000j` |
| 02. 12-Qubit QFT Simulation | 0.469 ms | 10.090 ms | `Amplitudes: |000>: 0.5000+0.0000j, |001>: 0.5000+0.0000j, |010>: 0.5000+0.0000j, |011>: 0.5000+0.0000j` | `Amplitudes: |000>: 0.5000+0.0000j, |001>: 0.5000+0.0000j, |010>: 0.5000+0.0000j, |011>: 0.5000+0.0000j` |
| 03. 14-Qubit GHZ State | 2.645 ms | 124.446 ms | `Amplitudes: |000>: 0.7071+0.0000j, |11111111111111>: 0.7071+0.0000j` | `Amplitudes: |000>: 0.7071+0.0000j, |11111111111111>: 0.7071+0.0000j` |
| 04. Fibonacci 22 Recursion | 2.592 ms | 348.910 ms | `Result: 17711` | `Result: 17711` |
| 05. Array Bubble Sort 25 | 0.041 ms | 6.921 ms | `Result: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]` | `Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]` |
| 06. Bernstein-Vazirani s=341 | 0.498 ms | 35.652 ms | `Amplitudes: |000>: 1.0000+0.0000j` | `Amplitudes: |000>: 1.0000+0.0000j` |
| 07. Ackermann Recursion 3,2 | 0.036 ms | 4.647 ms | `Result: 29` | `Result: 29` |
| 08. Nested Exceptions | 0.180 ms | 9.269 ms | `Result: 668` | `Catches: 668` |
| 09. Complex Struct/Map Logic | 0.263 ms | 15.837 ms | `Result: 4497` | `Map['key']: 4497` |
| 10. Phase Estimation QPE-4 | 0.140 ms | 0.621 ms | `Amplitudes: |10000>: 0.2500+0.0000j, |10001>: -0.0000+0.2500j, |10010>: -0.2500+0.0000j, |10011>: -0.0000-0.2500j` | `Amplitudes: |10000>: 0.2500+0.0000j, |10001>: -0.0000+0.2500j, |10010>: -0.2500+0.0000j, |10011>: -0.0000-0.2500j` |
