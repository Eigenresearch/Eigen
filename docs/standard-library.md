# Eigen Standard Library Documentation

This document describes the API and implementation details of the Eigen standard library (`stdlib/`).

## 1. Module `quantum.bell`
File: [stdlib/quantum/bell.eig](file:///d:/Nuras-7/stdlib/quantum/bell.eig)

Provides subroutines to create maximally entangled states.

### `bell(qubit a, qubit b)`
Creates a Bell state \(| \Phi^+ \rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}\) across two qubits.

**Definition**:
```eigen
qfunc bell(qubit a, qubit b) {
    H a
    CNOT a, b
    return
}
```

---

## 2. Module `quantum.ghz`
File: [stdlib/quantum/ghz.eig](file:///d:/Nuras-7/stdlib/quantum/ghz.eig)

Provides subroutines to create Greenberger–Horne–Zeilinger (GHZ) states.

### `ghz(qubit a, qubit b, qubit c)`
Creates a 3-qubit entangled GHZ state \(| \text{GHZ} \rangle = \frac{|000\rangle + |111\rangle}{\sqrt{2}}\).

**Definition**:
```eigen
qfunc ghz(qubit a, qubit b, qubit c) {
    H a
    CNOT a, b
    CNOT b, c
    return
}
```

---

## 3. Module `quantum.deutsch`
File: [stdlib/quantum/deutsch.eig](file:///d:/Nuras-7/stdlib/quantum/deutsch.eig)

Provides standard oracles for testing the Deutsch algorithm.

### `oracle_constant_0(qubit x, qubit y)`
Constant oracle mapping \(f(x) = 0\). Does nothing.

### `oracle_constant_1(qubit x, qubit y)`
Constant oracle mapping \(f(x) = 1\). Applies \(X\) to the target qubit \(y\).

### `oracle_balanced_x(qubit x, qubit y)`
Balanced oracle mapping \(f(x) = x\). Applies a \(CNOT\) controlled by \(x\) onto target \(y\).

### `oracle_balanced_not_x(qubit x, qubit y)`
Balanced oracle mapping \(f(x) = \neg x\). 

**Definition**:
```eigen
qfunc oracle_balanced_not_x(qubit x, qubit y) {
    X x
    CNOT x, y
    X x
    return
}
```

---

## 4. Module `quantum.grover`
File: [stdlib/quantum/grover.eig](file:///d:/Nuras-7/stdlib/quantum/grover.eig)

Provides components for the Grover Database Search algorithm.

### `diffuse_2(qubit q0, qubit q1)`
Applies the Grover diffusion operator (inversion about the mean) on 2 qubits.

**Definition**:
```eigen
qfunc diffuse_2(qubit q0, qubit q1) {
    H q0
    H q1
    X q0
    X q1
    CZ q0, q1
    X q0
    X q1
    H q0
    H q1
    return
}
```

---

## 5. Runtime Guarantees and Backend Compatibility

### Runtime Guarantees
All standard library modules are guaranteed to compile, resolve namespace references, inline properly during EQIR generation, and run on both the **Eigen VM** and **topological Runtime**.

### Backend Compatibility
Because the standard library is written as pure, static quantum subroutines (`qfunc`), it possesses maximum compatibility across all compilation targets:

| Module / Function | Eigen VM Target | topological Runtime Target | Qiskit Exporter |
| --- | --- | --- | --- |
| `quantum.bell` | `FULL` | `FULL` | `FULL` |
| `quantum.ghz` | `FULL` | `FULL` | `FULL` |
| `quantum.deutsch` | `FULL` | `FULL` | `FULL` |
| `quantum.grover` | `FULL` | `FULL` | `FULL` |
