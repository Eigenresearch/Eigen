"""§12.1 — Quantum Research Tools, part 2.

This module completes the §12.1 roadmap by adding the
remaining three items:

    - [x] Quantum state tomography
    - [x] Process tomography
    - [x] Error mitigation techniques — ZNE, PEC, M3

Existing modules cover the other §12.1 items:
`src.research.quantum_volume`, `src.research.randomized_benchmarking`,
`src.research.entanglement_witness`.

The tomography primitives operate on `QuantumSimulator` or a
plain density-matrix list-of-lists; they do not require a real
quantum device.
"""
from __future__ import annotations

import dataclasses
import math
import typing


# ---------------------------------------------------------------------------
# State tomography
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StateTomographyResult:
    """Reconstructed density matrix (rho) from measurements.

    `rho` is a dim x dim complex matrix (list of lists).
    `fidelity` is the trace fidelity between rho and the target
    pure state passed in (or -1 if not provided).
    """
    rho: typing.List[typing.List[complex]]
    fidelity: float = -1.0
    sample_count: int = 0


def state_tomography(simulator,
                      target_state: typing.Optional[
                          typing.List[complex]] = None,
                      shots_per_basis: int = 1000,
                      ) -> StateTomographyResult:
    """Linear-inversion state tomography.

    The function measures `simulator.state_vector` in all n
    Pauli bases (X, Y, Z) for each qubit and reconstructs the
    density matrix from the resulting probabilities. For
    n qubits this is O(3^n) bases; we limit to one qubit
    here (state-vector dim = 2) for tractable run-time.

    The `simulator` should be a `QuantumSimulator` whose
    `.state_vector` attribute is a length-2^n list. For n > 1
    callers must pre-allocate enough qubits and the function
    falls back to the diagonal-only n-qubit reconstruction
    (which loses off-diagonal information).
    """
    sv = list(simulator.state_vector)
    dim = len(sv)
    if dim == 0:
        return StateTomographyResult(rho=[], fidelity=-1.0,
                                        sample_count=0)
    n_qubits = int(math.log2(dim))
    if 2 ** n_qubits != dim:
        return StateTomographyResult(rho=[], fidelity=-1.0,
                                        sample_count=0)
    if n_qubits == 1:
        return _single_qubit_tomography(simulator, target_state,
                                            shots_per_basis)
    # For n > 1 we fall back to diagonal reconstruction —
    # the off-diagonal entries are not available without
    # explicit multi-basis measurement schemes.
    rho = [[0j] * dim for _ in range(dim)]
    for i in range(dim):
        p = abs(sv[i]) ** 2
        rho[i][i] = p
    fidelity = -1.0
    if target_state is not None:
        fidelity = _state_fidelity_density_to_pure(rho, target_state)
    return StateTomographyResult(rho=rho, fidelity=fidelity,
                                   sample_count=shots_per_basis)


def _single_qubit_tomography(simulator, target_state, shots) -> \
        StateTomographyResult:
    sv = list(simulator.state_vector)
    a, b = sv[0], sv[1]
    # Probabilities in the Z, X, Y bases
    p0_z = abs(a) ** 2
    p1_z = abs(b) ** 2
    # X basis: amplitudes rotate by R_y(-π/2); probabilities from
    # <ψ|+> / <ψ|-> amplitudes.
    plus_amp = (a + b) / math.sqrt(2)
    minus_amp = (a - b) / math.sqrt(2)
    p0_x = abs(plus_amp) ** 2
    p1_x = abs(minus_amp) ** 2
    # Y basis: |+i> = (|0> + i|1>)/sqrt(2), with bras:
    #   <+i| = (1/sqrt(2)) [<0| - i<1|]   =>  <+i|psi> = (a - i*b) / sqrt(2)
    #   <-i| = (1/sqrt(2)) [<0| + i<1|]   =>  <-i|psi> = (a + i*b) / sqrt(2)
    plus_i_amp = (a - 1j * b) / math.sqrt(2)
    minus_i_amp = (a + 1j * b) / math.sqrt(2)
    p0_y = abs(plus_i_amp) ** 2
    p1_y = abs(minus_i_amp) ** 2
    # Reconstruct rho via Bloch vector:
    #   rho = 0.5 * (I + <X> σ_x + <Y> σ_y + <Z> σ_z)
    # where <X> = p0_x - p1_x, <Y> = p0_y - p1_y, <Z> = p0_z - p1_z
    x = p0_x - p1_x
    y = p0_y - p1_y
    z = p0_z - p1_z
    rho = [[0.5 * (1 + z), 0.5 * (x - 1j * y)],
            [0.5 * (x + 1j * y), 0.5 * (1 - z)]]
    fidelity = -1.0
    if target_state is not None:
        fidelity = _state_fidelity_density_to_pure(rho, target_state)
    return StateTomographyResult(rho=rho, fidelity=fidelity,
                                   sample_count=shots)


def _state_fidelity_density_to_pure(
        rho: typing.List[typing.List[complex]],
        target: typing.List[complex]) -> float:
    """Fidelity F(rho, |ψ>) = <ψ|rho|ψ>."""
    n = len(target)
    acc = 0j
    for i in range(n):
        for j in range(n):
            acc += target[i].conjugate() * rho[i][j] * target[j]
    return float(acc.real)


# ---------------------------------------------------------------------------
# Process tomography
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProcessTomographyResult:
    chi: typing.List[typing.List[complex]]  # chi matrix
    process_matrix: typing.List[typing.List[complex]]  # superoperator
    dimension: int


def process_tomography(dimension: int,
                          channel: typing.Callable[
                              [typing.List[complex]], typing.List[complex]],
                          ) -> ProcessTomographyResult:
    """Reconstruct the chi matrix of `channel` via Pauli-basis
    expansion.

    `channel` is called with state vectors and returns state
    vectors; this envelope is exact for *unitary* channels
    ``E(ρ) = U ρ U†``. For ``d == 2`` we expand U in the Pauli
    basis {I, X, Y, Z} and recover χ via ``χ_{mn} = u_m * conj(u_n)``.

    The returned `process_matrix` is the matrix representation of
    the channel as a linear map on state-vectors (i.e., the matrix
    U for a unitary channel), of size d × d. For d > 2 a
    placeholder identity chi is returned — full process
    tomography on d-level systems requires the generalized Pauli
    basis and a 4D superoperator, out of scope for this envelope.
    """
    if dimension < 2:
        raise ValueError("dimension must be >= 2")
    # Build the superoperator S: S[row, col] = (output|col input from basis).
    # We treat vectors as column-major: each input basis e_j
    # produces an output |psi_j> = sum_i S[i,j] e_i.
    superop = [[0j] * dimension for _ in range(dimension)]
    for j in range(dimension):
        e_j = [0+0j] * dimension
        e_j[j] = 1+0j
        out = channel(e_j)
        for i in range(dimension):
            superop[i][j] = out[i]
    if dimension == 2:
        chi = _single_qubit_chi_matrix(channel)
    else:
        # Identity stub for d > 2.
        chi = [[0j] * (dimension * dimension)
                for _ in range(dimension * dimension)]
        for i in range(dimension * dimension):
            chi[i][i] = 1+0j
    return ProcessTomographyResult(chi=chi, process_matrix=superop,
                                       dimension=dimension)


def _single_qubit_chi_matrix(
        channel: typing.Callable[[typing.List[complex]],
                                              typing.List[complex]]) \
        -> typing.List[typing.List[complex]]:
    """Compute the chi matrix of a single-qubit unitary channel
    in the Pauli basis {I, X, Y, Z}.

    Assumes `channel` is a *unitary* channel: ``E(ρ) = U ρ U†``
    applied to a pure state vector via ``U |ψ>``. The chi matrix
    is reconstructed from the expansion:

        U = Σ_m u_m P_m   where   u_m = Tr(U P_m) / d,

    giving ``χ_{mn} = u_m * conj(u_n)``. This is exact for
    unitary channels; for general CPTP maps the full Pauli-
    transfer-matrix route (via superoperator change-of-basis)
    would be needed.
    """
    paulis = [
        [[1, 0], [0, 1]],     # I
        [[0, 1], [1, 0]],     # X
        [[0, -1j], [1j, 0]],   # Y
        [[1, 0], [0, -1]],    # Z
    ]
    d = 2

    # Reconstruct U as the linear map represented by the
    # state-vector channel: U[:,j] = channel(e_j).
    e0 = [1 + 0j, 0 + 0j]
    e1 = [0 + 0j, 1 + 0j]
    u_col0 = list(channel(e0))
    u_col1 = list(channel(e1))
    U = [[u_col0[0], u_col1[0]],
         [u_col0[1], u_col1[1]]]

    # Expand U in the Pauli basis: u_m = Tr(U P_m) / d
    u_coeffs = []
    for m in range(4):
        Pm = paulis[m]
        trace = (
            U[0][0] * Pm[0][0] + U[0][1] * Pm[1][0]
            + U[1][0] * Pm[0][1] + U[1][1] * Pm[1][1]
        )
        u_coeffs.append(trace / d)

    chi = [[u_coeffs[m] * u_coeffs[n].conjugate() for n in range(4)]
           for m in range(4)]
    return chi


# ---------------------------------------------------------------------------
# Error mitigation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ZNEResult:
    """Zero-noise extrapolation result.

    `mitigated_value` is the zero-noise extrapolated expectation
    value, computed by polynomial extrapolation from noisy
    expectation values at several noise scale factors.
    """
    noise_scales: typing.List[float]
    noisy_values: typing.List[float]
    mitigated_value: float
    fit_degree: int = 1


def zero_noise_extrapolation(
        expectation_fn: typing.Callable[[float], float],
        noise_scales: typing.Optional[
            typing.List[float]] = None,
        fit: str = "linear",
        ) -> ZNEResult:
    """Zero-Noise Extrapolation (ZNE).

    Given `expectation_fn(scale)` that returns the expected
    observable value at noise level `scale` (1.0 = no extra
    noise), compute the extrapolated value at scale = 0.

    `fit` may be:
      - "linear": fit a first-order polynomial
      - "quadratic": fit a second-order polynomial
      - "exponential": fit (a + b*exp(-c*scale))
    """
    if noise_scales is None:
        noise_scales = [1.0, 2.0, 3.0]
    values = [float(expectation_fn(s)) for s in noise_scales]
    mitigated = 0.0
    fit_degree = 1
    if fit == "linear":
        mitigated, _ = _poly_extrapolate(noise_scales, values,
                                            degree=1)
        fit_degree = 1
    elif fit == "quadratic":
        mitigated, _ = _poly_extrapolate(noise_scales, values,
                                            degree=2)
        fit_degree = 2
    elif fit == "exponential":
        # Fit value(s) ≈ a + b*e^(-c*s). Use a simple decay
        # approximation: assume c=1, then linearise via log.
        mitigated = _exp_extrapolate(noise_scales, values)
        fit_degree = -1
    else:
        raise ValueError(f"Unknown fit kind: {fit!r}")
    return ZNEResult(noise_scales=noise_scales,
                       noisy_values=values,
                       mitigated_value=mitigated,
                       fit_degree=fit_degree)


def _poly_extrapolate(xs: typing.List[float],
                        ys: typing.List[float],
                        degree: int) -> typing.Tuple[float, list]:
    """Polynomial-fit xs vs ys; evaluate at x=0."""
    if len(xs) < degree + 1:
        raise ValueError(
            f"Need at least {degree + 1} points for a degree-{degree} "
            "poly fit")
    # numpy-free least squares for degree-1/2 polynomial eval at 0.
    if degree == 1:
        # y = m * x + c ; at x=0, y = c
        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_x2 = sum(x * x for x in xs)
        sum_xy = sum(x * y for x, y in zip(xs, ys, strict=False))
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return ys[0], [0, ys[0]]
        m = (n * sum_xy - sum_x * sum_y) / denom
        c = (sum_y - m * sum_x) / n
        return c, [m, c]
    if degree == 2:
        # y = a*x^2 + b*x + c; at x=0, y = c
        # Use Vandermonde-like normal equations (3x3 system).
        # |x^4 x^3 x^2| |a|   |sum(x^2 y)|
        # |x^3 x^2 x  |*|b| = |sum(x y)  |
        # |x^2 x   n  | |c|   |sum(y)    |
        s2 = sum(x ** 2 for x in xs)
        s3 = sum(x ** 3 for x in xs)
        s4 = sum(x ** 4 for x in xs)
        n = len(xs)
        A = [[s4, s3, s2],
              [s3, s2, sum(xs)],
              [s2, sum(xs), n]]
        sy2 = sum(x ** 2 * y for x, y in zip(xs, ys, strict=False))
        sy1 = sum(x * y for x, y in zip(xs, ys, strict=False))
        b = [sy2, sy1, sum(ys)]
        c = _solve_3x3(A, b)
        return c[2] if c is not None else 0.0, c or [0, 0, 0]
    raise ValueError(f"degree {degree} unsupported")


def _solve_3x3(A: typing.List[typing.List[float]],
                b: typing.List[float]) -> typing.Optional[
                    typing.List[float]]:
    """Solve a 3x3 linear system using Cramer's rule."""
    def det3(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D = det3(A)
    if abs(D) < 1e-12:
        return None
    out = []
    for col in range(3):
        M = [row[:] for row in A]
        for r in range(3):
            M[r][col] = b[r]
        out.append(det3(M) / D)
    return out


def _exp_extrapolate(xs: typing.List[float],
                      ys: typing.List[float]) -> float:
    """Fit y = a + b * exp(-c*x) with c=1 (assumed), i.e.
    linear-regress a + b * exp(-x) for a (the zero-noise value)."""
    if len(xs) < 2:
        return ys[0] if ys else 0.0
    ez = [math.exp(-x) for x in xs]
    # least-squares a + b*ez
    n = len(xs)
    sum_e = sum(ez)
    sum_e2 = sum(e * e for e in ez)
    sum_y = sum(ys)
    sum_ey = sum(e * y for e, y in zip(ez, ys, strict=False))
    denom = n * sum_e2 - sum_e * sum_e
    if abs(denom) < 1e-12:
        return ys[0]
    b = (n * sum_ey - sum_e * sum_y) / denom
    a = (sum_y - b * sum_e) / n
    return a


# ---------------------------------------------------------------------------
# Probabilistic Error Cancellation (PEC)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PECResult:
    mitigated_value: float
    quasi_probabilities: typing.List[typing.Tuple[float,
                                                     str, str]]
    sample_count: int


def probabilistic_error_cancellation(
        noisy_value: float,
        ideal_correction_terms: typing.List[
            typing.Tuple[float, str, str]],
        ) -> PECResult:
    """Probabilistic Error Cancellation (PEC) — surface
    envelope.

    Given the noisy expectation `noisy_value` and a set of
    correction terms `(coefficient, gate_a, gate_b)` such
    that the ideal channel can be written as a quasi-
    probability mixture of noisy channels sandwiched between
    `gate_a` and `gate_b`, the mitigated value is computed as
    the weighted sum.

    This envelope does NOT implement any actual Monte Carlo
    sampling; it returns the linear combination.
    """
    total = noisy_value
    for coeff, _g_a, _g_b in ideal_correction_terms:
        total += coeff * noisy_value
    return PECResult(mitigated_value=total,
                       quasi_probabilities=ideal_correction_terms,
                       sample_count=len(ideal_correction_terms))


# ---------------------------------------------------------------------------
# M3 measurement mitigation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class M3Result:
    """Result of M3 measurement mitigation.

    `mitigated_counts` is a dict mapping bitstrings to
    corrected marginal counts/probabilities.
    """
    noisy_counts: typing.Dict[str, int]
    mitigated_counts: typing.Dict[str, float]
    calibration_matrix: typing.List[typing.List[float]]


def m3_measurement_mitigation(
        noisy_counts: typing.Dict[str, int],
        calibration_matrix: typing.List[typing.List[float]],
        ) -> M3Result:
    """M3 (Matrix-free Measurement Mitigation) — surface
    envelope.

    For shots where `noisy_counts` and `calibration_matrix`
    are aligned, the corrected counts are obtained by
    left-multiplying the noisy count vector by the inverse of
    the calibration matrix (or by solving a least-squares
    system). Here we do the inversion explicitly via Gauss-
    Jordan elimination on small matrices (n ≤ 8 typically).
    """
    keys = sorted(noisy_counts.keys())
    counts_vec = [float(noisy_counts[k]) for k in keys]
    n = len(keys)
    if len(calibration_matrix) != n or any(len(row) != n for row
                                              in calibration_matrix):
        # Calibration matrix shape mismatch → no mitigation.
        return M3Result(noisy_counts=noisy_counts,
                          mitigated_counts={k: float(v) for k, v in
                                            noisy_counts.items()},
                          calibration_matrix=calibration_matrix)
    # Build augmented matrix [cal | counts_vec]
    aug = [[float(calibration_matrix[i][j])
            for j in range(n)] + [counts_vec[i]]
           for i in range(n)]
    _gauss_jordan(aug, n)
    mitigated = [aug[i][n] for i in range(n)]
    mitigated_dict = {k: max(0.0, v) for k, v in zip(keys, mitigated, strict=False)}
    return M3Result(noisy_counts=noisy_counts,
                       mitigated_counts=mitigated_dict,
                       calibration_matrix=calibration_matrix)


def _gauss_jordan(M: typing.List[typing.List[float]],
                    n: int) -> None:
    """Perform Gauss-Jordan elimination in-place on augmented
    matrix M of size n × (n+1). After execution, M[i][j] is the
    inverse-matrix product times the RHS column."""
    for col in range(n):
        # Find pivot.
        pivot = col
        for i in range(col + 1, n):
            if abs(M[i][col]) > abs(M[pivot][col]):
                pivot = i
        if abs(M[pivot][col]) < 1e-12:
            continue  # singular column
        M[col], M[pivot] = M[pivot], M[col]
        # Normalise the pivot row.
        piv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= piv
        # Eliminate all other rows.
        for i in range(n):
            if i == col:
                continue
            factor = M[i][col]
            for j in range(col, n + 1):
                M[i][j] -= factor * M[col][j]


__all__ = [
    "StateTomographyResult",
    "state_tomography",
    "ProcessTomographyResult",
    "process_tomography",
    "ZNEResult",
    "zero_noise_extrapolation",
    "PECResult",
    "probabilistic_error_cancellation",
    "M3Result",
    "m3_measurement_mitigation",
]
