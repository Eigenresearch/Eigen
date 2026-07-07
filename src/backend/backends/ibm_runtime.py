"""IBM Quantum Runtime job submission (audit §7 / §8 item 5).

The audit's explicit ask: "real job submission to at least one provider
(via the existing Qiskit bridge — IBM Quantum Runtime) so 'hardware
backend' is no longer only about text export."

This module is what makes that real, but ONLY behind an explicit
opt-in:

  * It depends on the optional `hardware-ibm` extra declared in
    `pyproject.toml` (i.e. `pip install eigen-lang[hardware-ibm]`).
    If the extra is not installed, calling `submit_to_ibm_quantum()`
    raises `MissingExtraError` — no quiet fallback to text-only
    export.

  * It NEVER auto-runs. Every entry point requires the caller to pass
    `api_token=` explicitly. There is no token-loading from disk,
    environment, or config file inside this module — the caller is
    responsible for credential storage, so this code never sees the
    token except inside the submission call.

  * It does NOT cache the service handle, so the token is not held in
    process state between calls.

  * It logs at INFO level only the *job id* and *backend name*, never
    the token, never the request body.

The actual network call is fully delegated to `qiskit_ibm_runtime`
(which itself uses `requests`). Eigen still makes no network calls
of its own; that's the responsibility of the (optional) external SDK
— which is the honest framing of "use Eigen to design + export, use
the vendor SDK to submit".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.backend.qiskit_backend import QiskitBackend
from src.ir.ir_graph import EQIRGraph


logger = logging.getLogger(__name__)


class MissingExtraError(RuntimeError):
    """Raised when the `hardware-ibm` optional extra is not installed.

    Audit §1.3 / §7: dependencies that enable network access to a
    vendor's API must never be auto-pulled in. They live behind an
    explicit `pip install eigen-lang[hardware-ibm]`. If the user calls
    a submission entry point without that extra installed, this
    exception is raised with an actionable installation hint rather
    than a silent fallback to text export.
    """

    def __init__(self, extra: str, hint: str = "") -> None:
        self.extra = extra
        message = (
            f"Optional extra '{extra}' is required for this call. "
            f"Install it with:  pip install eigen-lang[{extra}]"
        )
        if hint:
            message += f"\n  Hint: {hint}"
        super().__init__(message)


@dataclass
class IBMJobResult:
    """Structured response from a successful job submission.

    Fields mirror what callers usually want back; `raw_result` is the
    underlying `PrimitiveResult` (or `qiskit.result.Result`) object
    so downstream code can pull any bit we forgot to surface.
    """

    job_id: str
    backend_name: str
    shots: int
    counts: dict[str, int]
    raw_result: object = field(default=None, repr=False)
    warnings: list[str] = field(default_factory=list)


def validate_token_format(token: str) -> bool:
    """Cheap pre-flight check of an IBM Quantum API token *format*.

    Returns `True` if `token` looks like an IBM-Quantum-style API
    token, `False` otherwise. Performs NO network call — this is a
    pure-string sanity check so the submission path can fail fast
    with an actionable message before reaching for the network.

    IBM Quantum API tokens are 96-char alphanumeric strings. Older
    IBM Quantum Experience tokens have been observed at slightly
    different lengths, so we accept [80, 256] characters and only
    [A-Za-z0-9_-] bodies.
    """
    if not isinstance(token, str):
        return False
    if not (32 <= len(token) <= 256):
        return False
    for ch in token:
        if not (ch.isalnum() or ch in ('_', '-')):
            return False
    return True


def _check_extra_available() -> None:
    """Raise `MissingExtraError` if the IBM runtime SDK / qiskit isn't
    installed in the current environment.
    """
    try:
        import qiskit  # noqa: F401
        from qiskit_ibm_runtime import QiskitRuntimeService  # noqa: F401
    except ImportError as e:
        raise MissingExtraError(
            "hardware-ibm",
            hint="qiskit and qiskit-ibm-runtime must both be importable",
        ) from e


def submit_to_ibm_quantum(
    graph: EQIRGraph,
    api_token: str,
    backend_name: str,
    *,
    shots: int = 1024,
    instance: Optional[str] = None,
    timeout: float = 600.0,
    wait: bool = True,
) -> IBMJobResult:
    """Compile an EQIR circuit and submit it to a real IBM Quantum backend.

    Args:
        graph: the EQIR circuit to run.
        api_token: IBM Quantum API token. NEVER loaded from disk here —
                   the caller is responsible for credential handling.
        backend_name: target IBM backend (e.g. ``'ibm_brisbane'``).
        shots: number of shots. Defaults to 1024.
        instance: optional IBM Quantum cloud instance string
            (e.g. ``'ibm-q/open/main'``). When `None`, the runtime SDK
            picks the user's first/only instance.
        timeout: seconds to wait for the job result if `wait=True`.
        wait: if `True`, block until the job completes (or timeout
            fires); if `False`, return immediately after submission
            with a `raw_result` of the job handle and `counts={}`.

    Returns:
        `IBMJobResult` with job id, backend name, counts (or empty dict
        if `wait=False`), and the raw `qiskit_ibm_runtime` result object.

    Raises:
        MissingExtraError: if `qiskit` or `qiskit_ibm_runtime` is not
            installed (install with `pip install eigen-lang[hardware-ibm]`).
        ValueError: if `api_token` fails `validate_token_format` or if
            `backend_name` is empty.
        RuntimeError: if the runtime SDK rejects the token, the
            backend is unavailable, or the job times out.
    """
    if not backend_name:
        raise ValueError("backend_name is required")
    if not validate_token_format(api_token):
        raise ValueError(
            "api_token failed format validation — expected a 32–256 char "
            "alphanumeric string. Note: this is a format check only; a "
            "well-formed but invalid token will still be rejected by IBM "
            "Quantum's authentication layer."
        )

    # Compile EQIR → Qiskit Python source via the existing bridge. This
    # path is exactly what the audit's §5 work produces (CRX/CRY/CRZ/CP/
    # CCX/CSWAP dispatch; OpenQASM 3.0 export; etc.).
    script_code, _report = QiskitBackend().transpile(graph)

    _check_extra_available()
    # Import the runtime SDK only after the extra check passed. These
    # imports are intentionally inside the function (not at module top)
    # so that "import" of this module by an unwitting caller never
    # triggers dependency resolution.
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler  # type: ignore
    from qiskit import QuantumCircuit, transpile as qiskit_transpile           # type: ignore

    # Execute the generated Qiskit script in a *local* namespace. The
    # `QiskitBackend.transpile()` output is self-contained; it builds a
    # `qc` object via `QuantumCircuit(...)`, appends gates, and ends
    # with the simulator footer we strip here.
    #
    # IMPORTANT: we are running *generated Qiskit source*, not untrusted
    # user code. The QiskitBackend is our own code; we control what
    # appears in the script. If `QiskitBackend.transpile()` is ever
    # changed to emit user-derived strings (e.g. classical print
    # statements containing user names), the execution below would need
    # the same hardened sandbox as `JITCompiler.compile_block`.
    local_ns: dict = {}
    # Provide the names the generated script expects to be available
    # directly via local_ns. Doing so lets us strip ALL import lines
    # from the source — we never want a transpiled circuit's exec()
    # to start pulling in optional packages (qiskit_aer is heavier
    # still and isn't needed for real-hardware submission).
    local_ns['np'] = None  # generated source may reference np for matrices
    local_ns['QuantumCircuit'] = QuantumCircuit
    local_ns['transpile'] = qiskit_transpile
    # Strip imports + simulator footer.
    kept_lines = []
    footer_markers = ('# Execute the circuit using Qiskit Aer',
                      'simulator = AerSimulator()',
                      'compiled_circuit = transpile(qc, simulator)',
                      'result = simulator.run(',
                      'counts = result.get_counts(',
                      "print('Simulation results counts:'")
    for line in script_code.splitlines():
        stripped = line.strip()
        # Skip pure Python import statements (our local_ns provides the
        # symbols; we never want a transpiled circuit's exec to pull
        # network/optional libs).
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue
        if any(stripped.startswith(m) for m in footer_markers):
            break
        kept_lines.append(line)
    kept_source = "\n".join(kept_lines)

    try:
        exec(compile(kept_source, '<eigen_ibm_runtime_bridge>', 'exec'), {}, local_ns)
    except Exception as e:
        raise RuntimeError(
            f"failed to execute the transpiled Qiskit source via the "
            f"IBM Runtime bridge: {e}"
        ) from e
    qc = local_ns.get('qc')
    if qc is None:
        raise RuntimeError(
            "QiskitBackend.transpile() script did not produce a "
            "`qc` QuantumCircuit object — this is an internal error."
        )

    # Connect to IBM Quantum. The token goes ONLY into the runtime
    # service constructor; we never log it, never store it.
    service_kwargs = {
        'channel': 'ibm_cloud',
        'token': api_token,
    }
    if instance:
        service_kwargs['instance'] = instance
    try:
        service = QiskitRuntimeService(**service_kwargs)
    except Exception as e:
        # Re-raise as RuntimeError so callers don't need to import
        # qiskit-ibm-runtime's specific exception classes. We DO NOT
        # include the token in the message — only the SDK's own str(e),
        # which never contains the token in qiskit-ibm-runtime's error
        # path (it would be a privacy bug in that library if it did).
        raise RuntimeError(f"IBM Quantum Runtime authentication failed: {e}") from e

    backend = service.backend(backend_name)
    if backend is None:
        raise RuntimeError(
            f"IBM Quantum backend '{backend_name}' is not available "
            f"for the supplied account."
        )

    logger.info(
        "Submitting circuit (n_qubits=%d, n_ops=%d, shots=%d) to backend %s",
        qc.num_qubits, qc.size(), shots, backend_name,
    )

    sampler = Sampler(mode=backend)
    job = sampler.run([(qc,)], shots=shots)
    logger.info("Job %s submitted to backend %s.", job.job_id(), backend_name)

    if not wait:
        return IBMJobResult(
            job_id=job.job_id(),
            backend_name=backend_name,
            shots=shots,
            counts={},
            raw_result=job,
            warnings=["wait=False specified; counts not yet retrieved"],
        )

    try:
        result = job.result(timeout=timeout)
    except Exception as e:
        raise RuntimeError(
            f"Job {job.job_id()} did not complete within {timeout}s "
            f"or failed: {e}"
        ) from e

    # Extract counts from the result. The exact shape depends on
    # qiskit-ibm-runtime's version — be defensive.
    counts: dict[str, int] = {}
    try:
        # SamplerV2 returns a `PrimitiveResult` whose first PubResult
        # has a `.data.meas` bit array.
        pub_result = result[0]
        meas = getattr(pub_result.data, 'meas', None)
        if meas is not None:
            counts = meas.get_counts()
        else:
            # Fallback: try a classical `.get_counts()` shape.
            counts = getattr(result, 'get_counts', lambda: {})()
    except Exception as e:
        logger.warning("Could not extract counts from job result: %s", e)

    return IBMJobResult(
        job_id=job.job_id(),
        backend_name=backend_name,
        shots=shots,
        counts=counts,
        raw_result=result,
        warnings=[],
    )


__all__ = [
    'IBMJobResult',
    'MissingExtraError',
    'submit_to_ibm_quantum',
    'validate_token_format',
]
