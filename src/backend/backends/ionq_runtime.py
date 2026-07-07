"""IonQ cloud job submission.

A second real-submission backend (alongside the IBM Quantum Runtime
bridge) so Eigen users can target IonQ QPUs and IonQ's cloud simulator
without changing their circuit descriptions. Like the IBM bridge, this
module NEVER auto-runs, NEVER caches the token, NEVER logs the token,
and NEVER persists the token in exception messages.

Unlike the IBM bridge, this is implemented against IonQ's public REST
API (`https://api.ionq.co/v0.3`) using stdlib `urllib.request` only — so
no optional pip-install extra is required. Token storage and credential
handling remain the caller's responsibility; this module touches the
token only long enough to set the `Authorization` header on a HTTPS
request, then lets it go out of scope.

The IonQ REST shape:
  POST /v0.3/jobs                          (submit, returns {"id": "..."})
  GET  /v0.3/jobs/{id}                     (poll status)
  GET  /v0.3/jobs/{id}/result              (retrieve histograms/probabilities)

We deliberately translate EQIR gates to the IonQ gate vocabulary (h, x,
y, z, s, t, rx, ry, rz, cnot, cz, swap, ...) one-to-one; multi-controlled
and rotation-controlled variants that IonQ doesn't natively support are
rejected with a clear `UnsupportedGateError` BEFORE any network call so
the user gets an actionable error message rather than a 4xx from IonQ.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

from src.ir.ir_graph import EQIRGraph


logger = logging.getLogger(__name__)


# IonQ REST endpoint and a deliberately modest per-request timeout so a
# mis-routed request does not hang forever.
IONQ_API_BASE = "https://api.ionq.co/v0.3"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0

# IonQ gate vocabulary we can map directly from EQIR GATE nodes. Anything
# not here raises `UnsupportedGateError` before any network call.
_IONQ_GATE_MAP = {
    "H":   ("h",   1, 0),
    "X":   ("x",   1, 0),
    "Y":   ("y",   1, 0),
    "Z":   ("z",   1, 0),
    "S":   ("s",   1, 0),
    "T":   ("t",   1, 0),
    "RX":  ("rx",  1, 1),
    "RY":  ("ry",  1, 1),
    "RZ":  ("rz",  1, 1),
    "CNOT": ("cnot", 2, 0),
    "CZ":  ("cz",  2, 0),
    "SWAP": ("swap", 2, 0),
    "CCX":  ("ccx", 3, 0),  # IonQ accepts ccx natively.
}


class UnsupportedGateError(RuntimeError):
    """Raised when an EQIR gate has no entry in the IonQ vocabulary.

    Bailing out before the network call means the user gets an
    actionable error message listing the offending gate + qubit
    targets rather than a generic 400 from IonQ's API.
    """
    def __init__(self, gate_name: str, targets: list[str]) -> None:
        supported = sorted(_IONQ_GATE_MAP.keys())
        super().__init__(
            f"Gate '{gate_name}' on {targets} has no IonQ equivalent in "
            f"the supported vocabulary {supported}. Transpile the circuit "
            f"to the supported basis before submitting, or use a backend "
            f"that supports this gate natively."
        )


class IonQAPIError(RuntimeError):
    """Raised when IonQ's REST API returns anything other than 2xx or a
    schema we can parse. The HTTP body text (which IonQ documents as
    non-credentialed) is included for diagnosis, but the token is NEVER
    placed in the message.
    """
    def __init__(self, status: int, body: str) -> None:
        body_excerpt = body[:512] if body else "(empty)"
        super().__init__(
            f"IonQ API returned HTTP {status}; body: {body_excerpt}"
        )
        self.status = status
        self.body = body


class IonQJobResult:
    """Structured response from a successful IonQ job submission."""

    def __init__(
        self,
        job_id: str,
        backend: str,
        shots: int,
        status: str,
        counts: dict[str, int] | None = None,
        raw_result: object = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.job_id = job_id
        self.backend = backend
        self.shots = shots
        self.status = status
        self.counts = counts if counts is not None else {}
        self.raw_result = raw_result
        self.warnings = warnings if warnings is not None else []

    def __repr__(self) -> str:
        return (
            f"IonQJobResult(job_id={self.job_id!r}, backend={self.backend!r}, "
            f"shots={self.shots}, status={self.status!r}, "
            f"counts_keys={list(self.counts.keys())})"
        )


def _build_circuit_payload(graph: EQIRGraph) -> tuple[list[dict], list[tuple[str, str]]]:
    """Translate ``graph`` to IonQ's ``circuit`` JSON shape.

    Returns ``(operations, measures)`` where each operation is the dict
    IonQ expects, and measures is a list of (qubit_name, cbit_name) for
    post-processing on our side. The IonQ API only requires MEASURE
    declarations for the qubits you want returned; in our generated
    payload every MEASURE node in the EQIR becomes a measurement of the
    corresponding qubit index.
    """
    qubit_index: dict[str, int] = {}
    next_idx = 0
    for node in graph.topological_sort():
        if node.type == "ALLOC":
            qubit_index[node.targets[0]] = next_idx
            next_idx += 1

    measures: list[tuple[str, str]] = []
    for node in graph.nodes.values():
        if node.type == "MEASURE":
            measures.append((node.targets[0], node.cbit_name))

    operations: list[dict] = []
    for node in graph.topological_sort():
        if node.type != "GATE":
            continue
        g_name = node.gate_name
        spec = _IONQ_GATE_MAP.get(g_name)
        if spec is None:
            raise UnsupportedGateError(g_name, node.targets)
        ionq_name, arity, n_args = spec
        if len(node.targets) != arity:
            raise UnsupportedGateError(g_name, node.targets)
        op: dict = {"gate": ionq_name}
        # IonQ expects qubit target indices, not names. The first target for
        # single-qubit gates is "target"; for two-qubit gates the *control*
        # is "control" and the *target* is "target". We follow Eigen's
        # convention: targets[0] is control, targets[1] is target.
        if arity == 1:
            op["target"] = qubit_index[node.targets[0]]
        elif arity == 2:
            op["control"] = qubit_index[node.targets[0]]
            op["target"] = qubit_index[node.targets[1]]
        elif arity == 3:
            op["controls"] = [qubit_index[node.targets[0]], qubit_index[node.targets[1]]]
            op["target"] = qubit_index[node.targets[2]]
        if n_args == 1:
            if not node.args:
                raise UnsupportedGateError(g_name, node.targets)
            op["rotation"] = float(node.args[0])
        operations.append(op)
    # Add explicit MEASURE statements per IonQ docs, one per measured qubit.
    for q_name, _c_name in measures:
        if q_name in qubit_index:
            operations.append({"gate": "measure", "target": qubit_index[q_name]})
    return operations, measures


def _post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    """POST JSON to ``url`` and return the parsed JSON response. Raises
    `IonQAPIError` on any non-2xx status code. Raises the original URLError
    on transport-level failures so callers can distinguish them.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            payload = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise IonQAPIError(e.code, body_text) from e
    if status < 200 or status >= 300:
        raise IonQAPIError(status, payload)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise IonQAPIError(status, payload) from e


def _get_json(url: str, headers: dict, timeout: float) -> dict:
    try:
        req = urllib.request.Request(url=url, method="GET")
        for k, v in headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            payload = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise IonQAPIError(e.code, body_text) from e
    if status < 200 or status >= 300:
        raise IonQAPIError(status, payload)
    return json.loads(payload)


def submit_to_ionq(
    graph: EQIRGraph,
    api_token: str,
    *,
    backend: str = "qpu",
    job_name: str = "eigen-submitted",
    shots: int = 1024,
    timeout: float = 600.0,
    poll_interval: float = 5.0,
    wait: bool = True,
    api_base: str = IONQ_API_BASE,
) -> IonQJobResult:
    """Compile ``graph`` to IonQ's REST circuit payload and submit it.

    Args:
        graph: the EQIR graph to run.
        api_token: IonQ API key. NEVER loaded from disk here — the caller
            is responsible for credential storage. The token is used
            only as the `Authorization: api_key <token>` header value and
            is otherwise unreferenced.
        backend: IonQ target, one of ``'qpu'`` (real hardware) or
            ``'simulator'`` (IonQ cloud simulator). Defaults to ``'qpu'``.
        job_name: human-readable label for the job.
        shots: number of shots. Defaults to 1024.
        timeout: seconds to wait for the job result if `wait=True`.
        poll_interval: seconds between poll requests.
        wait: if `True`, block until the job completes (or the timeout
            fires); if `False`, return immediately after submission with
            empty counts and the raw job handle.
        api_base: IonQ REST base URL — overridable for testing.

    Returns:
        `IonQJobResult` with job id, backend, status, counts (or empty
        dict if `wait=False`).

    Raises:
        ValueError: if ``api_token`` is empty/None or ``backend`` is not
            'qpu' or 'simulator'.
        UnsupportedGateError: if the graph contains a gate not in IonQ's
            vocabulary. No network call is made in that case.
        IonQAPIError: if IonQ's API returns a non-2xx status, returns
            unparseable JSON, or behaves unexpectedly. The token is
            never placed in the resulting message body.
        urllib.error.URLError: on transport-level failures (DNS, TLS,
            connection refused, etc.).
    """
    if not isinstance(api_token, str) or not api_token:
        raise ValueError("api_token is required (non-empty string)")
    if backend not in ("qpu", "simulator"):
        raise ValueError(
            f"backend must be 'qpu' or 'simulator', got {backend!r}"
        )

    # We hold the Authorization header in a dict that goes ONLY into the
    # urllib request — never logged, never copied into anything that
    # could outlive the call.
    headers = {
        "Authorization": f"api_key {api_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Translate graph to IonQ payload. Does NOT touch the network.
    operations, _measures = _build_circuit_payload(graph)

    submit_body = {
        "target": backend,
        "name": job_name,
        "shots": int(shots),
        "circuit": operations,
    }

    submit_url = f"{api_base.rstrip('/')}/jobs"

    logger.info(
        "Submitting circuit to IonQ (backend=%s, shots=%d, ops=%d, name=%s)",
        backend, shots, len(operations), job_name,
    )
    # NOTE: the Authorization header value is intentionally NOT included in
    # any log line above or below. This is a deliberate privacy contract.
    response = _post_json(submit_url, submit_body, headers, timeout=DEFAULT_TIMEOUT_SECONDS)
    job_id = response.get("id")
    if not job_id:
        raise IonQAPIError(0, f"submission response missing 'id' field: {response!r}")
    logger.info("IonQ job %s submitted to backend %s.", job_id, backend)

    if not wait:
        return IonQJobResult(
            job_id=job_id,
            backend=backend,
            shots=shots,
            status="submitted",
            counts={},
            raw_result=response,
            warnings=["wait=False specified; counts not yet retrieved"],
        )

    deadline = time.monotonic() + timeout
    last_status: str = "submitted"
    poll_url = f"{submit_url}/{job_id}"
    while time.monotonic() < deadline:
        try:
            poll_resp = _get_json(poll_url, headers, timeout=DEFAULT_TIMEOUT_SECONDS)
        except IonQAPIError as e:
            # Transient 5xx are common during heavy queue load; retry.
            if 500 <= e.status < 600:
                time.sleep(poll_interval)
                continue
            raise
        last_status = poll_resp.get("status", "unknown")
        if last_status in ("ready", "failed", "canceled"):
            break
        time.sleep(poll_interval)

    if last_status != "ready":
        return IonQJobResult(
            job_id=job_id,
            backend=backend,
            shots=shots,
            status=last_status,
            counts={},
            raw_result=poll_resp if 'poll_resp' in locals() else None,
            warnings=[
                f"job did not reach 'ready' within {timeout}s "
                f"or ended in status {last_status!r}"
            ],
        )

    # Fetch results.
    result_url = f"{poll_url}/result"
    result_resp = _get_json(result_url, headers, timeout=DEFAULT_TIMEOUT_SECONDS)
    counts = _extract_counts(result_resp, shots)
    return IonQJobResult(
        job_id=job_id,
        backend=backend,
        shots=shots,
        status="ready",
        counts=counts,
        raw_result=result_resp,
        warnings=[],
    )


def _extract_counts(result: dict, shots: int) -> dict[str, int]:
    """IonQ returns probabilities in the `metadata` / `probabilities` block;
    older payloads used `histogram`. Convert to integer counts for parity
    with the IBM bridge's output shape."""
    if not isinstance(result, dict):
        return {}
    # v0.3 prefers `probabilities: {"0": 0.97, "1": 0.03}` at top level.
    probs = result.get("probabilities")
    if probs is None:
        # Some payloads nest under `metadata.histogram`.
        meta = result.get("metadata") or {}
        if isinstance(meta, dict):
            probs = meta.get("histogram") or meta.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return {}
    out: dict[str, int] = {}
    for bitstring, p in probs.items():
        try:
            int_count = int(round(float(p) * shots))
        except (TypeError, ValueError):
            continue
        out[bitstring] = int_count
    return out


__all__ = [
    "IONQ_API_BASE",
    "UnsupportedGateError",
    "IonQAPIError",
    "IonQJobResult",
    "submit_to_ionq",
]
