"""Tests for the IonQ submission bridge (`src/backend/backends/ionq_runtime.py`).

The bridge makes NO network calls when these tests run. We stub out
``urllib.request.urlopen`` with a small fake-response dispatcher that
asserts the URL shape, the headers (specifically ``Authorization``),
and the request body shape, then returns canned JSON.

Token-handling safety contract (mirrors the IBM bridge's):

  * No network call without an explicit non-empty ``api_token``.
  * The ``Authorization`` header carries the token, but the token is
    NEVER placed in exception messages and never logged.
  * Unsupported gates are rejected BEFORE any HTTP call.
"""

from __future__ import annotations

import io
import json
import logging
import unittest
from unittest import mock

from src.backend.backends import ionq_runtime
from src.backend.backends.ionq_runtime import (
    IonQAPIError,
    IonQJobResult,
    UnsupportedGateError,
    submit_to_ionq,
)
from src.ir.ir_graph import EQIRGraph


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #

class _FakeResponse:
    """Fake HTTPS response usable as a context manager (like urlopen)."""
    def __init__(self, status: int, payload: dict | str):
        self.status = status
        if isinstance(payload, (dict, list)):
            self._bytes = json.dumps(payload).encode("utf-8")
        else:
            self._bytes = payload.encode("utf-8")
        self._stream = io.BytesIO(self._bytes)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._stream.close()
        return False

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._stream.read()


class _FakeHTTPError(Exception):
    """Stands in for urllib.error.HTTPError in tests; carries status+body
    just enough for the bridge's `_post_json` / `_get_json` to read
    bytes off it and propagate the message."""
    def __init__(self, code: int, body: bytes):
        super().__init__(f"fake error {code}")
        self.code = code
        self._body_bytes = body

    def read(self) -> bytes:
        return self._body_bytes


class _FakeDispatcher:
    """Routes ``urlopen(Request)`` to a programmed handler keyed by URL/method.

    Tests subclass or instantiate this and program the handler closure.
    """
    def __init__(self, handler):
        self._handler = handler
        self.requests_seen: list = []
        # Token-safety: tests assert that the Authorization header is set
        # correctly but never log the token's value.

    def __call__(self, req, timeout=None, **kwargs):
        url = req.full_url
        method = req.get_method()
        body = None
        if req.data is not None:
            body = req.data.decode("utf-8") if isinstance(req.data, (bytes, bytearray)) else req.data
        # Capture headers minus Authorization.
        headers_sanitised = {k: v for k, v in req.headers.items() if k.lower() != "authorization"}
        # Confirm Authorization was present (test the contract):
        auth_header = req.get_header("Authorization")
        self.requests_seen.append({"url": url, "method": method, "body": body,
                                   "headers_sanitised": headers_sanitised,
                                   "_has_auth": auth_header is not None})

        return self._handler(url, method, body, auth_header)


def _patched_urlopen(handler):
    """Patch ``ionq_runtime.urllib.request.urlopen`` with ``handler``.

    The bridge calls ``urlopen(req, timeout=...)`` — mock passes those
    extra kwargs straight through to ``side_effect``, so we wrap the
    handler in a small adapter that strips them.
    """
    def _adapter(req, *args, **kwargs):
        url = req.full_url
        method = req.get_method()
        body = None
        if req.data is not None:
            body = req.data.decode("utf-8") if isinstance(req.data, (bytes, bytearray)) else req.data
        auth_header = req.get_header("Authorization")
        return handler(url, method, body, auth_header)
    return mock.patch.object(ionq_runtime.urllib.request, "urlopen", side_effect=_adapter)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

def _bell_graph() -> EQIRGraph:
    """Build a Bell-pair EQIR graph via the public `add_operation` API."""
    graph = EQIRGraph()
    graph.add_operation("ALLOC", targets=["q0"])
    graph.add_operation("ALLOC", targets=["q1"])
    graph.add_operation("GATE", gate_name="H", targets=["q0"])
    graph.add_operation("GATE", gate_name="CNOT", targets=["q0", "q1"])
    graph.add_operation("MEASURE", targets=["q0"], cbit_name="c0")
    graph.add_operation("MEASURE", targets=["q1"], cbit_name="c1")
    return graph


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #

class TestIonQSubmit(unittest.TestCase):
    def test_unsupported_gate_raises_before_network(self):
        graph = _bell_graph()
        # Replace CNOT with something IonQ doesn't have (e.g. CRX).
        for n in graph.nodes.values():
            if n.type == "GATE" and n.gate_name == "CNOT":
                n.gate_name = "CRX"
                n.args = [0.5]
                break

        with mock.patch.object(ionq_runtime.urllib.request, "urlopen") as u:
            with self.assertRaises(UnsupportedGateError) as ctx:
                submit_to_ionq(graph, api_token="abc123", backend="qpu")
            # No HTTP request was issued.
            self.assertFalse(u.called)
            # Mention the offending gate.
            self.assertIn("CRX", str(ctx.exception))

    def test_missing_token_rejected(self):
        graph = _bell_graph()
        with self.assertRaises(ValueError):
            submit_to_ionq(graph, api_token="", backend="qpu")
        with self.assertRaises(ValueError):
            submit_to_ionq(graph, api_token=None, backend="qpu")

    def test_invalid_backend_rejected(self):
        graph = _bell_graph()
        with self.assertRaises(ValueError):
            submit_to_ionq(graph, api_token="abc", backend="quantumcomputer")
        with self.assertRaises(ValueError):
            submit_to_ionq(graph, api_token="abc", backend="")

    def test_auth_failure_message_omits_token(self):
        """If IonQ returns 401, the IonQAPIError message must NOT include
        the token — even though we did send it in the Authorization header."""
        graph = _bell_graph()
        secret_token = "very_secret_token_DO_NOT_LEAK_123"

        def handler(url, method, body, auth_header):
            # Verify the token went OUT as Authorization.
            self.assertEqual(auth_header, f"api_key {secret_token}")
            # Now simulate IonQ's 401.
            raise ionq_runtime.urllib.error.HTTPError(
                url, 401, "Unauthorized", {},
                io.BytesIO(b'{"error": "invalid api key"}'),
            )

        with _patched_urlopen(handler):
            with self.assertRaises(IonQAPIError) as ctx:
                submit_to_ionq(graph, api_token=secret_token, backend="qpu")
        # The exception message must NOT include the token text.
        self.assertNotIn(secret_token, str(ctx.exception))
        self.assertEqual(ctx.exception.status, 401)

    def test_submit_success_and_wait(self):
        """Happy path: POST /jobs returns id, poll returns "ready", then
        GET /jobs/{id}/result returns probabilities, which we convert to
        integer counts."""
        graph = _bell_graph()
        token = "good-token-123"

        submit_response = {"id": "job-abc", "status": "submitted"}
        poll_responses = iter([
            {"id": "job-abc", "status": "ready"},
        ])
        result_response = {"probabilities": {"00": 0.5, "11": 0.5}}

        def handler(url, method, body, auth_header):
            if url.endswith("/jobs") and method == "POST":
                # Verify payload.
                parsed = json.loads(body)
                self.assertEqual(parsed["target"], "qpu")
                self.assertEqual(parsed["shots"], 1024)
                self.assertEqual(parsed["name"], "eigen-submitted")
                # Confirm circuit was translated correctly.
                gate_names = [op.get("gate") for op in parsed["circuit"]]
                self.assertIn("h", gate_names)
                self.assertIn("cnot", gate_names)
                self.assertIn("measure", gate_names)
                return _FakeResponse(200, submit_response)
            if url.endswith("/jobs/job-abc") and method == "GET":
                return _FakeResponse(200, next(poll_responses))
            if url.endswith("/jobs/job-abc/result") and method == "GET":
                return _FakeResponse(200, result_response)
            raise AssertionError(f"unexpected {method} {url}")

        with _patched_urlopen(handler):
            result = submit_to_ionq(
                graph, api_token=token, backend="qpu",
                poll_interval=0.0,
                timeout=5.0,
            )
        self.assertIsInstance(result, IonQJobResult)
        self.assertEqual(result.job_id, "job-abc")
        self.assertEqual(result.backend, "qpu")
        self.assertEqual(result.shots, 1024)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.counts, {"00": 512, "11": 512})

    def test_wait_false_returns_immediately(self):
        graph = _bell_graph()
        token = "good-token-123"
        submitted_response = {"id": "job-immediate", "status": "submitted"}

        urls_called: list[str] = []

        def handler(url, method, body, auth_header):
            urls_called.append(url)
            self.assertEqual(method, "POST")
            return _FakeResponse(200, submitted_response)

        with _patched_urlopen(handler):
            result = submit_to_ionq(
                graph, api_token=token, backend="qpu", wait=False,
            )
        self.assertEqual(result.job_id, "job-immediate")
        self.assertEqual(result.counts, {})
        self.assertEqual(result.status, "submitted")
        # Only one call should have happened.
        self.assertEqual(len(urls_called), 1)
        # Wait=False should not poll.
        self.assertTrue(urls_called[0].endswith("/jobs"))

    def test_timeout_returns_status_when_not_ready_in_time(self):
        graph = _bell_graph()
        token = "good-token-123"

        submission_count = [0]
        poll_count = [0]

        def handler(url, method, body, auth_header):
            if method == "POST":
                submission_count[0] += 1
                return _FakeResponse(200, {"id": "slow-job", "status": "submitted"})
            # Always return "running" — never reaches "ready".
            poll_count[0] += 1
            return _FakeResponse(200, {"id": "slow-job", "status": "running"})

        with _patched_urlopen(handler):
            result = submit_to_ionq(
                graph, api_token=token, backend="qpu",
                wait=True,
                timeout=0.5,
                poll_interval=0.05,
            )
        self.assertEqual(result.job_id, "slow-job")
        self.assertEqual(result.status, "running")
        # Counts must be empty since we never reached 'ready'.
        self.assertEqual(result.counts, {})
        self.assertTrue(poll_count[0] >= 1)
        # The warning should explain the failure mode.
        self.assertTrue(any("did not reach" in w for w in result.warnings))

    def test_failed_status_propagates_warning(self):
        """If the job ends in 'failed' rather than 'ready', we report that
        with a warning rather than raising."""
        graph = _bell_graph()
        token = "good-token-123"

        def handler(url, method, body, auth_header):
            if method == "POST":
                return _FakeResponse(200, {"id": "bad-job", "status": "submitted"})
            return _FakeResponse(200, {"id": "bad-job", "status": "failed",
                                       "error": "QPU crashed"})

        with _patched_urlopen(handler):
            result = submit_to_ionq(
                graph, api_token=token, backend="qpu",
                timeout=5.0, poll_interval=0.0,
            )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.counts, {})
        self.assertTrue(any("failed" in w for w in result.warnings))

    def test_token_never_in_logs_or_result_repr(self):
        """Even when a job succeeds, the token must not appear in log
        messages, the IonQJobResult repr, warnings, exception strings,
        or any attribute on the result object the bridge returns."""
        graph = _bell_graph()
        secret = "TOP-SECRET-123"

        log_buf = io.StringIO()
        handler = logging.StreamHandler(log_buf)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger("src.backend.backends.ionq_runtime")
        old_level = logger.level
        old_handlers = logger.handlers
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        def http_handler(url, method, body, auth_header):
            self.assertEqual(auth_header, f"api_key {secret}")
            if method == "POST":
                return _FakeResponse(200, {"id": "logtest", "status": "submitted"})
            if url.endswith("/logtest"):
                return _FakeResponse(200, {"id": "logtest", "status": "ready"})
            return _FakeResponse(200, {"probabilities": {"00": 1.0}})

        try:
            with _patched_urlopen(http_handler):
                result = submit_to_ionq(graph, api_token=secret, backend="qpu",
                                        poll_interval=0.0, timeout=5.0)
        finally:
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            handler.flush()
            log_buf.flush()

        log_text = log_buf.getvalue()
        self.assertNotIn(secret, log_text)
        self.assertNotIn(secret, repr(result))
        self.assertNotIn(secret, str(result))
        for w in result.warnings:
            self.assertNotIn(secret, w)

    def test_5xx_during_poll_retries(self):
        """If the poll endpoint returns 5xx, the bridge should retry
        once before crashing out — QPUs under heavy load frequently
        emit transient 503s."""
        graph = _bell_graph()
        token = "good-token-123"

        poll_attempts = [0]

        def handler(url, method, body, auth_header):
            if method == "POST":
                return _FakeResponse(200, {"id": "polyjob", "status": "submitted"})
            if url.endswith("/polyjob"):
                poll_attempts[0] += 1
                if poll_attempts[0] == 1:
                    # Simulate HTTPError 503.
                    raise ionq_runtime.urllib.error.HTTPError(
                        url, 503, "Service Unavailable", {},
                        io.BytesIO(b"backend overloaded"),
                    )
                return _FakeResponse(200, {"id": "polyjob", "status": "ready"})
            return _FakeResponse(200, {"probabilities": {"00": 1.0}})

        with _patched_urlopen(handler):
            result = submit_to_ionq(graph, api_token=token, backend="qpu",
                                     poll_interval=0.0, timeout=30.0)
        self.assertEqual(result.status, "ready")
        self.assertEqual(poll_attempts[0], 2)  # one retry, one success.

    def test_simulator_backend_provided_in_payload(self):
        graph = _bell_graph()
        submitted = {"id": "sim-job", "status": "submitted"}
        captured: dict = {}

        def handler(url, method, body, auth_header):
            if method == "POST":
                captured["body"] = json.loads(body)
                return _FakeResponse(200, submitted)
            if url.endswith("/sim-job"):
                return _FakeResponse(200, {"id": "sim-job", "status": "ready"})
            return _FakeResponse(200, {"probabilities": {"00": 1.0}})

        with _patched_urlopen(handler):
            result = submit_to_ionq(
                graph, api_token="tok",
                backend="simulator",
                shots=2048,
                poll_interval=0.0,
                timeout=30.0,
            )
        self.assertEqual(captured["body"]["target"], "simulator")
        self.assertEqual(captured["body"]["shots"], 2048)
        self.assertEqual(result.backend, "simulator")

    def test_counts_extraction_legacy_histogram_format(self):
        """Older IonQ payloads nest under `metadata.histogram`. Test that
        the extractor handles both."""
        result_dict = {"metadata": {"histogram": {"0": 0.7, "1": 0.3}}}
        counts = ionq_runtime._extract_counts(result_dict, shots=1000)
        self.assertEqual(counts, {"0": 700, "1": 300})


if __name__ == "__main__":
    unittest.main()
