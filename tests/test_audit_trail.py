"""
P2 §6.4 — Audit Trail tests.

The roadmap (`sol.md` §6.4 "Воспроизводимость") lists audit trail as one
of the four reproducibility tools. We implement:

  * An `AuditEntry` dataclass with stable, JSON-serialisable fields.
  * An `AuditTrail` class that buffers entries in-memory and persists
    them to JSONL when configured with a path. The trail is
    append-only — each ``record()`` writes one line, so partial
    writes don't corrupt earlier records.
  * A module-level singleton (`get_default_trail()`) with explicit
    `reset_default_trail()` so tests can isolate.
  * Optional integration with `EigenVM.execute(audit=...)` that wraps
    the existing `_execute_locked` in a try/finally, measuring the
    wall-clock duration and recording the outcome+error into the
    trail. The audit is opt-in (off by default) so existing tests
    aren't burdened with audit log writes.

Tests cover:
    - AuditEntry field defaults and JSONL serialisation.
    - AuditTrail in-memory vs persisted behaviour.
    - Append-only semantics on the JSONL file (no truncation).
    - Failure path: when the on-disk write fails (read-only file
      system, broken pipe), the audited run still completes and the
      `_failed_writes` counter increments.
    - Singleton get/set/reset isolation.
    - EigenVM.execute(audit=...) integration: success and failure both
      record their outcome with the right error type/message and a
      non-zero wall-clock duration.
"""
import json
import os
import platform
import stat
import sys
import tempfile
import unittest

from src.runtime_audit import (
    AuditEntry,
    AuditTrail,
    hash_program,
    fingerprint_result,
    get_default_trail,
    set_default_trail,
    reset_default_trail,
)
from src.backend.vm import EigenVM


# Use the same temp-dir + EBC compile pattern as test_vm_threadsafe.py:
# the EBC compiler needs a real filename for source locations, so we
# write each source to a content-hashed path before handing off to
# `to_ebc`.
_WORKSPACE = tempfile.mkdtemp(prefix="eigen_audit_trail_")


def _compile_to_ebc(src, filename="__audit_inline__.eig"):
    import hashlib
    content_hash = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
    path = os.path.join(_WORKSPACE, f"audit_{content_hash}_{filename}")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
    from src.compiler import to_ebc
    return to_ebc(path, _WORKSPACE, optimize=False)


_OK_SRC = """eigen 1.0
func main() -> int {
    return 1
}
let result: int = main()
"""


_RUNTIME_ERR_SRC = """eigen 1.0
func boom() -> int {
    return 1 / 0
}
let x: int = boom()
"""


class TestAuditEntry(unittest.TestCase):

    def test_default_fields(self):
        e = AuditEntry(
            program_hash="abc",
            started_at_ns=1,
            wall_clock_ns=2,
            outcome="success",
        )
        self.assertEqual(e.error_type, None)
        self.assertEqual(e.error_message, None)
        self.assertFalse(e.deterministic)
        self.assertEqual(e.extra, {})
        self.assertEqual(e.hostname, platform.node())

    def test_to_jsonl_roundtrip(self):
        e = AuditEntry(
            program_hash="abc",
            started_at_ns=10_000,
            wall_clock_ns=42,
            outcome="success",
            seed=7,
            sim_type="dense",
            extra={"cli": "eigen run --audit"},
        )
        line = e.to_jsonl()
        # JSONL = single newline-free line.
        self.assertNotIn("\n", line)
        obj = json.loads(line)
        self.assertEqual(obj["program_hash"], "abc")
        self.assertEqual(obj["outcome"], "success")
        self.assertEqual(obj["extra"], {"cli": "eigen run --audit"})

    def test_to_jsonl_handles_non_json_extra_values(self):
        # Path / non-JSON primitive in extra should be tolerated by
        # default=str fallback rather than raising.
        e = AuditEntry(
            program_hash="x",
            started_at_ns=0,
            wall_clock_ns=0,
            outcome="success",
            extra={"path": "D:\\Nuras-7"},
        )
        line = e.to_jsonl()
        obj = json.loads(line)
        self.assertEqual(obj["extra"]["path"], "D:\\Nuras-7")


class TestHashProgram(unittest.TestCase):

    def test_string_input(self):
        h = hash_program("hello")
        self.assertEqual(len(h), 64)
        # Same input → same output.
        self.assertEqual(h, hash_program("hello"))

    def test_string_vs_bytes_equivalence(self):
        self.assertEqual(hash_program("abc"),
                         hash_program(b"abc"))

    def test_different_inputs_differ(self):
        self.assertNotEqual(hash_program("a"), hash_program("b"))


class TestFingerprintResult(unittest.TestCase):

    def test_dict_order_doesnt_matter(self):
        # JSON sort_keys=True ensures dict key ordering doesn't affect
        # the fingerprint — important for reproducibility assertions
        # across Python impl.
        a = fingerprint_result({"x": 1, "y": 2})
        b = fingerprint_result({"y": 2, "x": 1})
        self.assertEqual(a, b)

    def test_non_json_values_survive_via_default_str(self):
        # `pathlib.Path` is non-JSON; default=str lets it through.
        from pathlib import Path
        fp = fingerprint_result({"p": Path("/tmp")})
        self.assertEqual(len(fp), 64)


class TestAuditTrailInMemory(unittest.TestCase):

    def test_disabled_trail_does_not_buffer_or_write(self):
        trail = AuditTrail(path=None, enabled=False)
        e = trail.record("abc")
        self.assertEqual(len(trail.entries()), 0)
        # The return value is still a fully-formed AuditEntry even
        # when disabled, so the caller can inspect it if needed.
        self.assertEqual(e.program_hash, "abc")
        self.assertEqual(e.outcome, "success")

    def test_enabled_trail_with_no_path_buffers_in_memory(self):
        trail = AuditTrail(path=None, enabled=True)
        trail.record("h1")
        trail.record("h2")
        self.assertEqual([e.program_hash for e in trail.entries()],
                         ["h1", "h2"])

    def test_buffer_evicts_oldest_beyond_cap(self):
        trail = AuditTrail(path=None, enabled=True, max_buffer=2)
        trail.record("h1")
        trail.record("h2")
        trail.record("h3")
        # h1 evicted.
        self.assertEqual([e.program_hash for e in trail.entries()],
                         ["h2", "h3"])

    def test_failure_outcome_inferred_from_error_arg(self):
        trail = AuditTrail(path=None, enabled=True)
        try:
            raise ValueError("boom")
        except ValueError as err:
            e = trail.record("h", error=err)
        self.assertEqual(e.outcome, "failure")
        self.assertEqual(e.error_type, "ValueError")
        self.assertEqual(e.error_message, "boom")

    def test_explicit_failure_outcome_kept(self):
        trail = AuditTrail(path=None, enabled=True)
        e = trail.record("h", outcome="failure", error=ValueError("x"))
        self.assertEqual(e.outcome, "failure")


class TestAuditTrailPersisted(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.audit_path = os.path.join(self.tmpdir, "audit.jsonl")
        self.trail = AuditTrail(path=self.audit_path, enabled=True)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_appends_one_line(self):
        self.trail.record("h1")
        self.trail.record("h2")
        self.assertTrue(os.path.exists(self.audit_path))
        with open(self.audit_path) as f:
            lines = f.read().strip().split("\n")
        self.assertEqual(len(lines), 2)
        objs = [json.loads(l) for l in lines]
        self.assertEqual([o["program_hash"] for o in objs],
                         ["h1", "h2"])

    def test_append_only_does_not_truncate_existing_file(self):
        # Manually seed the file with a non-trail record (this would
        # happen if external tooling appends a comment or marker line).
        # AuditTrail must not destroy it.
        with open(self.audit_path, "w") as f:
            f.write("{\"marker\": \"pre-trail\"}\n")
        self.trail.record("h1")
        with open(self.audit_path) as f:
            content = f.read()
        # The original marker line + our trail line both present.
        self.assertIn("pre-trail", content)
        self.assertIn("\"h1\"", content)

    def test_read_jsonl_returns_list_of_dicts(self):
        self.trail.record("h1", seed=7)
        self.trail.record("h2", seed=8)
        out = self.trail.read_jsonl()
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["program_hash"], "h1")
        self.assertEqual(out[0]["seed"], 7)

    def test_read_jsonl_returns_empty_when_no_path(self):
        trail = AuditTrail(path=None, enabled=True)
        self.assertEqual(trail.read_jsonl(), [])

    def test_read_jsonl_skips_malformed_lines(self):
        with open(self.audit_path, "w") as f:
            f.write(json.dumps({"program_hash": "h1",
                                "started_at_ns": 0,
                                "wall_clock_ns": 0,
                                "outcome": "success"}) + "\n")
            f.write("not json at all\n")
            f.write(json.dumps({"program_hash": "h2",
                                "started_at_ns": 0,
                                "wall_clock_ns": 0,
                                "outcome": "success"}) + "\n")
        out = self.trail.read_jsonl()
        # Malformed middle line skipped silently.
        self.assertEqual([o["program_hash"] for o in out], ["h1", "h2"])
        # Failure counter bumped by the unparseable line.
        self.assertGreaterEqual(self.trail._failed_writes, 1)

    def test_clear_drops_buffer_and_file(self):
        self.trail.record("h1")
        self.trail.clear()
        self.assertEqual(self.trail.entries(), [])
        self.assertFalse(os.path.exists(self.audit_path))

    @unittest.skipUnless(sys.platform.startswith("linux") or
                         sys.platform == "darwin",
                         "read-only fs test requires POSIX chmod")
    def test_unwritable_path_increments_failed_writes(self):
        # chmod the directory to read-only — the next record call
        # shouldn't raise but should increment _failed_writes.
        os.chmod(self.tmpdir, stat.S_IRUSR | stat.S_IXUSR)
        try:
            self.trail.record("h1")
        finally:
            os.chmod(self.tmpdir, stat.S_IRWXU)
        self.assertGreater(self.trail._failed_writes, 0)


class TestAuditTrailSingleton(unittest.TestCase):

    def setUp(self):
        reset_default_trail()

    def tearDown(self):
        reset_default_trail()

    def test_default_trail_is_in_memory_by_default(self):
        t = get_default_trail()
        self.assertIsNone(t.path)
        self.assertTrue(t.enabled)

    def test_singleton_is_stable_until_reset(self):
        t1 = get_default_trail()
        t2 = get_default_trail()
        self.assertIs(t1, t2)

    def test_set_default_trail_replaces_singleton(self):
        custom = AuditTrail(path=None, enabled=True)
        set_default_trail(custom)
        self.assertIs(get_default_trail(), custom)

    def test_reset_unsets_singleton(self):
        t1 = get_default_trail()
        reset_default_trail()
        t2 = get_default_trail()
        self.assertIsNot(t1, t2)


class TestEigenVMExecuteAudited(unittest.TestCase):

    def test_audited_success_records_entry(self):
        trail = AuditTrail(path=None, enabled=True)
        vm = EigenVM(sim_type='dense', seed=42, deterministic=True)
        prog = _compile_to_ebc(_OK_SRC)
        vm.execute(prog, audit=trail, program_hash="h-success")
        # No exception; VM returns whatever _execute_locked returns
        # (a stack of locals). We don't pin the shape — only that
        # audit captured it.
        entries = trail.entries()
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e.program_hash, "h-success")
        self.assertEqual(e.outcome, "success")
        self.assertEqual(e.seed, 42)
        self.assertTrue(e.deterministic)
        self.assertEqual(e.sim_type, "dense")
        self.assertGreater(e.wall_clock_ns, 0)
        self.assertEqual(e.num_instructions, len(prog))

    def test_audited_failure_records_exception(self):
        trail = AuditTrail(path=None, enabled=True)
        vm = EigenVM(sim_type='dense', seed=42)
        prog = _compile_to_ebc(_RUNTIME_ERR_SRC)
        # Divide-by-zero raises inside the VM (some path). We don't pin
        # the exact exception class — we only assert that outcome=
        # 'failure', error_type/message are populated, AND the audit
        # log was actually written even though the failure propagated
        # out of execute().
        with self.assertRaises(Exception):
            vm.execute(prog, audit=trail,
                       program_hash="h-fail")
        entries = trail.entries()
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e.outcome, "failure")
        self.assertIsNotNone(e.error_type)
        self.assertEqual(e.program_hash, "h-fail")

    def test_unaudited_path_is_identical_to_before(self):
        # Without audit=, the existing code path runs (return value
        # from _execute_locked). No trail entry should be recorded.
        trail = AuditTrail(path=None, enabled=True)
        vm = EigenVM(sim_type='dense', seed=1)
        prog = _compile_to_ebc(_OK_SRC)
        vm.execute(prog)
        self.assertEqual(len(trail.entries()), 0)

    def test_default_program_hash_when_not_provided(self):
        # When the caller passes audit= but doesn't supply
        # program_hash=, the VM computes a fallback hash from the
        # instruction reprs so consecutive runs of the same program
        # still group together in the audit log.
        trail = AuditTrail(path=None, enabled=True)
        vm = EigenVM(sim_type='dense', seed=1, deterministic=True)
        prog = _compile_to_ebc(_OK_SRC)
        vm.execute(prog, audit=trail)
        vm.execute(prog, audit=trail)
        entries = trail.entries()
        self.assertEqual(len(entries), 2)
        # Same canonical instructions → same fallback hash.
        self.assertEqual(entries[0].program_hash,
                         entries[1].program_hash)


if __name__ == "__main__":
    unittest.main()
