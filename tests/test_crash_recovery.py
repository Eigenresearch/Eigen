"""
Tests for src/backend/crash_recovery.py — sol.md §7.2 Crash Recovery.
"""
import dataclasses
import json
import os
import platform
import shutil
import tempfile
import unittest

from src.backend.bytecode import Instruction, Opcode
from src.backend.crash_recovery import (
    CrashReport,
    CrashReportBuilder,
    _safe_repr,
    _operand_stack_top,
    _call_trace,
    _locals_snapshot,
    _globals_snapshot,
    _last_instruction_repr,
    _compute_crash_id,
    _reproduction_hint,
    serialize_crash_report,
    dump_crash_report,
    load_crash_report,
)
from src.backend.vm import EigenVM, ActivationFrame


class _FakeClock:
    """Deterministic clock for tests — returns a monotonically
    increasing float sequence."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self):
        v = self.t
        self.t += 1.0
        return v


def _make_vm_with_state(*, ip: int = 5, instruction_count: int = 12,
                         dispatch_mode: str = "fast",
                         locals_map=None, globals_map=None,
                         operand_stack=None, func_name: str = "main",
                         return_address: int = 3) -> EigenVM:
    """Construct a VM seeded with synthetic state for crash-recovery
    testing without forcing full bytecode execution."""
    vm = EigenVM(dispatch_mode=dispatch_mode)
    vm.ip = ip
    vm.instruction_count = instruction_count
    vm.operand_stack = list(operand_stack or [])
    vm.globals = dict(globals_map or {})
    frame = vm.get_frame(return_address, func_name)
    frame.locals = dict(locals_map or {})
    vm.call_stack = [frame]
    return vm


class TestCrashReportBuilder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="eigen_crash_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_from_vm_captures_all_fields(self):
        vm = _make_vm_with_state(
            ip=3, instruction_count=10,
            locals_map={"x": 42, "y": "hello"},
            operand_stack=[1, 2, {"nested": [1, 2, 3]}],
        )
        instructions = [
            Instruction(Opcode.LOAD_CONST, 1, line=1),
            Instruction(Opcode.LOAD_VAR, "x", line=2),
            Instruction(Opcode.ADD, None, line=2),
        ]
        exc = ValueError("test crash")
        builder = CrashReportBuilder(clock=_FakeClock())
        report = builder.build_from_vm(vm, exc, instructions=instructions)
        self.assertEqual(report.exception_type, "ValueError")
        self.assertEqual(report.exception_message, "test crash")
        self.assertIn("ValueError", report.exception_repr)
        self.assertEqual(report.vm_ip, 3)
        self.assertEqual(report.vm_instruction_count, 10)
        self.assertEqual(report.vm_dispatch_mode, "fast")
        # 3 items in stack, all should be repr'd
        self.assertEqual(len(report.operand_stack_top), 3)
        # One frame (main)
        self.assertEqual(len(report.call_trace), 1)
        self.assertEqual(report.call_trace[0]["function_name"], "main")
        self.assertEqual(report.call_trace[0]["return_address"], 3)
        self.assertEqual(len(report.locals_snapshot), 1)
        self.assertEqual(report.locals_snapshot[0]["function_name"], "main")
        self.assertIn("x", report.locals_snapshot[0]["locals"])
        # Last instruction should be at index max(0, ip-1) = 2 → ADD
        self.assertIn("ADD", report.last_instruction_repr)

    def test_crash_id_is_deterministic_across_rebuilds(self):
        vm_kwargs = dict(
            ip=2, instruction_count=4,
            locals_map={"a": 1},
            operand_stack=[1, 2],
        )
        vm1 = _make_vm_with_state(**vm_kwargs)
        vm2 = _make_vm_with_state(**vm_kwargs)
        exc = RuntimeError("boom")
        instructions = [Instruction(Opcode.LOAD_CONST, 1),
                         Instruction(Opcode.LOAD_CONST, 2),
                         Instruction(Opcode.ADD)]
        builder = CrashReportBuilder(clock=_FakeClock())
        r1 = builder.build_from_vm(vm1, exc, instructions=instructions)
        r2 = builder.build_from_vm(vm2, exc, instructions=instructions)
        self.assertEqual(r1.crash_id, r2.crash_id)
        # Make sure it's a 64-char SHA-256 hex
        self.assertEqual(len(r1.crash_id), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in r1.crash_id))

    def test_crash_id_changes_when_locals_change(self):
        vm1 = _make_vm_with_state(locals_map={"x": 1})
        vm2 = _make_vm_with_state(locals_map={"x": 999})
        exc = ValueError("test")
        builder = CrashReportBuilder(clock=_FakeClock())
        r1 = builder.build_from_vm(vm1, exc,
                                    instructions=[Instruction(Opcode.ADD)])
        r2 = builder.build_from_vm(vm2, exc,
                                    instructions=[Instruction(Opcode.ADD)])
        self.assertNotEqual(r1.crash_id, r2.crash_id)

    def test_crash_id_changes_when_dispatch_mode_changes(self):
        vm1 = _make_vm_with_state(dispatch_mode="fast")
        vm2 = _make_vm_with_state(dispatch_mode="table")
        exc = ValueError("test")
        builder = CrashReportBuilder(clock=_FakeClock())
        instructions = [Instruction(Opcode.ADD)]
        r1 = builder.build_from_vm(vm1, exc, instructions=instructions)
        r2 = builder.build_from_vm(vm2, exc, instructions=instructions)
        self.assertNotEqual(r1.crash_id, r2.crash_id)

    def test_crash_id_stable_when_timestamp_and_hostname_vary(self):
        """Two crashes in different environments but same canonical
        state should produce the same crash-id (deduplication is the
        point of the deterministic id)."""
        vm = _make_vm_with_state(locals_map={"x": 1})
        exc = ValueError("test")
        instructions = [Instruction(Opcode.ADD)]
        b1 = CrashReportBuilder(clock=_FakeClock(start=1.0))
        b2 = CrashReportBuilder(clock=_FakeClock(start=9999.0))
        r1 = b1.build_from_vm(vm, exc, instructions=instructions)
        r2 = b2.build_from_vm(vm, exc, instructions=instructions)
        self.assertEqual(r1.crash_id, r2.crash_id)
        self.assertNotEqual(r1.timestamp_ns, r2.timestamp_ns)

    def test_write_persists_file_to_disk(self):
        vm = _make_vm_with_state(locals_map={"x": 1})
        exc = ValueError("test")
        builder = CrashReportBuilder(clock=_FakeClock(),
                                       crash_report_dir=self.tmp)
        report, path = builder.build_and_write(vm, exc,
                                                 instructions=[Instruction(Opcode.ADD)])
        self.assertTrue(os.path.isfile(path))
        self.assertIn(self.tmp, path)
        self.assertIn(report.crash_id, os.path.basename(path))
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        self.assertEqual(blob["crash_id"], report.crash_id)
        self.assertEqual(blob["exception_type"], "ValueError")

    def test_write_path_explicit_overrides_dir(self):
        vm = _make_vm_with_state()
        exc = ValueError("test")
        builder = CrashReportBuilder(clock=_FakeClock(),
                                       crash_report_dir=self.tmp)
        explicit = os.path.join(self.tmp, "custom.json")
        report = builder.build_from_vm(vm, exc,
                                        instructions=[Instruction(Opcode.ADD)])
        out = builder.write(report, path=explicit)
        self.assertEqual(out, explicit)
        self.assertTrue(os.path.isfile(explicit))

    def test_write_raises_if_no_dir_and_no_path(self):
        vm = _make_vm_with_state()
        exc = ValueError("test")
        builder = CrashReportBuilder(clock=_FakeClock())  # no dir
        report = builder.build_from_vm(vm, exc,
                                        instructions=[Instruction(Opcode.ADD)])
        with self.assertRaises(RuntimeError):
            builder.write(report)

    def test_build_and_write_round_trip(self):
        vm = _make_vm_with_state(locals_map={"x": 1, "y": "ok"})
        exc = RuntimeError("rt")
        builder = CrashReportBuilder(clock=_FakeClock(),
                                       crash_report_dir=self.tmp)
        report, path = builder.build_and_write(vm, exc,
                                                 instructions=[Instruction(Opcode.LOAD_CONST, 42)])
        loaded = load_crash_report(path)
        self.assertEqual(loaded["crash_id"], report.crash_id)
        self.assertEqual(loaded["exception_type"], "RuntimeError")
        self.assertEqual(loaded["vm_dispatch_mode"], "fast")

    def test_serialize_and_dump_helpers(self):
        vm = _make_vm_with_state()
        exc = ValueError("x")
        builder = CrashReportBuilder(clock=_FakeClock())
        report = builder.build_from_vm(vm, exc,
                                        instructions=[Instruction(Opcode.ADD)])
        d = serialize_crash_report(report)
        self.assertIn("crash_id", d)
        self.assertIn("exception_type", d)
        path = os.path.join(self.tmp, "out.json")
        dump_crash_report(report, path)
        with open(path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["crash_id"], report.crash_id)


class TestSafeReprAndSnapshots(unittest.TestCase):
    def test_safe_repr_truncates_long_strings(self):
        long = "x" * 1024
        s = _safe_repr(long, n=20)
        self.assertLessEqual(len(s), 20 + len("...(truncated)"))
        self.assertIn("truncated", s)

    def test_safe_repr_handles_unreprable(self):
        class Bad:
            def __repr__(self):
                raise RuntimeError("boom")

        s = _safe_repr(Bad())
        self.assertIn("unrepr-able", s)

    def test_safe_repr_default_limit(self):
        s = _safe_repr("short")
        self.assertEqual(s, "'short'")

    def test_operand_stack_top_limit(self):
        big = list(range(1000))
        snap = _operand_stack_top(big)
        # Should be capped at _OPERAND_STACK_SNAPSHOT_LIMIT (=16)
        self.assertEqual(len(snap), 16)
        # Should be the LAST 16 (top of stack)
        self.assertEqual(snap[-1], "999")
        self.assertEqual(snap[0], "984")

    def test_operand_stack_top_empty(self):
        self.assertEqual(_operand_stack_top([]), [])

    def test_call_trace_walks_innermost_first(self):
        f1 = ActivationFrame(0, "main")
        f1.locals = {"global_x": 1}
        f2 = ActivationFrame(5, "helper")
        f2.locals = {"local_y": 2}
        f3 = ActivationFrame(10, "innermost")
        f3.locals = {"z": 3}
        trace = _call_trace([f1, f2, f3])
        # Reversed so innermost is first
        self.assertEqual([t["function_name"] for t in trace],
                          ["innermost", "helper", "main"])
        self.assertEqual(trace[0]["locals_count"], 1)

    def test_call_trace_empty(self):
        self.assertEqual(_call_trace([]), [])

    def test_locals_snapshot_includes_function(self):
        f1 = ActivationFrame(0, "main")
        f1.locals = {"x": 1, "big": [1, 2, 3]}
        snap = _locals_snapshot([f1])
        self.assertEqual(snap[0]["function_name"], "main")
        self.assertIn("x", snap[0]["locals"])
        self.assertIn("big", snap[0]["locals"])

    def test_locals_snapshot_handles_empty_locals(self):
        f1 = ActivationFrame(0, "main")
        # locals is {} by default
        snap = _locals_snapshot([f1])
        self.assertEqual(snap[0]["locals"], {})

    def test_globals_snapshot_handles_non_dict(self):
        self.assertEqual(_globals_snapshot(None), {})
        self.assertEqual(_globals_snapshot([]), {})
        self.assertEqual(_globals_snapshot(42), {})

    def test_globals_snapshot_reprs_values(self):
        snap = _globals_snapshot({"a": 1, "b": "hello"})
        self.assertEqual(snap["a"], "1")
        self.assertEqual(snap["b"], "'hello'")

    def test_last_instruction_repr_empty(self):
        self.assertEqual(_last_instruction_repr([], 0),
                         "<empty instruction list>")

    def test_last_instruction_repr_at_ip(self):
        insts = [
            Instruction(Opcode.LOAD_CONST, 1, line=1),
            Instruction(Opcode.LOAD_CONST, 2, line=2),
            Instruction(Opcode.ADD, None, line=2),
            Instruction(Opcode.STORE_VAR, "x", line=2),
        ]
        # ip=3 → index 2 → ADD (last executed before crash)
        s = _last_instruction_repr(insts, 3)
        self.assertIn("ADD", s)
        self.assertIn("@2", s)

    def test_last_instruction_repr_clamps_ip(self):
        insts = [Instruction(Opcode.LOAD_CONST, 1)]
        # ip=999 → clamp to index 0
        s = _last_instruction_repr(insts, 999)
        self.assertIn("@0", s)
        self.assertIn("LOAD_CONST", s)

    def test_compute_crash_id_is_sha256_hex(self):
        payload = {
            "exception_type": "E",
            "exception_message": "M",
            "vm_dispatch_mode": "fast",
            "last_instruction_repr": "ir",
            "call_trace_repr": "[]",
            "locals_repr": "{}",
            "globals_repr": "{}",
        }
        h = _compute_crash_id(payload)
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_reproduction_hint_mentions_dispatch_mode(self):
        hint = _reproduction_hint("ValueError", "table")
        self.assertIn("table", hint)
        self.assertIn("ValueError", hint)

    def test_reproduction_hint_mentions_fast(self):
        hint = _reproduction_hint("RuntimeError", "fast")
        self.assertIn("fast", hint)


class TestCrashReportSerialization(unittest.TestCase):
    def test_to_json_round_trip(self):
        vm = _make_vm_with_state(locals_map={"x": 1})
        exc = ValueError("x")
        builder = CrashReportBuilder(clock=_FakeClock())
        report = builder.build_from_vm(vm, exc,
                                        instructions=[Instruction(Opcode.ADD)])
        s = report.to_json()
        loaded = json.loads(s)
        self.assertEqual(loaded["crash_id"], report.crash_id)
        self.assertEqual(loaded["vm_dispatch_mode"], "fast")

    def test_to_dict_returns_dataclass_fields(self):
        import dataclasses
        vm = _make_vm_with_state()
        exc = ValueError("x")
        builder = CrashReportBuilder(clock=_FakeClock())
        report = builder.build_from_vm(vm, exc,
                                        instructions=[])
        d = report.to_dict()
        expected_keys = {f.name for f in dataclasses.fields(CrashReport)}
        self.assertEqual(set(d.keys()), expected_keys)

    def test_report_is_frozen(self):
        vm = _make_vm_with_state()
        exc = ValueError("x")
        builder = CrashReportBuilder(clock=_FakeClock())
        report = builder.build_from_vm(vm, exc,
                                        instructions=[])
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            # frozen dataclass raises FrozenInstanceError (3.10+)
            # or AttributeError on earlier
            report.vm_ip = 999

    def test_json_serializes_with_numpy_fallback(self):
        """CrashReport.to_json should not crash when a snapshot
        contains non-JSON-native values. (We already pre-repr
        everything, but pass an arbitrary object via `globals`
        to exercise the `default=str` fallback path of json.dumps.)"""
        # globals_snapshot pre-reprs to strings; we want to ensure
        # the JSON dump handles arbitrary non-string values via
        # the `default=str` fallback. Use a hand-crafted dict.
        obj = object()  # not JSON-serializable
        builder = CrashReportBuilder(clock=_FakeClock())
        vm = _make_vm_with_state(globals_map={"weird": obj})
        exc = RuntimeError("rt")
        report = builder.build_from_vm(vm, exc,
                                        instructions=[Instruction(Opcode.ADD)])
        # Should not raise
        s = report.to_json()
        self.assertIn("weird", s)


class TestCrashReportWithTableDispatch(unittest.TestCase):
    def test_table_dispatch_mode_recorded(self):
        vm = _make_vm_with_state(dispatch_mode="table")
        exc = RuntimeError("rt")
        builder = CrashReportBuilder(clock=_FakeClock())
        report = builder.build_from_vm(vm, exc,
                                        instructions=[Instruction(Opcode.ADD)])
        self.assertEqual(report.vm_dispatch_mode, "table")


class TestCrashReportWithoutInstructions(unittest.TestCase):
    def test_build_with_no_instructions_argument_uses_vm_attribute(self):
        vm = _make_vm_with_state()
        vm.instructions = [Instruction(Opcode.LOAD_CONST, 1),
                            Instruction(Opcode.LOAD_VAR, "x")]
        exc = ValueError("x")
        builder = CrashReportBuilder(clock=_FakeClock())
        # No instructions passed → falls back to vm.instructions
        report = builder.build_from_vm(vm, exc)
        self.assertIn("LOAD_VAR", report.last_instruction_repr)


class TestCrashReportWithNestedCallStack(unittest.TestCase):
    def test_nested_frames_all_captured(self):
        vm = EigenVM()
        vm.ip = 7
        vm.instruction_count = 20
        vm.operand_stack = [42, "hello"]
        vm.globals = {"global_a": 100}
        # Build two frames
        f_outer = vm.get_frame(0, "main")
        f_outer.locals = {"a": 1}
        f_inner = vm.get_frame(5, "compute_sum")
        f_inner.locals = {"acc": 42, "i": 3, "big_list": list(range(200))}
        vm.call_stack = [f_outer, f_inner]
        exc = RuntimeError("nested crash")
        builder = CrashReportBuilder(clock=_FakeClock())
        instructions = [Instruction(Opcode.LOAD_CONST, i) for i in range(10)]
        instructions.append(Instruction(Opcode.ADD))
        report = builder.build_from_vm(vm, exc, instructions=instructions)
        # Inner first, outer last
        self.assertEqual([t["function_name"] for t in report.call_trace],
                          ["compute_sum", "main"])
        self.assertEqual(len(report.locals_snapshot), 2)
        self.assertEqual(report.locals_snapshot[0]["function_name"],
                         "compute_sum")
        self.assertEqual(report.locals_snapshot[1]["function_name"], "main")
        # Inner-frame locals truncated cleanly
        big_list_repr = report.locals_snapshot[0]["locals"]["big_list"]
        self.assertIn("truncated", big_list_repr)


class TestCrashReportAutoFileNaming(unittest.TestCase):
    def test_filename_is_crash_id_json(self):
        tmp = tempfile.mkdtemp(prefix="eigen_crash_")
        try:
            builder = CrashReportBuilder(clock=_FakeClock(),
                                           crash_report_dir=tmp)
            vm = _make_vm_with_state()
            exc = ValueError("x")
            _, path = builder.build_and_write(vm, exc,
                                                instructions=[Instruction(Opcode.ADD)])
            self.assertEqual(os.path.basename(path),
                             f"{CrashReportBuilder(clock=_FakeClock())
                                .build_from_vm(vm, exc,
                                  instructions=[Instruction(Opcode.ADD)]).crash_id}.json")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
