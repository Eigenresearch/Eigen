import os
import tempfile
import unittest
from unittest import mock
from src.crash_report import write_crash_report


class _FakeFrame:
    def __init__(self, func_name="test_func", line=42, locals_map=None):
        self.func_name = func_name
        self.current_line = line
        self.locals = locals_map if locals_map is not None else {"x": 1, "y": "abc"}


class TestCrashReportBasics(unittest.TestCase):
    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmpdir = tempfile.mkdtemp(prefix="eigen_crash_test_")
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)

    def test_writes_file_with_timestamp_name(self):
        error = RuntimeError("test error")
        frame = _FakeFrame()
        write_crash_report(error, [frame], 42, "ADD", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".log"))
        self.assertRegex(files[0], r"crash-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log")

    def test_report_contains_error_message(self):
        error = RuntimeError("specific error message")
        write_crash_report(error, [], 0, "NOP", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("specific error message", content)
        self.assertIn("Error Message:", content)

    def test_report_contains_ip(self):
        write_crash_report(ValueError("e"), [], 99, "MUL", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("99", content)
        self.assertIn("MUL", content)

    def test_report_contains_call_stack(self):
        frame1 = _FakeFrame(func_name="outer", line=10)
        frame2 = _FakeFrame(func_name="inner", line=20)
        write_crash_report(RuntimeError("e"), [frame1, frame2], 0, "NOP", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("outer", content)
        self.assertIn("inner", content)
        self.assertIn("line 10", content)
        self.assertIn("line 20", content)

    def test_report_contains_globals(self):
        globals_map = {"g_var": 42, "name": "eigen"}
        write_crash_report(RuntimeError("e"), [], 0, "NOP", globals_map)
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("g_var", content)
        self.assertIn("eigen", content)

    def test_report_contains_traceback(self):
        try:
            raise ValueError("test tb error")
        except ValueError as e:
            err = e
        write_crash_report(err, [], 0, "NOP", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Traceback", content)
        self.assertIn("test tb error", content)
        self.assertIn("ValueError", content)


class TestCrashReportFallback(unittest.TestCase):
    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmpdir = tempfile.mkdtemp(prefix="eigen_crash_test_")
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)

    def test_permission_error_falls_back_to_tempdir(self):
        with mock.patch("os.getcwd", side_effect=PermissionError("denied")):
            error = RuntimeError("test error")
            write_crash_report(error, [], 0, "NOP", {})
        temp_files = [f for f in os.listdir(tempfile.gettempdir()) if f.startswith("crash-")]
        self.assertGreaterEqual(len(temp_files), 1)

    def test_open_error_falls_back_to_tempdir(self):
        orig_open = open
        calls = {"count": 0}

        def _denied_open(path, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise PermissionError("denied first attempt")
            return orig_open(path, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=_denied_open):
            error = RuntimeError("test error")
            write_crash_report(error, [], 0, "NOP", {})

        temp_files = [f for f in os.listdir(tempfile.gettempdir()) if f.startswith("crash-")]
        self.assertGreaterEqual(len(temp_files), 1)


class TestCrashReportMultiple(unittest.TestCase):
    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmpdir = tempfile.mkdtemp(prefix="eigen_crash_test_")
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)

    def test_multiple_crashes_produce_distinct_files(self):
        import time

        write_crash_report(RuntimeError("e1"), [], 0, "NOP", {})
        time.sleep(1.05)
        write_crash_report(RuntimeError("e2"), [], 0, "NOP", {})
        files = sorted(f for f in os.listdir(self._tmpdir) if f.startswith("crash-"))
        self.assertEqual(len(files), 2)
        self.assertNotEqual(files[0], files[1])

    def test_rapid_crashes_may_share_filename(self):
        write_crash_report(RuntimeError("e1"), [], 0, "NOP", {})
        write_crash_report(RuntimeError("e2"), [], 0, "NOP", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        self.assertGreaterEqual(len(files), 1)


class TestCrashReportEdgeCases(unittest.TestCase):
    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmpdir = tempfile.mkdtemp(prefix="eigen_crash_test_")
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)

    def test_empty_call_stack(self):
        write_crash_report(RuntimeError("e"), [], 0, "NOP", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Call Stack:", content)

    def test_frame_with_none_line(self):
        frame = _FakeFrame()
        frame.current_line = None
        write_crash_report(RuntimeError("e"), [frame], 0, "NOP", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("unknown line", content)

    def test_error_with_no_traceback(self):
        err = ValueError("simple error")
        self.assertIsNone(getattr(err, "__traceback__", None))
        write_crash_report(err, [], 0, "NOP", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("ValueError", content)
        self.assertIn("simple error", content)

    def test_complex_locals_serialized(self):
        complex_locals = {
            "arr": [1, 2, 3],
            "nested": {"key": "value"},
            "tup": (1, "two", 3.0),
        }
        write_crash_report(RuntimeError("e"), [], 0, "NOP", complex_locals)
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("arr", content)
        self.assertIn("nested", content)

    def test_report_has_header(self):
        write_crash_report(RuntimeError("e"), [], 0, "NOP", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("EIGEN VM CRASH REPORT", content)
        self.assertIn("=" * 60, content)

    def test_opcode_recorded(self):
        write_crash_report(RuntimeError("e"), [], 7, "Q_MEASURE", {})
        files = [f for f in os.listdir(self._tmpdir) if f.startswith("crash-")]
        with open(os.path.join(self._tmpdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Q_MEASURE", content)


if __name__ == "__main__":
    unittest.main()
