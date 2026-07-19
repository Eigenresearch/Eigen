"""§4.3 — Foreign Function Interface (FFI) tests."""
import unittest

from src.ffi import (
    FFIType,
    FFIFunction,
    CHeaderEmitter,
    RustFFIEmitter,
    PythonFFIBindingEmitter,
    WASMFunction,
    WASMModule,
)


# ---------------------------------------------------------------------------
# FFIType and FFIFunction
# ---------------------------------------------------------------------------

class TestFFIType(unittest.TestCase):
    def test_has_core_types(self):
        self.assertEqual(len(FFIType), 12)
        names = {t.value for t in FFIType}
        for n in ("void", "int32_t", "int64_t", "uint32_t", "uint64_t",
                    "size_t", "float", "bool", "const char*",
                    "eigen_handle", "void*", "double"):
            self.assertIn(n, names)


class TestFFIFunction(unittest.TestCase):
    def test_default_return_is_void(self):
        fn = FFIFunction(name="x")
        self.assertEqual(fn.return_type, FFIType.VOID)
        self.assertEqual(fn.parameters, [])
        self.assertEqual(fn.description, "")

    def test_full_construction(self):
        fn = FFIFunction(name="add",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)],
                            description="adds two ints")
        self.assertEqual(fn.name, "add")
        self.assertEqual(fn.return_type, FFIType.INT32)
        self.assertEqual(fn.parameters,
                           [("a", FFIType.INT32), ("b", FFIType.INT32)])
        self.assertEqual(fn.description, "adds two ints")


# ---------------------------------------------------------------------------
# C header emitter
# ---------------------------------------------------------------------------

class TestCHeaderEmitter(unittest.TestCase):
    def test_empty_header_has_include_guard(self):
        em = CHeaderEmitter(header_name="foo.h")
        out = em.emit()
        self.assertIn("#ifndef FOO_H", out)
        self.assertIn("#define FOO_H", out)
        self.assertIn("#endif  // FOO_H", out)

    def test_header_includes_system_headers(self):
        em = CHeaderEmitter()
        out = em.emit()
        self.assertIn("#include <stdint.h>", out)
        self.assertIn("#include <stdbool.h>", out)

    def test_wraps_in_extern_c(self):
        em = CHeaderEmitter()
        out = em.emit()
        self.assertIn("#ifdef __cplusplus", out)
        self.assertIn('extern "C" {', out)

    def test_emits_eigen_handle_typedef(self):
        em = CHeaderEmitter()
        out = em.emit()
        self.assertIn("typedef struct EigenHandle EigenHandle;", out)

    def test_function_signature(self):
        em = CHeaderEmitter()
        em.add(FFIFunction(name="add",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)]))
        out = em.emit()
        self.assertIn("int32_t add(int32_t a, int32_t b);", out)

    def test_function_with_void_params_renders_void(self):
        em = CHeaderEmitter()
        em.add(FFIFunction(name="ping", return_type=FFIType.VOID))
        out = em.emit()
        self.assertIn("void ping(void);", out)

    def test_eigen_handle_param_uses_typedef(self):
        em = CHeaderEmitter()
        em.add(FFIFunction(name="release",
                            return_type=FFIType.VOID,
                            parameters=[
                                ("handle", FFIType.EIGEN_HANDLE)]))
        out = em.emit()
        self.assertIn("void release(EigenHandle* handle);", out)

    def test_description_emitted_as_comment(self):
        em = CHeaderEmitter()
        em.add(FFIFunction(name="add",
                            description="adds two ints",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32)]))
        out = em.emit()
        self.assertIn("/* adds two ints */", out)


# ---------------------------------------------------------------------------
# Rust emitter
# ---------------------------------------------------------------------------

class TestRustFFIEmitter(unittest.TestCase):
    def test_emits_eigen_handle_struct(self):
        em = RustFFIEmitter()
        out = em.emit()
        self.assertIn("#[repr(C)]", out)
        self.assertIn("pub struct EigenHandle", out)

    def test_emits_no_mangle_extern_c(self):
        em = RustFFIEmitter()
        em.add(FFIFunction(name="add",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)]))
        out = em.emit()
        self.assertIn('#[no_mangle]', out)
        self.assertIn('pub extern "C" fn add', out)
        self.assertIn('-> i32', out)

    def test_void_return_omits_arrow(self):
        em = RustFFIEmitter()
        em.add(FFIFunction(name="hello", return_type=FFIType.VOID))
        out = em.emit()
        # void fn shouldn't have `-> void` part (Rust uses `()`)
        self.assertNotIn('-> void', out)

    def test_param_uses_rust_types(self):
        em = RustFFIEmitter()
        em.add(FFIFunction(name="recv",
                            return_type=FFIType.VOID,
                            parameters=[("data", FFIType.CHAR_PTR)]))
        out = em.emit()
        self.assertIn("data: *const c_char", out)


# ---------------------------------------------------------------------------
# Python binding emitter
# ---------------------------------------------------------------------------

class TestPythonFFIBindingEmitter(unittest.TestCase):
    def test_emits_module_docstring(self):
        em = PythonFFIBindingEmitter(module_name="my_ffi")
        out = em.emit()
        # Docstring names the module; allow `my_ffi` with optional
        # trailing punctuation.
        first_line = out.split("\n", 1)[0]
        self.assertTrue(('"""Auto-generated Python FFI '
                            'bindings') in first_line,
                            msg=f"first_line={first_line!r}")
        self.assertIn("my_ffi", first_line)

    def test_emits_lib_placeholder(self):
        out = PythonFFIBindingEmitter().emit()
        self.assertIn("_lib = None", out)

    def test_emits_function_def(self):
        em = PythonFFIBindingEmitter()
        em.add(FFIFunction(name="add",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)],
                            description="add two ints"))
        out = em.emit()
        self.assertIn("def add(a, b) -> int:", out)
        self.assertIn("add two ints", out)

    def test_emit_void_returns_none(self):
        em = PythonFFIBindingEmitter()
        em.add(FFIFunction(name="release",
                            return_type=FFIType.VOID,
                            parameters=[
                                ("handle", FFIType.EIGEN_HANDLE)]))
        out = em.emit()
        self.assertIn("def release(handle) -> None:", out)
        # Production ctypes bindings use RuntimeError, not NotImplementedError
        self.assertIn("ctypes", out)


# ---------------------------------------------------------------------------
# WASM text-format emitter
# ---------------------------------------------------------------------------

class TestWASMModule(unittest.TestCase):
    def test_empty_module_emits(self):
        wm = WASMModule()
        out = wm.emit_wat()
        self.assertIn("(module", out)
        self.assertIn(")", out)

    def test_default_add_function_body(self):
        wm = WASMModule()
        wm.add_function(FFIFunction(name="add",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)]))
        out = wm.emit_wat()
        self.assertIn("(func $add", out)
        self.assertIn("(param i32)", out)
        self.assertIn("(result i32)", out)
        self.assertIn("local.get 0", out)
        self.assertIn("local.get 1", out)
        self.assertIn("i32.add", out)

    def test_void_function_default_nop(self):
        wm = WASMModule()
        wm.add_function(FFIFunction(name="noop",
                            return_type=FFIType.VOID))
        out = wm.emit_wat()
        self.assertIn("(func $noop", out)
        self.assertIn("nop", out)
        # Void function should not have a (result) clause.
        self.assertNotIn("(result", out)

    def test_custom_body(self):
        wm = WASMModule()
        wm.add_function(FFIFunction(name="sub",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)]),
                          body=["local.get 0", "local.get 1", "i32.sub"])
        out = wm.emit_wat()
        self.assertIn("i32.sub", out)

    def test_float64_constant_default_body(self):
        wm = WASMModule()
        wm.add_function(FFIFunction(name="get_pi",
                            return_type=FFIType.FLOAT64))
        out = wm.emit_wat()
        # Default body for a no-param f64 function should produce
        # an f64.const instruction.
        self.assertTrue("f64.const" in out,
                          msg=f"missing f64.const in {out!r}")

    def test_two_functions_in_module(self):
        wm = WASMModule()
        wm.add_function(FFIFunction(name="add",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)]))
        wm.add_function(FFIFunction(name="sub",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)]))
        out = wm.emit_wat()
        self.assertEqual(out.count("(func $"), 2)
        self.assertIn("(func $add", out)
        self.assertIn("(func $sub", out)


class TestWASMFunction(unittest.TestCase):
    def test_dataclass_fields(self):
        fn = FFIFunction(name="foo")
        wf = WASMFunction(spec=fn, body=["nop"])
        self.assertIs(wf.spec, fn)
        self.assertEqual(wf.body, ["nop"])

    def test_default_body_used_when_none(self):
        # If the user passes body=None... the dataclass doesn't
        # auto-apply default_body — only WASMModule.add_function()
        # does.  Here, we explicitly check that WASMFunction stores
        # the body as provided.
        wm = WASMModule()
        wm.add_function(FFIFunction(name="add",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)]))
        self.assertEqual(wm._functions[0].body,
                           ["local.get 0", "local.get 1", "i32.add"])


# ---------------------------------------------------------------------------
# Cross-emitter integration
# ---------------------------------------------------------------------------

class TestCrossEmitter(unittest.TestCase):
    def test_all_emitters_accept_same_function(self):
        fn = FFIFunction(name="add",
                            return_type=FFIType.INT32,
                            parameters=[("a", FFIType.INT32),
                                          ("b", FFIType.INT32)],
                            description="add")
        c = CHeaderEmitter()
        c.add(fn)
        r = RustFFIEmitter()
        r.add(fn)
        p = PythonFFIBindingEmitter()
        p.add(fn)
        w = WASMModule()
        w.add_function(fn)
        # Just confirm all four emitters produce non-empty output
        # referencing the function name.
        for emitter_name, out in (("C", c.emit()),
                                    ("Rust", r.emit()),
                                    ("Python", p.emit()),
                                    ("WASM", w.emit_wat())):
            self.assertIn("add", out)
            self.assertGreater(len(out), 10,
                                  msg=f"{emitter_name} emitter empty")


if __name__ == "__main__":
    unittest.main()
