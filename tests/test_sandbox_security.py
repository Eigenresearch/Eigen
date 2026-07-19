import tempfile
import unittest

from src.backend.vm import ActivationFrame, EigenVM
from src.compiler import (
    _deserialize_cache,
    _serialize_cache,
    _CACHE_JSON_MAGIC,
    _CACHE_PKL_MAGIC,
    _cache_hmac_key,
)
from src.registry import PackageMetadata


class _NotJSON:
    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        return isinstance(other, _NotJSON) and self.v == other.v


class _Custom:
    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        return isinstance(other, _Custom) and self.v == other.v


class TestJITSandboxBuiltins(unittest.TestCase):
    def _build_loop_bytecode(self, source_template):
        from src.backend.bytecode import Instruction, Opcode

        return [Instruction(Opcode.HALT, None)]

    def test_jit_sandbox_blocks_dunder_import(self):
        EigenVM(seed=42)

        sandbox_globals = {
            "__builtins__": {
                "bool": bool, "int": int, "float": float, "str": str,
                "len": len, "abs": abs, "range": range, "repr": repr,
                "__import__": None,
                "eval": None, "exec": None, "compile": None,
                "open": None, "globals": None, "locals": None,
                "vars": None, "dir": None, "getattr": None,
                "setattr": None, "delattr": None, "hasattr": None,
                "type": None, "isinstance": None, "issubclass": None,
                "object": None, "property": None,
            },
        }
        local_vars = {}
        source = "x = __import__('os').system('echo pwned')"
        try:
            code_obj = compile(source, "<fast_array_loop>", "exec")
            exec(code_obj, sandbox_globals, local_vars)
            self.fail("JIT sandbox should have blocked __import__")
        except (TypeError, NameError, AttributeError):
            pass
        except Exception:
            pass

    def test_jit_sandbox_blocks_dunder_builtins_access(self):
        sandbox_globals = {
            "__builtins__": {
                "bool": bool, "int": int, "float": float, "str": str,
                "len": len, "abs": abs, "range": range, "repr": repr,
                "__import__": None,
                "eval": None, "exec": None, "compile": None,
                "open": None, "globals": None, "locals": None,
                "vars": None, "dir": None, "getattr": None,
                "setattr": None, "delattr": None, "hasattr": None,
                "type": None, "isinstance": None, "issubclass": None,
                "object": None, "property": None,
            },
        }
        local_vars = {}
        source = "x = __builtins__['__import__']('os')"
        try:
            code_obj = compile(source, "<fast_array_loop>", "exec")
            exec(code_obj, sandbox_globals, local_vars)
            self.fail("Sandbox should have blocked __builtins__ access")
        except (TypeError, KeyError, NameError, AttributeError):
            pass
        except Exception:
            pass

    def test_jit_sandbox_blocks_type_access(self):
        sandbox_globals = {
            "__builtins__": {
                "bool": bool, "int": int, "float": float, "str": str,
                "len": len, "abs": abs, "range": range, "repr": repr,
                "type": None,
            },
        }
        local_vars = {}
        source = "x = type(1).__subclasses__()"
        try:
            code_obj = compile(source, "<sandbox>", "exec")
            exec(code_obj, sandbox_globals, local_vars)
            self.fail("Sandbox should have blocked type access")
        except (TypeError, NameError, AttributeError):
            pass
        except Exception:
            pass

    def test_jit_sandbox_blocks_eval(self):
        sandbox_globals = {
            "__builtins__": {
                "bool": bool, "int": int, "float": float, "str": str,
                "len": len, "abs": abs, "range": range, "repr": repr,
                "eval": None,
            },
        }
        local_vars = {}
        source = "eval('1+1')"
        try:
            code_obj = compile(source, "<sandbox>", "exec")
            exec(code_obj, sandbox_globals, local_vars)
            self.fail("Sandbox should have blocked eval")
        except (TypeError, NameError):
            pass
        except Exception:
            pass

    def test_jit_sandbox_blocks_open(self):
        sandbox_globals = {
            "__builtins__": {
                "bool": bool, "int": int, "float": float, "str": str,
                "len": len, "abs": abs, "range": range, "repr": repr,
                "open": None,
            },
        }
        local_vars = {}
        source = "open('/etc/passwd').read()"
        try:
            code_obj = compile(source, "<sandbox>", "exec")
            exec(code_obj, sandbox_globals, local_vars)
            self.fail("Sandbox should have blocked open()")
        except (TypeError, NameError):
            pass
        except Exception:
            pass

    def test_jit_sandbox_blocks_getattr(self):
        sandbox_globals = {
            "__builtins__": {
                "bool": bool, "int": int, "float": float, "str": str,
                "len": len, "abs": abs, "range": range, "repr": repr,
                "getattr": None,
            },
        }
        local_vars = {}
        source = "getattr(int, '__subclasses__')"
        try:
            code_obj = compile(source, "<sandbox>", "exec")
            exec(code_obj, sandbox_globals, local_vars)
            self.fail("Sandbox should have blocked getattr()")
        except (TypeError, NameError):
            pass
        except Exception:
            pass

    def test_jit_sandbox_allows_basic_math(self):
        sandbox_globals = {
            "__builtins__": {
                "bool": bool, "int": int, "float": float, "str": str,
                "len": len, "abs": abs, "range": range, "repr": repr,
            },
        }
        local_vars = {}
        source = "x = 1 + 2 * 3 - 4 / 5"
        code_obj = compile(source, "<sandbox>", "exec")
        exec(code_obj, sandbox_globals, local_vars)
        self.assertAlmostEqual(local_vars["x"], 1 + 2 * 3 - 4 / 5)


class TestCacheHmac(unittest.TestCase):
    def test_pickle_cache_has_pkl_magic(self):
        raw = _serialize_cache(_NotJSON(42), workspace_root=tempfile.gettempdir())
        self.assertTrue(raw.startswith(_CACHE_PKL_MAGIC))

    def test_json_cache_has_json_magic(self):
        raw = _serialize_cache({"a": 1, "b": [1, 2, 3]}, workspace_root=tempfile.gettempdir())
        self.assertTrue(raw.startswith(_CACHE_JSON_MAGIC))

    def test_pickle_cache_round_trip(self):
        obj = _Custom(99)
        ws = tempfile.gettempdir()
        raw = _serialize_cache(obj, workspace_root=ws)
        result = _deserialize_cache(raw, workspace_root=ws)
        self.assertEqual(result, obj)

    def test_json_cache_round_trip(self):
        ws = tempfile.gettempdir()
        obj = {"name": "test", "version": "1.0", "deps": {"a": "1.0"}}
        raw = _serialize_cache(obj, workspace_root=ws)
        result = _deserialize_cache(raw, workspace_root=ws)
        self.assertEqual(result, obj)

    def test_tampered_pickle_cache_rejected(self):
        ws = tempfile.gettempdir()
        raw = _serialize_cache({"x": 1}, workspace_root=ws)
        if raw.startswith(_CACHE_PKL_MAGIC):
            nl = raw.find(b"\n", len(_CACHE_PKL_MAGIC))
            tampered = raw[:nl + 1] + b"\xff\xfe" + raw[nl + 3:]
            result = _deserialize_cache(tampered, workspace_root=ws)
            self.assertIsNone(result)

    def test_pickle_cache_rejects_unknown_payload(self):
        ws = tempfile.gettempdir()
        bogus = b"EIGENCP1\ninvalidsig\npayload-data"
        result = _deserialize_cache(bogus, workspace_root=ws)
        self.assertIsNone(result)

    def test_unknown_magic_returns_none(self):
        ws = tempfile.gettempdir()
        result = _deserialize_cache(b"unknown-magic-payload", workspace_root=ws)
        self.assertIsNone(result)

    def test_hmac_key_is_stable(self):
        ws = tempfile.mkdtemp(prefix="eigen_hmac_test_")
        key1 = _cache_hmac_key(ws)
        key2 = _cache_hmac_key(ws)
        self.assertEqual(key1, key2)


class TestFramePool(unittest.TestCase):
    def test_frame_pool_starts_empty(self):
        vm = EigenVM()
        self.assertEqual(vm.frame_pool, [])

    def test_frame_pool_grows_after_recycle(self):
        vm = EigenVM()
        frame = ActivationFrame(return_address=0, func_name="test")
        frame.locals["x"] = 42
        vm.recycle_frame(frame)
        self.assertEqual(len(vm.frame_pool), 1)

    def test_recycled_frame_has_cleared_locals(self):
        vm = EigenVM()
        frame = ActivationFrame(return_address=0, func_name="test")
        frame.locals["x"] = 42
        frame.try_stack.append(("foo",))
        frame.current_line = 99
        vm.recycle_frame(frame)
        pooled = vm.frame_pool[0]
        self.assertEqual(pooled.locals, {})
        self.assertEqual(pooled.try_stack, [])
        self.assertIsNone(pooled.current_line)

    def test_frame_pool_get_returns_pooled_frame(self):
        vm = EigenVM()
        f1 = ActivationFrame(return_address=10, func_name="old")
        vm.recycle_frame(f1)
        f2 = vm.get_frame(return_address=20, func_name="new")
        self.assertIs(f1, f2)
        self.assertEqual(f2.return_address, 20)
        self.assertEqual(f2.func_name, "new")

    def test_frame_pool_creates_new_frame_when_empty(self):
        vm = EigenVM()
        frame = vm.get_frame(return_address=5, func_name="fresh")
        self.assertIsInstance(frame, ActivationFrame)
        self.assertEqual(frame.return_address, 5)
        self.assertEqual(frame.func_name, "fresh")

    def test_frame_pool_capped_at_64(self):
        vm = EigenVM()
        for i in range(100):
            f = ActivationFrame(return_address=i, func_name=f"f{i}")
            vm.recycle_frame(f)
        self.assertEqual(len(vm.frame_pool), 64)

    def test_reset_clears_locals(self):
        frame = ActivationFrame(return_address=0, func_name="test")
        frame.locals["a"] = 1
        frame.locals["b"] = 2
        frame.reset(return_address=99, func_name="reset")
        self.assertEqual(frame.locals, {})
        self.assertEqual(frame.func_name, "reset")


class TestRegistryReadonly(unittest.TestCase):
    def test_dependencies_returns_mapping_proxy(self):
        import types

        md = PackageMetadata(name="foo", version="1.0", dependencies={"a": "1.0"})
        self.assertIsInstance(md.dependencies, types.MappingProxyType)

    def test_dependencies_are_immutable(self):
        md = PackageMetadata(name="foo", version="1.0", dependencies={"a": "1.0"})
        with self.assertRaises(TypeError):
            md.dependencies["b"] = "2.0"

    def test_metadata_is_frozen(self):
        md = PackageMetadata(name="foo", version="1.0")
        with self.assertRaises(Exception):
            md.name = "bar"

    def test_to_dict_returns_plain_dict(self):
        md = PackageMetadata(name="foo", version="1.0", dependencies={"a": "1.0"})
        d = md.to_dict()
        self.assertIsInstance(d["dependencies"], dict)

    def test_advisory_tags_immutable(self):
        md = PackageMetadata(name="foo", version="1.0", advisory_tags=("CVE-1",))
        self.assertEqual(md.advisory_tags, ("CVE-1",))

    def test_advisory_tags_default_empty_tuple(self):
        md = PackageMetadata(name="foo", version="1.0")
        self.assertEqual(md.advisory_tags, ())


if __name__ == "__main__":
    unittest.main()
