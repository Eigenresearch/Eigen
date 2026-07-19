"""§4.3 — Foreign Function Interface (FFI) envelope.

Roadmap ("4.3 FFI"):

    - [x] Python FFI — codegen helper that bridges Eigen `FFIFunction`
          specs to a Python-side `cffi`-style stub emitter (the
          complementary Python↔Rust direction is already covered by
          `src.native_integration_envelope`).
    - [x] Rust FFI — codegen helper that emits `#[no_mangle] extern "C"`
          Rust source from `FFIFunction` specs.
    - [x] C FFI — codegen helper that emits a portable C99 header
          with `#include <stdint.h>` / `<stdbool.h>` and the function
          declarations.
    - [x] WASM target — minimal `.wat` text-format emitter that
          produces a valid WebAssembly text module from
          `FFIFunction` specs + a small body instruction list.

This is an envelope module: the codegen is textual and does NOT
require the actual `wasmtime`/`cbindgen`/`maturin` toolchain to
exist; the caller can write the resulting strings to disk.
"""
from __future__ import annotations

import dataclasses
import enum
import typing


# ---------------------------------------------------------------------------
# FFI types
# ---------------------------------------------------------------------------

class FFIType(enum.Enum):
    """Subset of C99 stdint types + a few Eigen-specific ones.

    `EIGEN_HANDLE` is an opaque pointer to runtime data; the WASM
    emitter represents it as an `i32` opaque index (32-bit memory
    offsets).
    """
    VOID = "void"
    INT32 = "int32_t"
    INT64 = "int64_t"
    UINT32 = "uint32_t"
    UINT64 = "uint64_t"
    SIZE = "size_t"
    FLOAT32 = "float"
    FLOAT64 = "double"
    BOOL = "bool"
    CHAR_PTR = "const char*"
    VOID_PTR = "void*"
    EIGEN_HANDLE = "eigen_handle"


_RUST_TYPE_MAP: typing.Mapping[FFIType, str] = {
    FFIType.VOID: "void",                      # `void` return only
    FFIType.INT32: "i32",
    FFIType.INT64: "i64",
    FFIType.UINT32: "u32",
    FFIType.UINT64: "u64",
    FFIType.SIZE: "usize",
    FFIType.FLOAT32: "f32",
    FFIType.FLOAT64: "f64",
    FFIType.BOOL: "bool",
    FFIType.CHAR_PTR: "*const c_char",
    FFIType.VOID_PTR: "*mut c_void",
    FFIType.EIGEN_HANDLE: "EigenHandle",
}

_WASM_TYPE_MAP: typing.Mapping[FFIType, str] = {
    FFIType.VOID: "",  # empty result list
    FFIType.INT32: "i32",
    FFIType.INT64: "i64",
    FFIType.UINT32: "i32",
    FFIType.UINT64: "i64",
    FFIType.SIZE: "i32",
    FFIType.FLOAT32: "f32",
    FFIType.FLOAT64: "f64",
    FFIType.BOOL: "i32",
    FFIType.CHAR_PTR: "i32",  # pointer as i32 index (Linear mem)
    FFIType.VOID_PTR: "i32",
    FFIType.EIGEN_HANDLE: "i32",
}

_PYTHON_TYPE_MAP: typing.Mapping[FFIType, str] = {
    FFIType.VOID: "None",
    FFIType.INT32: "int",
    FFIType.INT64: "int",
    FFIType.UINT32: "int",
    FFIType.UINT64: "int",
    FFIType.SIZE: "int",
    FFIType.FLOAT32: "float",
    FFIType.FLOAT64: "float",
    FFIType.BOOL: "bool",
    FFIType.CHAR_PTR: "bytes",
    FFIType.VOID_PTR: "object",
    FFIType.EIGEN_HANDLE: "object",
}

_CTYPES_TYPE_MAP: typing.Mapping[FFIType, str] = {
    FFIType.VOID: "None",
    FFIType.INT32: "ctypes.c_int32",
    FFIType.INT64: "ctypes.c_int64",
    FFIType.UINT32: "ctypes.c_uint32",
    FFIType.UINT64: "ctypes.c_uint64",
    FFIType.SIZE: "ctypes.c_size_t",
    FFIType.FLOAT32: "ctypes.c_float",
    FFIType.FLOAT64: "ctypes.c_double",
    FFIType.BOOL: "ctypes.c_bool",
    FFIType.CHAR_PTR: "ctypes.c_char_p",
    FFIType.VOID_PTR: "ctypes.c_void_p",
    FFIType.EIGEN_HANDLE: "ctypes.c_void_p",
}

_GO_TYPE_MAP: typing.Mapping[FFIType, str] = {
    FFIType.VOID: "",
    FFIType.INT32: "int32",
    FFIType.INT64: "int64",
    FFIType.UINT32: "uint32",
    FFIType.UINT64: "uint64",
    FFIType.SIZE: "uintptr",
    FFIType.FLOAT32: "float32",
    FFIType.FLOAT64: "float64",
    FFIType.BOOL: "bool",
    FFIType.CHAR_PTR: "*byte",
    FFIType.VOID_PTR: "unsafe.Pointer",
    FFIType.EIGEN_HANDLE: "unsafe.Pointer",
}

_SWIFT_TYPE_MAP: typing.Mapping[FFIType, str] = {
    FFIType.VOID: "Void",
    FFIType.INT32: "Int32",
    FFIType.INT64: "Int64",
    FFIType.UINT32: "UInt32",
    FFIType.UINT64: "UInt64",
    FFIType.SIZE: "Int",
    FFIType.FLOAT32: "Float",
    FFIType.FLOAT64: "Double",
    FFIType.BOOL: "Bool",
    FFIType.CHAR_PTR: "UnsafePointer<CChar>",
    FFIType.VOID_PTR: "UnsafeMutableRawPointer",
    FFIType.EIGEN_HANDLE: "OpaquePointer",
}


@dataclasses.dataclass
class FFIFunction:
    """A single function signature for cross-language calling."""
    name: str
    return_type: FFIType = FFIType.VOID
    parameters: typing.List[typing.Tuple[str, FFIType]] = \
        dataclasses.field(default_factory=list)
    description: str = ""


@dataclasses.dataclass(frozen=True)
class FFIArrayType:
    """A fixed-length array type for FFI signatures.

    `length=None` denotes a dynamically-sized array (decays to a
    pointer in C). A concrete length renders as `element arr[length]`
    in C and `[element; length]` in Rust.
    """
    element_type: FFIType
    length: typing.Optional[int] = None


@dataclasses.dataclass(frozen=True)
class FFISliceType:
    """A length-prefixed view over contiguous elements.

    Renders as a `{T* ptr; size_t len}` struct in C, `&[T]` in Rust,
    and a pointer+length pair in other emitters.
    """
    element_type: FFIType


FFITypeLike = typing.Union[FFIType, FFIArrayType, FFISliceType]


def _is_array_type(t: FFITypeLike) -> bool:
    return isinstance(t, (FFIArrayType, FFISliceType))


def _c_type_str(t: FFITypeLike) -> str:
    if isinstance(t, FFIArrayType):
        if t.length is None:
            return f"{_c_type_str(t.element_type)}*"
        return f"{_c_type_str(t.element_type)}[{t.length}]"
    if isinstance(t, FFISliceType):
        return f"{_c_type_str(t.element_type)}*"
    if t is FFIType.EIGEN_HANDLE:
        return "EigenHandle*"
    return t.value


def _rust_type_str(t: FFITypeLike) -> str:
    if isinstance(t, FFIArrayType):
        if t.length is None:
            return f"*mut {_RUST_TYPE_MAP.get(t.element_type, 'i32')}"
        return f"[{_RUST_TYPE_MAP.get(t.element_type, 'i32')}; {t.length}]"
    if isinstance(t, FFISliceType):
        return f"&[{_RUST_TYPE_MAP.get(t.element_type, 'i32')}]"
    return _RUST_TYPE_MAP.get(t, "i32")


def _wasm_type_str(t: FFITypeLike) -> str:
    if isinstance(t, FFIArrayType):
        return _WASM_TYPE_MAP.get(t.element_type, "i32")
    if isinstance(t, FFISliceType):
        return _WASM_TYPE_MAP.get(t.element_type, "i32")
    return _WASM_TYPE_MAP.get(t, "i32")


def _python_type_str(t: FFITypeLike) -> str:
    if isinstance(t, FFIArrayType):
        return f"typing.List[{_PYTHON_TYPE_MAP.get(t.element_type, 'int')}]"
    if isinstance(t, FFISliceType):
        return f"typing.List[{_PYTHON_TYPE_MAP.get(t.element_type, 'int')}]"
    return _PYTHON_TYPE_MAP.get(t, "object")


def _ctypes_type_str(t: FFITypeLike) -> str:
    if isinstance(t, FFIArrayType):
        return f"ctypes.POINTER({_CTYPES_TYPE_MAP.get(t.element_type, 'ctypes.c_int32')})"
    if isinstance(t, FFISliceType):
        return f"ctypes.POINTER({_CTYPES_TYPE_MAP.get(t.element_type, 'ctypes.c_int32')})"
    return _CTYPES_TYPE_MAP.get(t, "ctypes.c_void_p")


def _go_type_str(t: FFITypeLike) -> str:
    if isinstance(t, FFIArrayType):
        if t.length is None:
            return f"[]{_GO_TYPE_MAP.get(t.element_type, 'int32')}"
        return f"[{t.length}]{_GO_TYPE_MAP.get(t.element_type, 'int32')}"
    if isinstance(t, FFISliceType):
        return f"[]{_GO_TYPE_MAP.get(t.element_type, 'int32')}"
    return _GO_TYPE_MAP.get(t, "int32")


def _swift_type_str(t: FFITypeLike) -> str:
    if isinstance(t, FFIArrayType):
        return f"UnsafePointer<{_SWIFT_TYPE_MAP.get(t.element_type, 'Int32')}>"
    if isinstance(t, FFISliceType):
        return f"UnsafePointer<{_SWIFT_TYPE_MAP.get(t.element_type, 'Int32')}>"
    return _SWIFT_TYPE_MAP.get(t, "Int32")


# ---------------------------------------------------------------------------
# C header emitter
# ---------------------------------------------------------------------------

class CHeaderEmitter:
    """Generate a portable C99 header from `FFIFunction` specs."""

    def __init__(self, header_name: str = "eigen_ffi.h"):
        self.header_name = header_name
        self.functions: typing.List[FFIFunction] = []

    def add(self, fn: FFIFunction) -> None:
        self.functions.append(fn)

    def emit(self) -> str:
        guard = self.header_name.upper().replace('.', '_').replace(
            '-', '_')
        lines: typing.List[str] = []
        lines.append(f"#ifndef {guard}")
        lines.append(f"#define {guard}")
        lines.append("")
        lines.append("#include <stdint.h>")
        lines.append("#include <stdbool.h>")
        lines.append("#include <stddef.h>")
        lines.append("")
        lines.append("#ifdef __cplusplus")
        lines.append('extern "C" {')
        lines.append("#endif")
        lines.append("")
        # Emit a forward-declared EigenHandle struct
        lines.append("typedef struct EigenHandle EigenHandle;")
        lines.append("")
        for fn in self.functions:
            lines.append(self._emit_function(fn))
            lines.append("")
        lines.append("#ifdef __cplusplus")
        lines.append("}")  # close extern "C"
        lines.append("#endif")
        lines.append("")
        lines.append(f"#endif  // {guard}")
        return "\n".join(lines)

    def _emit_function(self, fn: FFIFunction) -> str:
        ret = _c_type_str(fn.return_type)
        params_str = ", ".join(
            self._c_param(name, t)
            for name, t in fn.parameters) or "void"
        description = f"/* {fn.description} */\n" \
            if fn.description else ""
        return f"{description}{ret} {fn.name}({params_str});"

    @staticmethod
    def _c_param(name: str, t: FFITypeLike) -> str:
        if isinstance(t, FFIArrayType):
            elem = _c_type_str(t.element_type)
            if t.length is not None:
                return f"{elem} {name}[{t.length}]"
            return f"{elem}* {name}"
        if isinstance(t, FFISliceType):
            return f"{_c_type_str(t.element_type)}* {name}"
        return f"{_c_type_str(t)} {name}"

    @staticmethod
    def _c_type(t: FFITypeLike) -> str:
        return _c_type_str(t)


# ---------------------------------------------------------------------------
# Rust emitter
# ---------------------------------------------------------------------------

class RustFFIEmitter:
    """Generate a Rust source file with `#[no_mangle] extern "C"`
    functions from `FFIFunction` specs."""

    def __init__(self, module_name: str = "eigen_ffi"):
        self.module_name = module_name
        self.functions: typing.List[FFIFunction] = []

    def add(self, fn: FFIFunction) -> None:
        self.functions.append(fn)

    def emit(self) -> str:
        lines: typing.List[str] = []
        lines.append("#[repr(C)]")
        lines.append("pub struct EigenHandle {")
        lines.append("    _opaque: [u8; 0],")
        lines.append("}")
        lines.append("")
        for fn in self.functions:
            lines.append(self._emit_function(fn))
            lines.append("")
        return "\n".join(lines)

    def _emit_function(self, fn: FFIFunction) -> str:
        ret_rust = _rust_type_str(fn.return_type)
        ret_void = fn.return_type is FFIType.VOID
        ret_part = "" if ret_void else f" -> {ret_rust}"
        params_str = ", ".join(
            f"{name}: {_rust_type_str(t)}"
            for name, t in fn.parameters)
        # Generate compilable default implementations:
        # - void functions: empty body (no-op)
        # - numeric returns: default value (0)
        # - pointer returns: null
        if ret_void:
            body = f"    // {fn.name}: default no-op implementation"
        elif fn.return_type in (FFIType.INT32, FFIType.INT64,
                                  FFIType.UINT32, FFIType.UINT64,
                                  FFIType.SIZE):
            body = f"    // {fn.name}: default returns 0\n    0"
        elif fn.return_type in (FFIType.FLOAT32, FFIType.FLOAT64):
            body = f"    // {fn.name}: default returns 0.0\n    0.0"
        elif fn.return_type is FFIType.BOOL:
            body = f"    // {fn.name}: default returns false\n    false"
        else:
            body = (f"    // {fn.name}: default returns null\n"
                     f"    std::ptr::null_mut()")
        return (f"#[no_mangle]\npub extern \"C\" fn {fn.name}"
                f"({params_str}){ret_part} {{\n{body}\n}}")


class PythonFFIBindingEmitter:
    """Generate a Python-side stub module that documents the
    FFIFunction signatures (useful for `cffi`-style binding shims).

    The output is descriptive — no real dlopen/FFI is performed by
    the generated code. Users can replace the bodies with real
    `cffi.load(...)` calls when wiring to a real shared library."""

    def __init__(self, module_name: str = "eigen_ffi_python"):
        self.module_name = module_name
        self.functions: typing.List[FFIFunction] = []

    def add(self, fn: FFIFunction) -> None:
        self.functions.append(fn)

    def emit(self) -> str:
        lines = [
            f'"""Auto-generated Python FFI bindings for '
            f'{self.module_name}."""',
            "import ctypes",
            "import os",
            "from typing import Optional",
            "",
            f"_lib: Optional[ctypes.CDLL] = None",
            "",
            f"def _load_lib() -> ctypes.CDLL:",
            f'    """Load the shared library at runtime."""',
            f"    global _lib",
            f"    if _lib is not None:",
            f"        return _lib",
            f"    lib_name = '{self.module_name}'",
            f"    if os.name == 'nt':",
            f"        lib_name += '.dll'",
            f"    elif os.name == 'posix':",
            f"        lib_name = 'lib' + lib_name + '.so'",
            f"    try:",
            f"        _lib = ctypes.CDLL(lib_name)",
            f"    except OSError:",
            f"        _lib = None",
            f"    return _lib",
            "",
        ]
        for fn in self.functions:
            params = ", ".join(name for name, _
                                 in fn.parameters)
            ret_py = _python_type_str(fn.return_type)
            ret_ctype = _ctypes_type_str(fn.return_type)
            lines.append(f"def {fn.name}({params}) -> {ret_py}:")
            if fn.description:
                lines.append(f'    """{fn.description}"""')
            lines.append(f"    lib = _load_lib()")
            lines.append(f"    if lib is None:")
            lines.append(f'        raise RuntimeError(')
            lines.append(f'            "Shared library not loaded. '
                         f'Ensure {self.module_name} is installed.")')
            lines.append(f"    _fn = getattr(lib, '{fn.name}')")
            lines.append(f"    _fn.restype = {ret_ctype}")
            # Build argtypes
            argtypes = ", ".join(
                _ctypes_type_str(t)
                for name, t in fn.parameters)
            lines.append(f"    _fn.argtypes = [{argtypes}]")
            lines.append(f"    return _fn({params})")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# WASM text-format emitter
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WASMFunction:
    spec: FFIFunction
    body: typing.List[str]  # .wat instructions, e.g. ["i32.add"]
    export: typing.Optional[str] = None


@dataclasses.dataclass
class WASMExport:
    """A WebAssembly module export declaration.

    `kind` is one of `"func"`, `"table"`, `"memory"`, `"global"`.
    `name` is the export name visible to the host; `ref` is the
    internal WAT reference (e.g. `$add` for a function).
    """
    kind: str = "func"
    name: str = ""
    ref: str = ""


class WASMModule:
    """Minimal `.wat` text-format emitter."""

    def __init__(self, module_name: str = "eigen_wasm"):
        self.module_name = module_name
        self._functions: typing.List[WASMFunction] = []
        self._exports: typing.List[WASMExport] = []

    def add_function(self, fn: FFIFunction,
                       body: typing.Optional[
                           typing.List[str]] = None,
                       export: typing.Optional[str] = None) -> None:
        body = body or self.default_body(fn)
        self._functions.append(
            WASMFunction(spec=fn, body=list(body), export=export))

    def add_export(self, name: str, ref: str,
                     kind: str = "func") -> None:
        self._exports.append(WASMExport(kind=kind, name=name, ref=ref))

    @staticmethod
    def default_body(fn: FFIFunction) -> typing.List[str]:
        """Pick a reasonable default body based on signature."""
        param_types = [t for _, t in fn.parameters]
        if (fn.return_type is FFIType.INT32
                and param_types == [FFIType.INT32, FFIType.INT32]):
            return ["local.get 0", "local.get 1", "i32.add"]
        # No return, no params → a function that does nothing
        if fn.return_type is FFIType.VOID and not param_types:
            return ["nop"]
        # Default: return a constant based on first return type
        r = _wasm_type_str(fn.return_type)
        if r:
            return [f"{r}.const 0"]
        return []

    def emit_wat(self) -> str:
        lines = ["(module"]
        for f in self._functions:
            lines.append(self._emit_function(f))
        for ex in self._exports:
            lines.append(
                f'  (export "{ex.name}" ({ex.kind} {ex.ref}))')
        lines.append(")")
        return "\n".join(lines)

    def _emit_function(self, f: WASMFunction) -> str:
        fn = f.spec
        param_str = " ".join(
            f"(param {_wasm_type_str(t)})" for _, t in fn.parameters)
        ret_type = _wasm_type_str(fn.return_type)
        result_str = f"(result {ret_type})" if ret_type else ""
        export_str = f' (export "{f.export}")' if f.export else ""
        header = (f"  (func ${fn.name}{export_str}"
                    + (f" {param_str}" if param_str else "")
                    + (f" {result_str}" if result_str else ""))
        body_lines = "\n".join(f"    {ins}" for ins in f.body)
        return f"{header}\n{body_lines}\n  )"


# ---------------------------------------------------------------------------
# Go FFI emitter (stub)
# ---------------------------------------------------------------------------

class GoFFIEmitter:
    """Generate a Go source file with `//export`-style CGO bindings
    from `FFIFunction` specs.

    This is a stub emitter: the generated package compiles only when
    linked against a real C archive via `cgo`. The output documents
    the signatures and provides a reasonable default body so callers
    can wire real implementations later.
    """

    def __init__(self, package_name: str = "eigenffi"):
        self.package_name = package_name
        self.functions: typing.List[FFIFunction] = []

    def add(self, fn: FFIFunction) -> None:
        self.functions.append(fn)

    def emit(self) -> str:
        lines: typing.List[str] = []
        lines.append(f"package {self.package_name}")
        lines.append("")
        lines.append("// #cgo CFLAGS: -I.")
        lines.append("// #cgo LDFLAGS: -leigen_ffi")
        lines.append('import "C"')
        lines.append('import "unsafe"')
        lines.append("")
        lines.append("// EigenHandle is an opaque pointer to runtime data.")
        lines.append("type EigenHandle = unsafe.Pointer")
        lines.append("")
        for fn in self.functions:
            lines.append(self._emit_function(fn))
            lines.append("")
        return "\n".join(lines)

    def _emit_function(self, fn: FFIFunction) -> str:
        ret_go = _go_type_str(fn.return_type)
        ret_void = fn.return_type is FFIType.VOID
        ret_part = "" if ret_void else f" {ret_go}"
        params_str = ", ".join(
            f"{name} {_go_type_str(t)}"
            for name, t in fn.parameters)
        if ret_void:
            body = f"\t// {fn.name}: default no-op implementation"
        elif fn.return_type in (FFIType.INT32, FFIType.INT64,
                                  FFIType.UINT32, FFIType.UINT64,
                                  FFIType.SIZE):
            body = f"\t// {fn.name}: default returns 0\n\treturn 0"
        elif fn.return_type in (FFIType.FLOAT32, FFIType.FLOAT64):
            body = f"\t// {fn.name}: default returns 0.0\n\treturn 0.0"
        elif fn.return_type is FFIType.BOOL:
            body = f"\t// {fn.name}: default returns false\n\treturn false"
        else:
            body = f"\t// {fn.name}: default returns nil\n\treturn nil"
        export = f"//export {fn.name}\n" if fn.description else ""
        return (f"{export}func {fn.name}({params_str}){ret_part} {{\n"
                f"{body}\n}}")


# ---------------------------------------------------------------------------
# Swift FFI emitter (stub)
# ---------------------------------------------------------------------------

class SwiftFFIEmitter:
    """Generate a Swift source file bridging to the C header via
    `import CEigenFFI` from `FFIFunction` specs.

    This is a stub emitter: the generated file imports a Clang module
    named `CEigenFFI` that the user must build from the C header.
    Default bodies are placeholders.
    """

    def __init__(self, module_name: str = "CEigenFFI"):
        self.module_name = module_name
        self.functions: typing.List[FFIFunction] = []

    def add(self, fn: FFIFunction) -> None:
        self.functions.append(fn)

    def emit(self) -> str:
        lines: typing.List[str] = []
        lines.append(f"import {self.module_name}")
        lines.append("import Foundation")
        lines.append("")
        lines.append("/// Opaque handle to Eigen runtime data.")
        lines.append("public typealias EigenHandle = OpaquePointer")
        lines.append("")
        for fn in self.functions:
            lines.append(self._emit_function(fn))
            lines.append("")
        return "\n".join(lines)

    def _emit_function(self, fn: FFIFunction) -> str:
        ret_swift = _swift_type_str(fn.return_type)
        ret_void = fn.return_type is FFIType.VOID
        ret_part = "" if ret_void else f" -> {ret_swift}"
        params_str = ", ".join(
            f"{name}: {_swift_type_str(t)}"
            for name, t in fn.parameters)
        if ret_void:
            body = f"    // {fn.name}: default no-op implementation"
        elif fn.return_type in (FFIType.INT32, FFIType.INT64,
                                  FFIType.UINT32, FFIType.UINT64,
                                  FFIType.SIZE):
            body = f"    // {fn.name}: default returns 0\n    return 0"
        elif fn.return_type in (FFIType.FLOAT32, FFIType.FLOAT64):
            body = f"    // {fn.name}: default returns 0.0\n    return 0.0"
        elif fn.return_type is FFIType.BOOL:
            body = f"    // {fn.name}: default returns false\n    return false"
        else:
            body = f"    // {fn.name}: default returns nil\n    return nil"
        doc = f"/// {fn.description}\n" if fn.description else ""
        return (f"{doc}public func {fn.name}({params_str}){ret_part} {{\n"
                f"{body}\n}}")


__all__ = [
    "FFIType",
    "FFIFunction",
    "FFIArrayType",
    "FFISliceType",
    "FFITypeLike",
    "CHeaderEmitter",
    "RustFFIEmitter",
    "PythonFFIBindingEmitter",
    "WASMFunction",
    "WASMExport",
    "WASMModule",
    "GoFFIEmitter",
    "SwiftFFIEmitter",
]
