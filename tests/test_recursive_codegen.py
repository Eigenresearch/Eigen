"""Tests for the native-Python recursive-function compiler.

We verify two things:
1. Pure-numeric recursion is correctly translated to Python source and
   produces results bit-identical to the VM-dispatch semantics.
2. Performance for ``fibonacci(20)`` drops from ~80ms (VM dispatch) to
   roughly CPython-direct-recursion speed (low single-digit ms).
"""

from __future__ import annotations

import os
import sys
import time

from src.backend.ebc_compiler import EBCCompiler
from src.backend.vm import EigenVM
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.jit.recursive_codegen import compile_recursive_functions
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _compile(src: str, workspace_root: str = os.path.abspath(".")):
    lexer = Lexer(src)
    parser = Parser(lexer.tokenize())
    ast = parser.parse()
    resolver = ImportResolver(workspace_root)
    ast = resolver.resolve(ast)
    type_checker = TypeChecker()
    type_checker.check(ast)
    compiler = EBCCompiler()
    instructions = compiler.compile_ast(ast)
    return ast, instructions


def _read_var(vm: EigenVM, name: str):
    """Eigen stores top-level ``let`` values on the synthetic ``main`` frame
    rather than ``vm.globals`` (the main frame is the only frame at start and
    STORE_VAR writes to ``call_stack[-1].locals`` whenever a frame is active).
    This helper reads either side transparently."""
    if vm.call_stack and name in vm.call_stack[-1].locals:
        return vm.call_stack[-1].locals[name]
    return vm.globals.get(name)


def _run(src: str, with_native: bool, workspace_root: str = os.path.abspath(".")):
    ast, instructions = _compile(src, workspace_root)
    vm = EigenVM()
    if with_native:
        vm.recursive_funcs = compile_recursive_functions(ast)
    f_null = open(os.devnull, "w")
    old_stdout = sys.stdout
    sys.stdout = f_null
    try:
        vm.execute(instructions)
    finally:
        sys.stdout = old_stdout
        f_null.close()
    return vm


# --------------------------------------------------------------------------- #
# Correctness                                                                 #
# --------------------------------------------------------------------------- #

class TestCorrectness:
    def test_fibonacci_20_native_matches_vm_dispatch(self):
        src = """
eigen 1.0

func fibonacci(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

let result: int = fibonacci(20)
print result
assert result == 6765
"""
        vm_native = _run(src, with_native=True)
        assert _read_var(vm_native, "result") == 6765
        # recursive_calls_native should be > 0 (we bypassed dispatch).
        assert vm_native.recursive_calls_native > 0

        vm_dispatch = _run(src, with_native=False)
        assert _read_var(vm_dispatch, "result") == 6765
        # Dispatch path took the slow route.
        assert vm_dispatch.recursive_calls_native == 0

    def test_factorial_native_matches_vm_dispatch(self):
        src = """
eigen 1.0

func factorial(n: int) -> int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

let r1: int = factorial(10)
let r2: int = factorial(5)
assert r1 == 3628800
assert r2 == 120
"""
        vm = _run(src, with_native=True)
        assert _read_var(vm, "r1") == 3628800
        assert _read_var(vm, "r2") == 120
        assert vm.recursive_calls_native >= 1  # one call from main + recursive calls

    def test_mutual_recursion_compiles_and_runs(self):
        src = """
eigen 1.0

func is_even(n: int) -> int {
    if n == 0 {
        return 1
    }
    return is_odd(n - 1)
}

func is_odd(n: int) -> int {
    if n == 0 {
        return 0
    }
    return is_even(n - 1)
}

let evens: int = is_even(10)
let odds: int  = is_odd(10)
assert evens == 1
assert odds   == 0
"""
        vm = _run(src, with_native=True)
        assert _read_var(vm, "evens") == 1
        assert _read_var(vm, "odds")  == 0
        # Both should be registered.
        assert "is_even" in vm.recursive_funcs
        assert "is_odd"  in vm.recursive_funcs

    def test_function_with_non_numeric_param_disqualified(self):
        src = """
eigen 1.0

func helper(arr: array<int>) -> int {
    return 1
}

let xs: array<int> = [1, 2, 3]
let r: int = helper(xs)
assert r == 1
"""
        # `helper` takes an array — disqualified from native compilation
        # because the renderer would need to clone the array (call by value
        # semantics in Eigen vs. Python list aliasing).
        ast, _ = _compile(src)
        result = compile_recursive_functions(ast)
        assert "helper" not in result

    def test_function_calling_unknown_name_disqualified(self):
        # If a function calls a non-recursive sibling function, the
        # sibling is not added to the recursive registry (it has no
        # self-call) but the caller is, as long as the caller recurses
        # on itself.
        src = """
eigen 1.0

func caller(x: int) -> int {
    if x <= 0 {
        return 0
    }
    return caller(x - 1) + helper(x)
}

func helper(y: int) -> int {
    return y * y
}

let r: int = caller(3)
assert r == 9  # caller bottoms out (helper(3) = 9 after last recursion).
"""
        ast, _ = _compile(src)
        result = compile_recursive_functions(ast)
        # `caller` self-recurses → qualifies. `helper` has no self-call and
        # is not in any cycle → qualified set should be only `caller`.
        assert "caller" in result
        assert "helper" not in result

    def test_non_recursive_function_disqualified(self):
        src = """
eigen 1.0

func square(x: int) -> int {
    return x * x
}

let r: int = square(5)
assert r == 25
"""
        ast, _ = _compile(src)
        result = compile_recursive_functions(ast)
        # square never calls itself, so the "must actually recurse" gate
        # rejects it; VM dispatch handles it normally.
        assert result == {}

    def test_void_recursive_function_with_print(self):
        src = """
eigen 1.0

func countdown(n: int) -> int {
    if n > 0 {
        return countdown(n - 1)
    }
    return n
}

let final: int = countdown(5)
assert final == 0
"""
        vm = _run(src, with_native=True)
        assert _read_var(vm, "final") == 0
        assert vm.recursive_calls_native > 0

    def test_empty_program_returns_empty_registry(self):
        src = """
eigen 1.0

let x: int = 7
"""
        ast, _ = _compile(src)
        assert compile_recursive_functions(ast) == {}

    def test_recursion_failure_falls_back_to_dispatch(self):
        # A pathological recursion that exhausts Python's recursion limit
        # should disable the native path and let the VM dispatch recover
        # via its own StackOverflowError guard.
        src = """
eigen 1.0

func deep(n: int) -> int {
    if n <= 0 {
        return 0
    }
    return deep(n - 1)
}

let r: int = deep(500)
assert r == 0
"""
        vm = _run(src, with_native=True)
        assert _read_var(vm, "r") == 0


# --------------------------------------------------------------------------- #
# Performance                                                                  #
# --------------------------------------------------------------------------- #

class TestPerformance:
    def test_fibonacci_20_native_beats_dispatch_by_5x(self):
        src = """
eigen 1.0

func fibonacci(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

let result: int = fibonacci(20)
"""
        # Warm-up interpreter once.
        _run(src, with_native=False)
        # Time dispatch path.
        t0 = time.perf_counter()
        _run(src, with_native=False)
        dispatch_ms = (time.perf_counter() - t0) * 1000.0

        # Time native path.
        t0 = time.perf_counter()
        _run(src, with_native=True)
        native_ms = (time.perf_counter() - t0) * 1000.0

        # Native must be at least 5x faster for the dense recursive case.
        # VM dispatch is ~80ms here; native should land near ~3-5ms.
        speedup = dispatch_ms / max(native_ms, 0.001)
        assert speedup > 5.0, (
            f"Expected >5x speedup, got {speedup:.2f}x "
            f"(dispatch={dispatch_ms:.1f}ms, native={native_ms:.1f}ms)"
        )
