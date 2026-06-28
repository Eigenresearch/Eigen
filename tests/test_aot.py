import os
import sys
import subprocess
import tempfile
import pytest
from contextlib import contextmanager

from src.aot.compiler import AOTCompiler

@contextmanager
def capture_fd1():
    # Save standard output file descriptor
    original_stdout_fd = os.dup(1)
    
    # Create temp file
    temp_file = tempfile.TemporaryFile(mode='w+b')
    
    # Redirect FD 1 to temp file
    os.dup2(temp_file.fileno(), 1)
    
    try:
        yield temp_file
    finally:
        # Restore FD 1
        os.dup2(original_stdout_fd, 1)
        os.close(original_stdout_fd)

def run_aot_compile(code: str, safe_mode: bool = False, seed: int = 0):
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
        
    try:
        aot = AOTCompiler(safe_mode=safe_mode)
        exe_path = aot.compile(f_path, os.getcwd(), optimize=True, seed=seed)
        res = subprocess.run([exe_path], capture_output=True, text=True)
        return res.returncode, res.stdout, res.stderr
    finally:
        try:
            os.remove(f_path)
            # Remove exe if exists
            exe_ext = ".exe" if sys.platform == "win32" else ""
            exe_to_remove = f_path.rsplit('.', 1)[0] + exe_ext
            if os.path.exists(exe_to_remove):
                os.remove(exe_to_remove)
        except Exception:
            pass

def run_aot_jit(code: str, seed: int = 0):
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
        
    try:
        aot = AOTCompiler()
        with capture_fd1() as temp_f:
            aot.jit_execute(f_path, os.getcwd(), seed=seed)
            sys.stdout.flush()
        temp_f.seek(0)
        output = temp_f.read().decode('utf-8')
        return output
    finally:
        try:
            os.remove(f_path)
        except Exception:
            pass


def test_aot_arithmetic():
    code = """
    eigen 1.0
    let a: int = 10
    let b: int = 20
    let c: int = a + b * 2
    let d: float = 1.5
    let e: float = d * 2.0
    print c
    print e
    """
    code_val, out, err = run_aot_compile(code)
    assert code_val == 0
    assert out.strip().split() == ["50", "3.0"]

def test_aot_bool_print():
    code = """
    eigen 1.0
    let t: bool = true
    let f: bool = false
    print t
    print f
    """
    code_val, out, err = run_aot_compile(code)
    assert code_val == 0
    assert out.strip().split() == ["True", "False"]

def test_aot_negative_division():
    # Python floor division: -7 // 2 = -4
    code = """
    eigen 1.0
    let a: int = -7
    let b: int = 2
    let res: int = a / b
    print res
    """
    code_val, out, err = run_aot_compile(code)
    assert code_val == 0
    assert out.strip() == "-4"

def test_aot_division_by_zero():
    code = """
    eigen 1.0
    let a: int = 5
    let b: int = 0
    let c: int = a / b
    print c
    """
    code_val, out, err = run_aot_compile(code)
    assert code_val != 0
    assert "ZeroDivisionError: division by zero" in err

def test_aot_division_overflow_wrapping():
    # INT_MIN / -1 should wrap to INT_MIN in default mode
    code = """
    eigen 1.0
    let a: int = -9223372036854775808
    let b: int = -1
    let c: int = a / b
    print c
    """
    code_val, out, err = run_aot_compile(code, safe_mode=False)
    assert code_val == 0
    assert out.strip() == "-9223372036854775808"

def test_aot_division_overflow_safe_mode():
    # INT_MIN / -1 should trap in safe mode
    code = """
    eigen 1.0
    let a: int = -9223372036854775808
    let b: int = -1
    let c: int = a / b
    print c
    """
    code_val, out, err = run_aot_compile(code, safe_mode=True)
    # Exits with crash
    assert code_val != 0

def test_aot_functions_call_ret():
    code = """
    eigen 1.0
    func add(x: int, y: int) -> int {
        let sum: int = x + y
        return sum
    }
    let res: int = add(5, 7)
    print res
    """
    code_val, out, err = run_aot_compile(code)
    assert code_val == 0
    assert out.strip() == "12"

def test_aot_qfunc_call():
    code = """
    eigen 1.0
    qfunc prepare(qubit q) {
        H q
        return
    }
    qubit q0
    prepare(q0)
    cbit c0
    measure q0 -> c0
    print c0
    """
    code_val, out, err = run_aot_compile(code, seed=42)
    assert code_val == 0
    assert out.strip() in ("True", "False")

def test_aot_seed_determinism():
    code = """
    eigen 1.0
    qubit q
    H q
    cbit c
    measure q -> c
    print c
    """
    # Seed 42 should always yield the exact same outcome
    _, out1, _ = run_aot_compile(code, seed=42)
    _, out2, _ = run_aot_compile(code, seed=42)
    _, out3, _ = run_aot_compile(code, seed=42)
    assert out1 == out2 == out3

def test_aot_jit_execute():
    code = """
    eigen 1.0
    let val: int = 15 + 25
    print val
    """
    output = run_aot_jit(code)
    assert output.strip() == "40"

def test_aot_vs_vm_corpus():
    # Test on one of the standard examples, e.g. coin_flip.eig
    # We run it via JIT (AOT) and verify it executes without error
    aot = AOTCompiler()
    # It should JIT run coin_flip.eig
    example_path = os.path.join("examples", "coin_flip.eig")
    if os.path.exists(example_path):
        with capture_fd1() as temp_f:
            aot.jit_execute(example_path, os.getcwd(), seed=42)
            sys.stdout.flush()
        temp_f.seek(0)
        out = temp_f.read().decode('utf-8')
        assert "State" in out or "True" in out or "False" in out or len(out) > 0

def run_vm(code: str):
    from src.frontend.lexer import Lexer
    from src.frontend.parser import Parser
    from src.semantic.import_resolver import ImportResolver
    from src.semantic.type_checker import TypeChecker
    from src.backend.ebc_compiler import EBCCompiler
    from src.backend.vm import EigenVM
    
    lexer = Lexer(code)
    parser = Parser(lexer.tokenize())
    ast = parser.parse()
    resolver = ImportResolver(os.getcwd())
    ast = resolver.resolve(ast)
    type_checker = TypeChecker()
    type_checker.check(ast)
    compiler = EBCCompiler()
    instrs = compiler.compile_ast(ast)
    
    import io
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        vm = EigenVM(seed=42)
        vm.execute(instrs)
    finally:
        sys.stdout = old_stdout
    return new_stdout.getvalue()

def test_aot_control_flow():
    code = """
    eigen 1.0
    let sum: int = 0
    let i: int = 0
    while i < 10 {
        sum = sum + i
        i = i + 1
    }
    print sum
    """
    code_val, out, err = run_aot_compile(code)
    assert code_val == 0
    assert out.strip() == "45"

def test_aot_quantum_bell():
    code = """
    eigen 1.0
    qubit q0
    qubit q1
    let count_00: int = 0
    let count_11: int = 0
    let i: int = 0
    while i < 100 {
        H q0
        CNOT q0, q1
        cbit c0
        cbit c1
        measure q0 -> c0
        measure q1 -> c1
        if c0 == c1 {
            if c0 == 0 {
                count_00 = count_00 + 1
            }
            if c0 == 1 {
                count_11 = count_11 + 1
            }
        }
        if c0 == 1 {
            X q0
        }
        if c1 == 1 {
            X q1
        }
        i = i + 1
    }
    print count_00
    print count_11
    """
    code_val, out, err = run_aot_compile(code, seed=42)
    assert code_val == 0
    lines = out.strip().split()
    assert len(lines) == 2
    c00 = int(lines[0])
    c11 = int(lines[1])
    assert c00 + c11 == 100
    assert c00 > 30
    assert c11 > 30

def test_aot_quantum_grover():
    code = """
    eigen 1.0
    qubit q0
    qubit q1
    H q0
    H q1
    CZ q0, q1
    H q0
    H q1
    X q0
    X q1
    CZ q0, q1
    X q0
    X q1
    H q0
    H q1
    cbit c0
    cbit c1
    measure q0 -> c0
    measure q1 -> c1
    print c0
    print c1
    """
    code_val, out, err = run_aot_compile(code, seed=42)
    assert code_val == 0
    lines = out.strip().split()
    assert lines == ["True", "True"]

def test_aot_unsupported_construct():
    code = """
    eigen 1.0
    struct Point {
        x: int
        y: int
    }
    let p: Point = Point { x: 1, y: 2 }
    print p.x
    """
    aot = AOTCompiler()
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
    try:
        with pytest.raises(TypeError) as excinfo:
            aot.compile(f_path, os.getcwd(), optimize=True, seed=0)
        assert "use --vm" in str(excinfo.value)
    finally:
        try:
            os.remove(f_path)
        except Exception:
            pass

def test_aot_perf_fib22():
    import time
    code = """
    eigen 1.0
    func fib(n: int) -> int {
        if n < 2 {
            return n
        }
        return fib(n - 1) + fib(n - 2)
    }
    let res: int = fib(22)
    print res
    """
    t0 = time.perf_counter()
    vm_out = run_vm(code)
    t_vm = time.perf_counter() - t0
    
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
    try:
        aot = AOTCompiler()
        exe_path = aot.compile(f_path, os.getcwd(), optimize=True, seed=42)
        
        t0 = time.perf_counter()
        res = subprocess.run([exe_path], capture_output=True, text=True)
        t_aot = time.perf_counter() - t0
        
        assert res.returncode == 0
        
        # Strip VM format
        vm_val = vm_out.replace("[PRINT DIRECTIVE] ", "").strip()
        assert res.stdout.strip() == vm_val
        
        speedup = t_vm / max(t_aot, 1e-6)
        print(f"VM: {t_vm:.4f}s, AOT: {t_aot:.4f}s, Speedup: {speedup:.1f}x")
        assert speedup >= 5.0
    finally:
        try:
            os.remove(f_path)
            exe_ext = ".exe" if sys.platform == "win32" else ""
            exe_to_remove = f_path.rsplit('.', 1)[0] + exe_ext
            if os.path.exists(exe_to_remove):
                os.remove(exe_to_remove)
        except Exception:
            pass

def test_aot_perf_quantum():
    import time
    code = """
    eigen 1.0
    qubit q0
    qubit q1
    let i: int = 0
    while i < 500 {
        H q0
        CNOT q0, q1
        cbit c0
        cbit c1
        measure q0 -> c0
        measure q1 -> c1
        if c0 == 1 { X q0 }
        if c1 == 1 { X q1 }
        i = i + 1
    }
    """
    t0 = time.perf_counter()
    run_vm(code)
    t_vm = time.perf_counter() - t0
    
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
    try:
        aot = AOTCompiler()
        exe_path = aot.compile(f_path, os.getcwd(), optimize=True, seed=42)
        
        t0 = time.perf_counter()
        res = subprocess.run([exe_path], capture_output=True, text=True)
        t_aot = time.perf_counter() - t0
        
        assert res.returncode == 0
        speedup = t_vm / max(t_aot, 1e-6)
        print(f"VM Quantum: {t_vm:.4f}s, AOT Quantum: {t_aot:.4f}s, Speedup: {speedup:.1f}x")
        assert speedup >= 0.5
    finally:
        try:
            os.remove(f_path)
            exe_ext = ".exe" if sys.platform == "win32" else ""
            exe_to_remove = f_path.rsplit('.', 1)[0] + exe_ext
            if os.path.exists(exe_to_remove):
                os.remove(exe_to_remove)
        except Exception:
            pass

def test_aot_perf_hybrid():
    import time
    code = """
    eigen 1.0
    let sum: int = 0
    let i: int = 0
    qubit q
    while i < 3000 {
        sum = sum + i
        H q
        cbit c
        measure q -> c
        if c == 1 { X q }
        i = i + 1
    }
    print sum
    """
    t0 = time.perf_counter()
    vm_out = run_vm(code)
    t_vm = time.perf_counter() - t0
    
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
    try:
        aot = AOTCompiler()
        exe_path = aot.compile(f_path, os.getcwd(), optimize=True, seed=42)
        
        t0 = time.perf_counter()
        res = subprocess.run([exe_path], capture_output=True, text=True)
        t_aot = time.perf_counter() - t0
        
        assert res.returncode == 0
        
        # Strip VM format
        vm_val = vm_out.replace("[PRINT DIRECTIVE] ", "").strip()
        assert res.stdout.strip() == vm_val
        
        speedup = t_vm / max(t_aot, 1e-6)
        print(f"VM Hybrid: {t_vm:.4f}s, AOT Hybrid: {t_aot:.4f}s, Speedup: {speedup:.1f}x")
        assert speedup >= 1.0
    finally:
        try:
            os.remove(f_path)
            exe_ext = ".exe" if sys.platform == "win32" else ""
            exe_to_remove = f_path.rsplit('.', 1)[0] + exe_ext
            if os.path.exists(exe_to_remove):
                os.remove(exe_to_remove)
        except Exception:
            pass


def test_aot_optimization_params():
    code = """
    eigen 1.0
    let a: int = 42
    print a
    """
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
    try:
        aot = AOTCompiler()
        exe_path = aot.compile(f_path, os.getcwd(), optimize=True, seed=0, opt_level=3, lto=True, strip=True)
        res = subprocess.run([exe_path], capture_output=True, text=True)
        assert res.returncode == 0
        assert res.stdout.strip() == "42"
    finally:
        try:
            os.remove(f_path)
            exe_ext = ".exe" if sys.platform == "win32" else ""
            exe_to_remove = f_path.rsplit('.', 1)[0] + exe_ext
            if os.path.exists(exe_to_remove):
                os.remove(exe_to_remove)
        except Exception:
            pass


def test_aot_qir_emission():
    code = """
    eigen 1.0
    qubit q0
    H q0
    cbit c0
    measure q0 -> c0
    """
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
    try:
        aot = AOTCompiler()
        llvm_module = aot._compile_to_llvm_module(f_path, os.getcwd(), optimize=True, seed=0, emit_qir=True)
        ir_str = str(llvm_module)
        assert "__quantum__rt__qubit_allocate" in ir_str
        assert "__quantum__qis__h__body" in ir_str
        assert "__quantum__qis__mz__body" in ir_str
        assert "__quantum__rt__result_equal" in ir_str
    finally:
        try:
            os.remove(f_path)
        except Exception:
            pass


def test_aot_qft_smoke():
    code = """
    eigen 1.0
    qubit q0
    qubit q1
    qubit q2
    H q0
    CNOT q0, q1
    RZ q1, 1.57079632679
    H q1
    CNOT q1, q2
    RZ q2, 0.78539816339
    H q2
    cbit c0
    cbit c1
    cbit c2
    measure q0 -> c0
    measure q1 -> c1
    measure q2 -> c2
    print c0
    print c1
    print c2
    """
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
    try:
        aot = AOTCompiler()
        exe_path = aot.compile(f_path, os.getcwd(), optimize=True, seed=42)
        res = subprocess.run([exe_path], capture_output=True, text=True)
        assert res.returncode == 0
        lines = res.stdout.strip().split()
        assert len(lines) == 3
        for l in lines:
            assert l in ("True", "False")
    finally:
        try:
            os.remove(f_path)
            exe_ext = ".exe" if sys.platform == "win32" else ""
            exe_to_remove = f_path.rsplit('.', 1)[0] + exe_ext
            if os.path.exists(exe_to_remove):
                os.remove(exe_to_remove)
        except Exception:
            pass
