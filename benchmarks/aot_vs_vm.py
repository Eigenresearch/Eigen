import os
import sys
import time
import subprocess
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aot.compiler import AOTCompiler

# VM execution helper
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
    
    # Time execution
    vm = EigenVM(seed=42)
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        t0 = time.perf_counter()
        vm.execute(instrs)
        t_vm = time.perf_counter() - t0
    finally:
        sys.stdout = old_stdout
    return t_vm

# AOT execution helper
def run_aot(code: str):
    with tempfile.NamedTemporaryFile(suffix=".eig", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        f_path = f.name
    try:
        aot = AOTCompiler()
        exe_path = aot.compile(f_path, os.getcwd(), optimize=True, seed=42)
        
        # Measure only execution time
        t0 = time.perf_counter()
        subprocess.run([exe_path], capture_output=True)
        t_aot = time.perf_counter() - t0
        return t_aot
    finally:
        try:
            os.remove(f_path)
            exe_ext = ".exe" if sys.platform == "win32" else ""
            exe_to_remove = f_path.rsplit('.', 1)[0] + exe_ext
            if os.path.exists(exe_to_remove):
                os.remove(exe_to_remove)
        except Exception:
            pass

# Benchmarks
programs = {
    "fib(22) classical": """
    eigen 1.0
    func fib(n: int) -> int {
        if n < 2 {
            return n
        }
        return fib(n - 1) + fib(n - 2)
    }
    let res: int = fib(22)
    print res
    """,
    "Bell pair (500 shots)": """
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
    """,
    "Grover 2-qubit (500 iter)": """
    eigen 1.0
    qubit q0
    qubit q1
    let i: int = 0
    while i < 500 {
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
        if c0 == 1 { X q0 }
        if c1 == 1 { X q1 }
        i = i + 1
    }
    """,
    "factorial(12) x 10000 classical": """
    eigen 1.0
    func fact(n: int) -> int {
        if n <= 1 {
            return 1
        }
        return n * fact(n - 1)
    }
    let i: int = 0
    let res: int = 0
    while i < 10000 {
        res = fact(12)
        i = i + 1
    }
    print res
    """
}

def main():
    print("Running benchmarks...")
    results = []
    for name, code in programs.items():
        print(f"Benchmarking {name}...")
        t_vm = run_vm(code)
        t_aot = run_aot(code)
        speedup = t_vm / max(t_aot, 1e-6)
        results.append((name, t_vm, t_aot, speedup))

    print("\n| Program | VM (ms) | AOT (ms) | Speedup |")
    print("|---|---|---|---|")
    for name, t_vm, t_aot, speedup in results:
        print(f"| {name} | {t_vm*1000:.2f} ms | {t_aot*1000:.2f} ms | {speedup:.2f}x |")

if __name__ == "__main__":
    main()
