import time
import math
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.backend.ebc_compiler import EBCCompiler
from src.backend.vm import EigenVM

PROGRAMS = {
    "Fibonacci Recursion": """eigen 1.0
func fib(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}
let res: int = fib(18)
""",
    "Arithmetic Loop": """eigen 1.0
let sum: int = 0
let i: int = 0
while i < 15000 {
    sum = sum + i
    i = i + 1
}
""",
    "Array Operations": """eigen 1.0
let arr: array = [1, 2, 3, 4, 5]
let i: int = 0
while i < 8000 {
    arr[0] = arr[0] + 1
    i = i + 1
}
""",
    "Exception Try-Catch": """eigen 1.0
let i: int = 0
let catches: int = 0
while i < 5000 {
    try {
        if i % 2 == 0 {
            throw "Error"
        }
    } catch {
        catches = catches + 1
    }
    i = i + 1
}
""",
    "Quantum CNOT Circuit": """eigen 1.0
qubit q0
qubit q1
let i: int = 0
while i < 3000 {
    H q0
    CNOT q0, q1
    i = i + 1
}
"""
}

def benchmark():
    results = {}
    compiler = EBCCompiler()
    
    for name, source in PROGRAMS.items():
        print(f"Compiling {name}...")
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        instrs = compiler.compile_ast(ast)
        
        # Benchmark Pure Python VM
        times_python = []
        for _ in range(50):
            vm = EigenVM()
            vm.jit_enabled = False
            start = time.perf_counter()
            vm.execute(instrs)
            times_python.append(time.perf_counter() - start)
            
        # Benchmark JIT VM
        times_jit = []
        for _ in range(50):
            vm = EigenVM()
            vm.jit_enabled = True
            start = time.perf_counter()
            vm.execute(instrs)
            times_jit.append(time.perf_counter() - start)
            
        avg_py = sum(times_python) / len(times_python) * 1000.0
        avg_jit = sum(times_jit) / len(times_jit) * 1000.0
        speedup = avg_py / avg_jit if avg_jit > 0 else 1.0
        
        results[name] = {
            "avg_py": avg_py,
            "avg_jit": avg_jit,
            "speedup": speedup
        }
        print(f"{name} -> Python: {avg_py:.3f} ms | JIT: {avg_jit:.3f} ms | Speedup: {speedup:.2f}x")

    with open("benchmark_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Eigen 2.6 «Nova» Execution Benchmarks\n\n")
        f.write("Comparison of execution time between Pure Python VM execution (JIT Disabled) and JIT-compiled VM execution (JIT Enabled) across 50 runs.\n\n")
        f.write("| Benchmark Test Case | Pure Python VM (ms) | JIT-Enabled VM (ms) | Speedup Factor |\n")
        f.write("| --- | --- | --- | --- |\n")
        for name, data in results.items():
            f.write(f"| {name} | {data['avg_py']:.3f} ms | {data['avg_jit']:.3f} ms | **{data['speedup']:.2f}x** |\n")
            
    print("Benchmark complete. Results saved in benchmark_comparison.md")

if __name__ == "__main__":
    benchmark()
