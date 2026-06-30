import time
import random
import math
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.backend.ebc_compiler import EBCCompiler
from src.backend.vm import EigenVM
from src.simulator import PythonDenseStatevector

# Eigen implementations of the 5 benchmark tasks
EIGEN_PROGRAMS = {
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
    "Quantum Teleportation": """eigen 1.0
qubit q0
qubit q1
qubit q2
let c0: int = 0
let c1: int = 0

RY q0, 1.04719755
H q1
CNOT q1, q2
CNOT q0, q1
H q0
measure q0 -> c0
measure q1 -> c1

if c1 == 1 {
    X q2
}
if c0 == 1 {
    Z q2
}
"""
}

# Python implementations of the 5 benchmark tasks
def py_fib(n):
    if n <= 1:
        return n
    return py_fib(n - 1) + py_fib(n - 2)

def py_arith():
    sum_val = 0
    i = 0
    while i < 15000:
        sum_val = sum_val + i
        i = i + 1
    return sum_val

def py_array():
    arr = [1, 2, 3, 4, 5]
    i = 0
    while i < 8000:
        arr[0] = arr[0] + 1
        i = i + 1
    return arr

def py_exception():
    i = 0
    catches = 0
    while i < 5000:
        try:
            if i % 2 == 0:
                raise ValueError("Error")
        except ValueError:
            catches = catches + 1
        i = i + 1
    return catches

def py_teleport():
    sim = PythonDenseStatevector()
    sim.allocate_qubit()
    sim.allocate_qubit()
    sim.allocate_qubit()
    
    sim.RY(0, 1.04719755)
    sim.H(1)
    sim.CNOT(1, 2)
    sim.CNOT(0, 1)
    sim.H(0)
    
    c0 = sim.measure(0, random.random())
    c1 = sim.measure(1, random.random())
    
    if c1 == 1:
        sim.X(2)
    if c0 == 1:
        sim.Z(2)
        
    return sim.get_state_vector()

def benchmark():
    results = {}
    compiler = EBCCompiler()
    
    for name, source in EIGEN_PROGRAMS.items():
        print(f"Compiling and running Eigen {name}...")
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        instrs = compiler.compile_ast(ast)
        
        # Benchmark Eigen VM
        times_eigen = []
        for _ in range(30):
            vm = EigenVM()
            vm.jit_enabled = True  # Run with JIT enabled
            start = time.perf_counter()
            vm.execute(instrs)
            times_eigen.append(time.perf_counter() - start)
            
        # Benchmark Native Python
        times_python = []
        for _ in range(30):
            start = time.perf_counter()
            if name == "Fibonacci Recursion":
                py_fib(18)
            elif name == "Arithmetic Loop":
                py_arith()
            elif name == "Array Operations":
                py_array()
            elif name == "Exception Try-Catch":
                py_exception()
            elif name == "Quantum Teleportation":
                py_teleport()
            times_python.append(time.perf_counter() - start)
            
        avg_eig = sum(times_eigen) / len(times_eigen) * 1000.0   # in ms
        avg_py = sum(times_python) / len(times_python) * 1000.0   # in ms
        slowdown = avg_eig / avg_py if avg_py > 0 else 1.0
        
        results[name] = {
            "avg_py": avg_py,
            "avg_eig": avg_eig,
            "slowdown": slowdown
        }
        print(f"{name} -> Python: {avg_py:.3f} ms | Eigen VM: {avg_eig:.3f} ms | Ratio: {slowdown:.2f}x")

    # Generate Markdown Table
    with open("python_vs_eigen.md", "w", encoding="utf-8") as f:
        f.write("# Language Benchmark: Native Python vs Eigen VM (Nova 2.6)\n\n")
        f.write("Comparison of execution time between native Python execution and Eigen VM (with JIT compilation enabled) across 30 runs.\n\n")
        f.write("| Benchmark Test Case | Native Python (ms) | Eigen VM (ms) | Slowdown / Ratio |\n")
        f.write("| --- | --- | --- | --- |\n")
        for name, data in sorted(results.items()):
            f.write(f"| {name} | {data['avg_py']:.3f} ms | {data['avg_eig']:.3f} ms | **{data['slowdown']:.2f}x** |\n")
            
    print("Benchmark complete. Results saved in python_vs_eigen.md")

if __name__ == "__main__":
    benchmark()
