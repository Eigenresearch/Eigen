import sys
import os
import time
import json
import io
import contextlib
from src.cli import register_command
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker
from src.backend.ebc_compiler import EBCCompiler
from src.backend.vm import EigenVM

def run_frontend_benchmarks():
    print("=" * 70)
    print("         EIGEN FRONTEND PERFORMANCE BENCHMARK (Python vs Rust)         ")
    print("=" * 70)
    print(f"{'Lines':<10} | {'Python Time (ms)':<18} | {'Rust Time (ms)':<16} | {'Speedup':<12}")
    print("-" * 70)

    sizes = [("1k", 1000), ("10k", 10000), ("100k", 100000)]
    for name, lines_count in sizes:
        # Generate dummy content
        content = "eigen 1.0\n" + "\n".join(f"let a{i}: int = 5" for i in range(lines_count - 1))
        
        # Bench Python
        from src.frontend.lexer import Lexer as PythonLexer
        from src.frontend.parser import Parser as PythonParser
        
        t0 = time.perf_counter()
        lexer = PythonLexer(content)
        tokens = lexer.tokenize()
        if hasattr(tokens, "source"):
            tokens.source = None
        parser = PythonParser(tokens)
        ast = parser.parse()
        python_time = (time.perf_counter() - t0) * 1000.0

        # Bench Rust
        import eigen_native
        t0 = time.perf_counter()
        ast_rust = eigen_native.parse_native(content)
        rust_time = (time.perf_counter() - t0) * 1000.0
        
        speedup = python_time / rust_time
        print(f"{name:<10} | {python_time:<18.2f} | {rust_time:<16.2f} | {speedup:.1f}x")
        
    print("=" * 70)

@register_command("bench")
def bench_command(args, workspace_root):
    if getattr(args, "frontend", False):
        run_frontend_benchmarks()
        return

    bench_dir = os.path.join(workspace_root, "benchmarks")
    if not os.path.isdir(bench_dir):
        print(f"Error: Benchmarks directory '{bench_dir}' not found.", file=sys.stderr)
        sys.exit(1)
        
    files = [f for f in os.listdir(bench_dir) if f.endswith('.eig')]
    if not files:
        print("No .eig benchmark files found in benchmarks/ directory.")
        return
        
    print("=" * 65)
    print("                 EIGEN BENCHMARK SUITE                 ")
    print("=" * 65)
    print(f"{'Benchmark Name':<20} | {'Compile (ms)':<12} | {'Exec (ms)':<12} | {'Total (ms)':<12}")
    print("-" * 65)
    
    current_results = {}
    for f_name in sorted(files):
        f_path = os.path.join(bench_dir, f_name)
        
        t_comp_start = time.perf_counter()
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                content = f.read()
            lexer = Lexer(content)
            parser = Parser(lexer.tokenize())
            ast = parser.parse()
            resolver = ImportResolver(workspace_root)
            ast = resolver.resolve(ast)
            type_checker = TypeChecker()
            type_checker.check(ast)
            compiler = EBCCompiler()
            instrs = compiler.compile_ast(ast)
            compile_time_ms = (time.perf_counter() - t_comp_start) * 1000.0
        except Exception as e:
            print(f"{f_name:<20} | {'ERROR':<12} | {'-':<12} | {'-':<12} ({e})")
            continue
            
        vm = EigenVM(trace_mode=False)
        t_exec_start = time.perf_counter()
        try:
            f_null = io.StringIO()
            with contextlib.redirect_stdout(f_null):
                vm.execute(instrs)
            exec_time_ms = (time.perf_counter() - t_exec_start) * 1000.0
            total_time_ms = compile_time_ms + exec_time_ms
            print(f"{f_name:<20} | {compile_time_ms:<12.3f} | {exec_time_ms:<12.3f} | {total_time_ms:<12.3f}")
            current_results[f_name] = total_time_ms
        except Exception as e:
            print(f"{f_name:<20} | {compile_time_ms:<12.3f} | {'EXEC ERROR':<12} | {'-':<12} ({e})")
            
    print("=" * 65)
    
    results_json_path = os.path.join(bench_dir, "results.json")
    if os.path.isfile(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                prev_results = json.load(f)
            
            print("\nRegression Tracking Delta Summary:")
            for b_name, curr_time in current_results.items():
                if b_name in prev_results:
                    prev_time = prev_results[b_name]
                    delta = ((curr_time - prev_time) / prev_time) * 100.0
                    delta_str = f"{delta:+.1f}%"
                    if delta > 10.0:
                        print(f"  WARNING: {b_name} had {delta_str} slowdown (Prev: {prev_time:.2f}ms, Curr: {curr_time:.2f}ms)")
                    else:
                        print(f"  {b_name:<20}: {delta_str}")
        except Exception as e:
            print(f"Failed to check regressions: {e}")
            
    with open(results_json_path, 'w') as f:
        json.dump(current_results, f, indent=2)
    print(f"\nSaved benchmark results to {results_json_path}")
