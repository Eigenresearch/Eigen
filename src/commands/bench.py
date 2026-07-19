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
        # TODO(F841): the parsed Python AST is meant to be compared/benchmarked
        # against the Rust parser output but is currently unused.
        _ast = parser.parse()
        python_time = (time.perf_counter() - t0) * 1000.0

        # Bench Rust
        import eigen_native
        t0 = time.perf_counter()
        # TODO(F841): the Rust-parsed AST is meant to be compared against the
        # Python parse but is currently unused.
        _ast_rust = eigen_native.parse_native(content)
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
        # Native-Python recursion fast path so benchmarks reflect the
        # production code path that the run/CLI uses, not just VM dispatch.
        try:
            from src.jit.recursive_codegen import compile_recursive_functions
            vm.recursive_funcs = compile_recursive_functions(ast)
        except Exception:
            vm.recursive_funcs = {}
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
                        print(f"  WARNING: {b_name} had {delta_str} slowdown "
                              f"(Prev: {prev_time:.2f}ms, Curr: {curr_time:.2f}ms)")
                    else:
                        print(f"  {b_name:<20}: {delta_str}")
        except Exception as e:
            print(f"Failed to check regressions: {e}")
            
    with open(results_json_path, 'w') as f:
        json.dump(current_results, f, indent=2)
    print(f"\nSaved benchmark results to {results_json_path}")

    if getattr(args, "html", False):
        _generate_html_dashboard(current_results, bench_dir)

def _generate_html_dashboard(results: dict, bench_dir: str):
    html_path = os.path.join(bench_dir, "dashboard.html")
    names = list(results.keys())
    times = list(results.values())

    bars_html = ""
    max_time = max(times) if times else 1
    for name, t in results.items():
        pct = (t / max_time) * 100
        color = "#4CAF50" if t < 50 else "#FF9800" if t < 200 else "#f44336"
        bars_html += (f'<div class="bar-row"><span class="bar-label">{name}</span>'
                      f'<div class="bar-bg"><div class="bar-fill" '
                      f'style="width:{pct:.1f}%;background:{color}"></div></div>'
                      f'<span class="bar-val">{t:.2f}ms</span></div>\n')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eigen Benchmark Dashboard</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #1e1e2e; color: #cdd6f4; }}
  h1 {{ color: #89b4fa; border-bottom: 2px solid #45475a; padding-bottom: 10px; }}
  .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
  .card {{ background: #313244; padding: 20px; border-radius: 12px; flex: 1; text-align: center; }}
  .card h2 {{ color: #fab387; margin: 0; font-size: 2em; }}
  .card p {{ color: #a6adc8; margin: 5px 0 0 0; }}
  .bar-row {{ display: flex; align-items: center; margin: 8px 0; }}
  .bar-label {{ width: 200px; font-size: 14px; }}
  .bar-bg {{ flex: 1; background: #45475a; border-radius: 4px; height: 24px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; }}
  .bar-val {{ width: 80px; text-align: right; font-size: 13px; color: #a6adc8; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th, td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid #45475a; }}
  th {{ color: #89b4fa; }}
  .footer {{ margin-top: 30px; color: #6c7086; font-size: 12px; }}
</style>
</head>
<body>
<h1>Eigen 2.8 Mars — Benchmark Dashboard</h1>
<div class="summary">
  <div class="card"><h2>{len(names)}</h2><p>Benchmarks Run</p></div>
  <div class="card"><h2>{sum(times)/len(times):.1f}ms</h2><p>Average Time</p></div>
  <div class="card"><h2>{max(times):.1f}ms</h2><p>Slowest</p></div>
  <div class="card"><h2>{min(times):.1f}ms</h2><p>Fastest</p></div>
</div>
<h2>Performance Breakdown</h2>
{bars_html}
<h2>Detailed Results</h2>
<table>
  <tr><th>Benchmark</th><th>Total Time (ms)</th></tr>
  {''.join(f'<tr><td>{n}</td><td>{t:.3f}</td></tr>' for n, t in results.items())}
</table>
<div class="footer">Generated by Eigen 2.8 Mars — {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML dashboard saved to {html_path}")
