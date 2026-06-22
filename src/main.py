import sys
import os
import argparse
import json

# Remove the script's directory (src/) from sys.path to avoid shadowing standard modules (like 'ast')
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

# Adjust sys.path to include the project root
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "..")))

from src.lexer import Lexer, TokenType, Token
from src.parser import Parser
from src.import_resolver import ImportResolver
from src.type_checker import TypeChecker, TypeErrorException
from src.ir_converter import EQIRConverter
from src.optimizer import EQIROptimizer
from src.runtime import EigenRuntime
from src.profiler import EQIRProfiler
from src.equivalence import EquivalenceChecker
from src.ebc_compiler import EBCCompiler
from src.vm import EigenVM
from src.bytecode import Instruction

import hashlib
from src.ir_graph import EQIRGraph

def get_workspace_root() -> str:
    return os.getcwd()

def get_project_hash(filepath: str, workspace_root: str) -> str:
    visited_files = set()
    files_to_process = [os.path.abspath(filepath)]
    stdlib_root = os.path.join(workspace_root, "stdlib")
    hasher = hashlib.sha256()
    processed_contents = []
    
    while files_to_process:
        current_path = files_to_process.pop(0)
        if current_path in visited_files:
            continue
        visited_files.add(current_path)
        
        if not os.path.isfile(current_path):
            continue
            
        with open(current_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        processed_contents.append((current_path, content))
        
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("import "):
                parts = line.split()
                if len(parts) >= 2:
                    module_name = parts[1]
                    relative_path = module_name.replace('.', '/') + ".eig"
                    local_path = os.path.join(workspace_root, relative_path)
                    if os.path.isfile(local_path):
                        files_to_process.append(os.path.abspath(local_path))
                    else:
                        stdlib_path = os.path.join(stdlib_root, relative_path)
                        if os.path.isfile(stdlib_path):
                            files_to_process.append(os.path.abspath(stdlib_path))
                            
    processed_contents.sort(key=lambda x: x[0])
    for path, content in processed_contents:
        hasher.update(path.encode('utf-8'))
        hasher.update(content.encode('utf-8'))
        
    return hasher.hexdigest()

def load_from_cache(filepath: str, workspace_root: str, cache_type: str):
    try:
        proj_hash = get_project_hash(filepath, workspace_root)
        cache_dir = os.path.join(workspace_root, ".eigen_cache")
        cache_path = os.path.join(cache_dir, f"{proj_hash}.{cache_type}")
        if os.path.isfile(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if cache_type == "ebc":
                return [Instruction.from_dict(d) for d in data]
            elif cache_type == "eqir":
                return EQIRGraph.from_dict(data)
    except Exception:
        pass
    return None

def save_to_cache(filepath: str, workspace_root: str, cache_type: str, obj):
    try:
        proj_hash = get_project_hash(filepath, workspace_root)
        cache_dir = os.path.join(workspace_root, ".eigen_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{proj_hash}.{cache_type}")
        if cache_type == "ebc":
            data = [inst.to_dict() for inst in obj]
        elif cache_type == "eqir":
            data = obj.to_dict()
        else:
            return
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def compile_to_eqir(filepath: str, workspace_root: str) -> tuple:
    if not os.path.isfile(filepath):
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Lexer
    lexer = Lexer(content)
    tokens = lexer.tokenize()

    # 2. Parser
    parser = Parser(tokens)
    ast = parser.parse()

    # 3. Import Resolver
    resolver = ImportResolver(workspace_root)
    ast = resolver.resolve(ast)

    # 4. Type Checker
    type_checker = TypeChecker()
    try:
        type_checker.check(ast)
    except TypeErrorException as e:
        print(f"Type Verification Failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    # 5. EQIR v1.1 Generation
    converter = EQIRConverter()
    graph = converter.convert(ast)
    
    return graph, ast

def main():
    parser = argparse.ArgumentParser(description="Eigen Language Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Compile and execute an Eigen (.eig) or bytecode (.ebc) file")
    run_parser.add_argument("file", help="Path to the source file")
    run_parser.add_argument("--optimize", action="store_true", help="Enable EQIR v1.1 DAG optimization pass")
    run_parser.add_argument("--trace", action="store_true", help="Enable execution step tracing")
    run_parser.add_argument("--vm", action="store_true", help="Execute using the Eigen VM (EBC)")
    run_parser.add_argument("--backend", choices=["sim", "qiskit", "ibmq"], default="sim", help="Execution backend target")

    # Verify-equiv command
    equiv_parser = subparsers.add_parser("verify-equiv", help="Verify if two Eigen programs are mathematically equivalent")
    equiv_parser.add_argument("file1", help="Path to first .eig file")
    equiv_parser.add_argument("file2", help="Path to second .eig file")
    equiv_parser.add_argument("--optimize", action="store_true", help="Enable optimization before verification")

    # Packaging commands
    init_parser = subparsers.add_parser("init", help="Initialize a new Eigen package")
    init_parser.add_argument("name", nargs="?", help="Name of the package")

    add_parser = subparsers.add_parser("add", help="Add a dependency to eigen.toml")
    add_parser.add_argument("dependency", help="Name of the dependency")
    add_parser.add_argument("--ver", default="0.1.0", help="Dependency version")

    build_parser = subparsers.add_parser("build", help="Build current package or compile .eig to .ebc")
    build_parser.add_argument("file", nargs="?", help="Path to .eig file to compile to .ebc")

    subparsers.add_parser("publish", help="Publish package to the Eigen registry")

    # Formatting and Docs commands
    fmt_parser = subparsers.add_parser("fmt", help="Format an Eigen source file")
    fmt_parser.add_argument("file", help="Path to the file to format")

    doc_parser = subparsers.add_parser("doc", help="Generate API documentation from Eigen comments")
    doc_parser.add_argument("file", help="Path to the source file")

    # Testing command
    subparsers.add_parser("test", help="Run the Eigen unit test suite")

    # Exec command
    exec_parser = subparsers.add_parser("exec", help="Execute compiled EBC bytecode (.ebc) directly on VM")
    exec_parser.add_argument("file", help="Path to compiled EBC file")
    exec_parser.add_argument("--trace", action="store_true", help="Enable VM trace mode")

    # Bench command
    subparsers.add_parser("bench", help="Run the benchmark suite and report performance results")

    args = parser.parse_args()
    workspace_root = get_workspace_root()

    # Handle Exec command
    if args.command == "exec":
        if not args.file.endswith('.ebc'):
            print("Error: 'eigen exec' expects a compiled EBC (.ebc) file.", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.file):
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        instructions = [Instruction.from_dict(d) for d in data]
        vm = EigenVM(trace_mode=args.trace)
        try:
            vm.execute(instructions)
        except AssertionError as ae:
            print(f"Assertion Error: {ae}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"VM Execution Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Handle Bench command
    elif args.command == "bench":
        import time
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
                import io
                import contextlib
                f_null = io.StringIO()
                with contextlib.redirect_stdout(f_null):
                    vm.execute(instrs)
                exec_time_ms = (time.perf_counter() - t_exec_start) * 1000.0
                total_time_ms = compile_time_ms + exec_time_ms
                print(f"{f_name:<20} | {compile_time_ms:<12.3f} | {exec_time_ms:<12.3f} | {total_time_ms:<12.3f}")
            except Exception as e:
                print(f"{f_name:<20} | {compile_time_ms:<12.3f} | {'EXEC ERROR':<12} | {'-':<12} ({e})")
                
        print("=" * 65)
        return

    # 1. Handle Packaging init
    elif args.command == "init":
        from src.packager import EigenPackager
        packager = EigenPackager(workspace_root)
        packager.init_package(args.name)
        return

    # 2. Handle Packaging add
    elif args.command == "add":
        from src.packager import EigenPackager
        packager = EigenPackager(workspace_root)
        packager.add_dependency(args.dependency, args.ver)
        return

    # 3. Handle Packaging build
    elif args.command == "build":
        if args.file:
            print(f"Compiling '{args.file}' to EBC bytecode...")
            with open(args.file, 'r', encoding='utf-8') as f:
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
            
            out_path = args.file.rsplit('.', 1)[0] + ".ebc"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump([inst.to_dict() for inst in instrs], f, indent=2)
            print(f"Compilation successful: '{out_path}'")
        else:
            from src.packager import EigenPackager
            packager = EigenPackager(workspace_root)
            packager.build_package()
        return

    # 4. Handle Packaging publish
    elif args.command == "publish":
        from src.packager import EigenPackager
        packager = EigenPackager(workspace_root)
        packager.publish_package()
        return

    # 5. Handle Formatting
    elif args.command == "fmt":
        from src.formatter import EigenFormatter
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
        formatter = EigenFormatter()
        formatted = formatter.format_code(content)
        with open(args.file, 'w', encoding='utf-8') as f:
            f.write(formatted)
        print(f"Formatted '{args.file}' successfully.")
        return

    # 6. Handle API Docs Generation
    elif args.command == "doc":
        from src.doc_generator import EigenDocGenerator
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
        doc_gen = EigenDocGenerator()
        docs = doc_gen.generate_docs(content, args.file)
        out_path = args.file.rsplit('.', 1)[0] + "_reference.md"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(docs)
        print(f"Generated API documentation at '{out_path}'")
        return

    # 7. Handle Run test suite
    elif args.command == "test":
        import unittest
        print("Running Eigen Test Suite...")
        suite = unittest.defaultTestLoader.discover(os.path.join(os.path.dirname(__file__), "../tests"))
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    # 8. Handle Verify-equiv
    elif args.command == "verify-equiv":
        graph1, _ = compile_to_eqir(args.file1, workspace_root)
        graph2, _ = compile_to_eqir(args.file2, workspace_root)
        
        if args.optimize:
            optimizer = EQIROptimizer()
            graph1 = optimizer.optimize(graph1)
            graph2 = optimizer.optimize(graph2)
            
        checker = EquivalenceChecker()
        try:
            equivalent = checker.are_equivalent(graph1, graph2)
            print("=" * 50)
            print("          EIGEN EQUIVALENCE CHECK          ")
            print("=" * 50)
            print(f"File 1: {args.file1}")
            print(f"File 2: {args.file2}")
            if equivalent:
                print("\nResult: Mathematically EQUIVALENT (up to global phase) [SUCCESS]")
            else:
                print("\nResult: NOT EQUIVALENT [FAIL]")
            print("=" * 50)
        except Exception as e:
            print(f"Equivalence Verification Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # 9. Handle Run command (.eig or .ebc)
    # 9. Handle Run command (.eig or .ebc)
    elif args.command == "run":
        if args.file.endswith('.ebc'):
            print(f"Executing EBC bytecode file '{args.file}' on VM...")
            with open(args.file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            instructions = [Instruction.from_dict(d) for d in data]
            vm = EigenVM(trace_mode=args.trace)
            vm.execute(instructions)
            return

        # Try loading from cache
        if args.vm:
            cached_instructions = load_from_cache(args.file, workspace_root, "ebc")
            if cached_instructions is not None:
                profiler = EQIRProfiler()
                profiler.start()
                vm = EigenVM(trace_mode=args.trace)
                try:
                    vm.execute(cached_instructions)
                except AssertionError as ae:
                    print(f"Assertion Error: {ae}", file=sys.stderr)
                    sys.exit(1)
                except Exception as e:
                    print(f"VM Execution Error: {e}", file=sys.stderr)
                    sys.exit(1)
                profiler.stop()
                print("=" * 40)
                print("          EIGEN RUNTIME PROFILE (CACHED)          ")
                print("=" * 40)
                print(f"Execution Time:      {profiler.execution_time_ms:.3f} ms")
                print("=" * 40)
                return
        else:
            cached_graph = load_from_cache(args.file, workspace_root, "eqir")
            if cached_graph is not None:
                profiler = EQIRProfiler()
                profiler.start()
                runtime = EigenRuntime(trace_mode=args.trace)
                try:
                    runtime.execute(cached_graph)
                except AssertionError as ae:
                    print(f"Assertion Error: {ae}", file=sys.stderr)
                    sys.exit(1)
                except Exception as e:
                    print(f"Runtime Error: {e}", file=sys.stderr)
                    sys.exit(1)
                profiler.stop()
                stats = profiler.profile(cached_graph)
                profiler.print_profile_report(stats)
                return

        # Compile to EQIR v1.1
        graph, ast = compile_to_eqir(args.file, workspace_root)
        
        # Optimize if requested
        if args.optimize:
            optimizer = EQIROptimizer()
            graph = optimizer.optimize(graph)
            print(f"EQIR v1.1 Optimizer: Performed {optimizer.optimizations_count} optimization rewrites.")
            
        # If Qiskit transpilation is selected
        if args.backend in ("qiskit", "ibmq"):
            from src.qiskit_backend import QiskitBackend
            backend = QiskitBackend()
            qiskit_script, report = backend.transpile(graph, ast)
            out_path = args.file.rsplit('.', 1)[0] + "_qiskit.py"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(qiskit_script)
            print(f"Transpiled Qiskit script saved to '{out_path}'")
            print("\nGenerated Qiskit code:")
            print("-" * 40)
            print(qiskit_script)
            print("-" * 40)
            print("\nBackend Report:")
            print("=" * 40)
            print(report)
            print("=" * 40)
            return

        # Profile & Run on simulator (either VM or standard interpreter runtime)
        profiler = EQIRProfiler()
        profiler.start()
        
        if args.vm:
            compiler = EBCCompiler()
            instructions = compiler.compile_eqir(graph)
            # Save to cache
            save_to_cache(args.file, workspace_root, "ebc", instructions)
            vm = EigenVM(trace_mode=args.trace)
            try:
                vm.execute(instructions)
            except AssertionError as ae:
                print(f"Assertion Error: {ae}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"VM Execution Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Save to cache
            save_to_cache(args.file, workspace_root, "eqir", graph)
            runtime = EigenRuntime(trace_mode=args.trace)
            try:
                runtime.execute(graph)
            except AssertionError as ae:
                print(f"Assertion Error: {ae}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Runtime Error: {e}", file=sys.stderr)
                sys.exit(1)
            
        profiler.stop()
        stats = profiler.profile(graph)
        profiler.print_profile_report(stats)

if __name__ == "__main__":
    main()
