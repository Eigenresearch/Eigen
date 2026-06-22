import sys
import os

# Adjust sys.path to include the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.lexer import Lexer
from src.parser import Parser
from src.import_resolver import ImportResolver
from src.type_checker import TypeChecker, TypeErrorException
from src.ir_converter import EQIRConverter
from src.optimizer import EQIROptimizer
from src.runtime import EigenRuntime
from src.profiler import EQIRProfiler
from src.equivalence import EquivalenceChecker

def get_workspace_root() -> str:
    # Use current working directory as default workspace root
    return os.getcwd()

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

    # 5. EQIR v1 Generation
    converter = EQIRConverter()
    graph = converter.convert(ast)
    
    return graph

def main():
    parser = argparse.ArgumentParser(description="Eigen Language Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Compile and execute an Eigen (.eig) file")
    run_parser.add_argument("file", help="Path to the .eig source file")
    run_parser.add_argument("--optimize", action="store_true", help="Enable EQIR v1 DAG optimization pass")
    run_parser.add_argument("--trace", action="store_true", help="Enable execution step tracing")

    # Verify-equiv command
    equiv_parser = subparsers.add_parser("verify-equiv", help="Verify if two Eigen programs are mathematically equivalent")
    equiv_parser.add_argument("file1", help="Path to first .eig file")
    equiv_parser.add_argument("file2", help="Path to second .eig file")
    equiv_parser.add_argument("--optimize", action="store_true", help="Enable optimization before verification")

    args = parser.parse_args()
    workspace_root = get_workspace_root()

    if args.command == "run":
        # Compile to EQIR v1
        graph = compile_to_eqir(args.file, workspace_root)
        
        # Optimize if requested
        if args.optimize:
            optimizer = EQIROptimizer()
            graph = optimizer.optimize(graph)
            print(f"EQIR v1 Optimizer: Performed {optimizer.optimizations_count} optimization rewrites.")
            
        # Profile & Run
        profiler = EQIRProfiler()
        profiler.start()
        
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
        
        # Output profile report
        stats = profiler.profile(graph)
        profiler.print_profile_report(stats)

    elif args.command == "verify-equiv":
        graph1 = compile_to_eqir(args.file1, workspace_root)
        graph2 = compile_to_eqir(args.file2, workspace_root)
        
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

if __name__ == "__main__":
    main()
