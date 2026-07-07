import sys
import os
import time
import tracemalloc
from src.cli import register_command
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker, TypeErrorException
from src.ir.ir_converter import EQIRConverter
from src.ir.optimizer import EQIROptimizer
from src.backend.ebc_compiler import EBCCompiler
from src.backend.vm import EigenVM
from src.crash_report import write_crash_report

@register_command("profile")
def profile_command(args, workspace_root):
    tracemalloc.start()
    t_start = time.perf_counter()
    
    t0 = time.perf_counter()
    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(args.file, 'r', encoding='utf-8') as f:
        content = f.read()
    lexer = Lexer(content)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    resolver = ImportResolver(workspace_root)
    ast = resolver.resolve(ast)
    t_parse_resolve = (time.perf_counter() - t0) * 1000.0
    
    t0 = time.perf_counter()
    type_checker = TypeChecker()
    try:
        type_checker.check(ast)
    except TypeErrorException as e:
        print(f"Type Verification Failed:\n{e}", file=sys.stderr)
        sys.exit(1)
    t_typecheck = (time.perf_counter() - t0) * 1000.0
    
    t0 = time.perf_counter()
    from src.compiler import compile_to_eqir
    graph, ast = compile_to_eqir(args.file, workspace_root)
    t_eqir = (time.perf_counter() - t0) * 1000.0
    
    t0 = time.perf_counter()
    # Note: EQIROptimizer is already integrated into the compiler/optimizations, or can be run here
    optimizer = EQIROptimizer()
    graph = optimizer.optimize(graph)
    t_opt = (time.perf_counter() - t0) * 1000.0
    
    t0 = time.perf_counter()
    compiler = EBCCompiler()
    instrs = compiler.compile_ast(ast)
    t_ebc_compile = (time.perf_counter() - t0) * 1000.0
    
    vm = EigenVM()
    try:
        from src.jit.recursive_codegen import compile_recursive_functions
        vm.recursive_funcs = compile_recursive_functions(ast)
    except Exception:
        vm.recursive_funcs = {}
    t_exec_start = time.perf_counter()
    try:
        vm.execute(instrs)
        t_exec = (time.perf_counter() - t_exec_start) * 1000.0
    except Exception as e:
        write_crash_report(e, vm.call_stack, vm.ip, instrs[vm.ip].opcode if vm.ip < len(instrs) else "HALT", vm.globals)
        sys.exit(1)
        
    t_total = (time.perf_counter() - t_start) * 1000.0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    quantum_ops = sum(1 for inst in instrs if inst.opcode == 'Q_GATE')
    
    opcode_counts = {}
    for inst in instrs:
        opcode_counts[inst.opcode] = opcode_counts.get(inst.opcode, 0) + 1
    opcode_counts_str = ", ".join(f"{op}:{count}" for op, count in sorted(opcode_counts.items()))
    
    print("=" * 60)
    print("                 EIGEN RESOURCE PROFILE                 ")
    print("=" * 60)
    print(f"Parse & Resolve Time:  {t_parse_resolve:.2f} ms")
    print(f"Typecheck Time:        {t_typecheck:.2f} ms")
    print(f"EQIR Conversion Time:  {t_eqir:.2f} ms")
    print(f"Optimization Time:     {t_opt:.2f} ms")
    print(f"Bytecode Compile Time: {t_ebc_compile:.2f} ms")
    print(f"VM Execution Time:     {t_exec:.2f} ms")
    print(f"Total Time:            {t_total:.2f} ms")
    print("-" * 60)
    print(f"Peak Memory:           {peak / 1024 / 1024:.4f} MB")
    print(f"Heap Allocations:      {vm.next_ref_id - 1} objects")
    print(f"Call Depth Peak:       {len(vm.call_stack)}")
    print(f"Quantum Ops:           {quantum_ops} gates")
    print(f"Opcode Counts:         {opcode_counts_str}")
    print("-" * 60)
    print(f"JIT Executions:        {vm.jit_hits}")
    print(f"JIT Deopts:            {vm.jit_deopts}")
    print(f"JIT Compiled Blocks:   {len(vm.jit.cache.cache)}")
    print("=" * 60)

    if getattr(args, 'flamegraph', False):
        print()
        print("=" * 60)
        print("                 FLAMEGRAPH BREAKDOWN                 ")
        print("=" * 60)
        phases = [
            ("Parse & Resolve", t_parse_resolve),
            ("Typecheck", t_typecheck),
            ("EQIR Conversion", t_eqir),
            ("Optimization", t_opt),
            ("Bytecode Compile", t_ebc_compile),
            ("VM Execution", t_exec),
        ]
        max_width = 50
        max_time = max(p[1] for p in phases) if phases else 1.0
        for name, t in phases:
            bar_len = int((t / max_time) * max_width) if max_time > 0 else 0
            bar = "\u2588" * max(bar_len, 1)
            pct = (t / t_total * 100) if t_total > 0 else 0
            print(f"  {name:<20s} {bar} {t:.2f}ms ({pct:.1f}%)")
        print("=" * 60)
