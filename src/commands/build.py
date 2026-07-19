import json
from src.cli import register_command
from src.packager import EigenPackager

@register_command("build")
def build_command(args, workspace_root):
    opt_level = 0
    if getattr(args, "O", None) is not None:
        opt_level = args.O
    elif getattr(args, "optimize", False):
        opt_level = 2
    optimize = (opt_level >= 1)

    if args.file:
        if getattr(args, 'qasm', False) or getattr(args, 'quil', False):
            from src.compiler import to_eqir
            graph, ast = to_eqir(args.file, workspace_root)
            if optimize:
                from src.ir.optimizer import EQIROptimizer
                optimizer = EQIROptimizer()
                graph = optimizer.optimize(graph)
            if getattr(args, 'qasm', False):
                print(f"Exporting '{args.file}' to OpenQASM 3.0...")
                from src.backend.qasm3_exporter import Qasm3Exporter
                exporter = Qasm3Exporter()
                code = exporter.export(graph)
                out_path = args.file.rsplit('.', 1)[0] + ".qasm"
            else:
                print(f"Exporting '{args.file}' to Quil...")
                from src.backend.quil_exporter import QuilExporter
                exporter = QuilExporter()
                code = exporter.export(graph)
                out_path = args.file.rsplit('.', 1)[0] + ".quil"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"Export successful: '{out_path}'")
            return

        if getattr(args, 'qir', False):
            print(f"Compiling '{args.file}' to QIR LLVM IR...")
            from src.aot.compiler import AOTCompiler
            seed_val = getattr(args, 'seed', 0)
            aot_comp = AOTCompiler()
            llvm_module = aot_comp._compile_to_llvm_module(
                args.file, workspace_root, optimize=optimize,
                seed=seed_val, emit_qir=True)
            out_path = args.file.rsplit('.', 1)[0] + ".qir.ll"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(str(llvm_module))
            print(f"QIR LLVM IR dumped successfully: '{out_path}'")
            return

        from src.compiler import get_db, to_ebc
        
        if getattr(args, 'explain_cache', False):
            db = get_db(workspace_root)
            target_query = "optimize" if optimize else "to_ebc"
            db.explain_cache(target_query, args.file)
            
        if getattr(args, 'llvm', False):
            print(f"Compiling '{args.file}' to LLVM IR (.ll)...")
        else:
            print(f"Compiling '{args.file}' to EBC bytecode...")
            
        db = get_db(workspace_root)
        instrs = to_ebc(args.file, workspace_root, optimize=optimize)
        
        if getattr(args, 'aot', False):
            from src.aot.compiler import AOTCompiler
            seed_val = getattr(args, 'seed', 0)
            aot_comp = AOTCompiler()
            if getattr(args, 'emit_llvm', False):
                llvm_module = aot_comp._compile_to_llvm_module(
                    args.file, workspace_root, optimize=optimize, seed=seed_val)
                out_path = args.file.rsplit('.', 1)[0] + ".ll"
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(str(llvm_module))
                print(f"LLVM IR dumped successfully: '{out_path}'")
            exe_path = aot_comp.compile(args.file, workspace_root, optimize=optimize, seed=seed_val)
            print(f"AOT Compilation successful: '{exe_path}'")
        elif getattr(args, 'llvm', False):
            from src.ir.ssa.ssa_builder import SSABuilder
            from src.backend.llvm_compiler import LLVMCompiler
            
            ssa_builder = SSABuilder()
            blocks, _ = ssa_builder.build_ssa(instrs)
            
            llvm_compiler = LLVMCompiler()
            llvm_ir = llvm_compiler.compile_ssa(blocks)
            
            out_path = args.file.rsplit('.', 1)[0] + ".ll"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(llvm_ir)
            print(f"LLVM Compilation successful: '{out_path}'")
        else:
            out_path = args.file.rsplit('.', 1)[0] + ".ebc"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "major": 3,
                    "minor": 0,
                    "instructions": [inst.to_dict() for inst in instrs]
                }, f, indent=2)
            print(f"Compilation successful: '{out_path}'")
    else:
        packager = EigenPackager(workspace_root)
        packager.build_package()
