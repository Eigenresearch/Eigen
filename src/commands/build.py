import os
import json
from src.cli import register_command
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker
from src.backend.ebc_compiler import EBCCompiler
from src.packager import EigenPackager

@register_command("build")
def build_command(args, workspace_root):
    if args.file:
        if getattr(args, 'llvm', False):
            print(f"Compiling '{args.file}' to LLVM IR (.ll)...")
        else:
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
        if args.optimize:
            from src.ir.ssa.optimizer import optimize_ebc
            instrs = optimize_ebc(instrs)
        
        if getattr(args, 'llvm', False):
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
