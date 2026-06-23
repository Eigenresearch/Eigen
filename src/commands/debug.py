import sys
import os
from src.cli import register_command
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker
from src.backend.ebc_compiler import EBCCompiler

@register_command("debug")
def debug_command(args, workspace_root):
    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)
        
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
    
    from src.debugger.debugger import EigenDebugger
    dbg = EigenDebugger()
    dbg.debug(instrs)
