from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.type_checker import TypeChecker
from src.backend.ebc_compiler import EBCCompiler
from src.backend.vm import EigenVM
from src.semantic.import_resolver import ImportResolver

def run_eigen_code(source: str, workspace_root: str = ".") -> EigenVM:
    lexer = Lexer(source)
    parser = Parser(lexer.tokenize())
    ast = parser.parse()
    
    resolver = ImportResolver(workspace_root)
    ast = resolver.resolve(ast)
    
    tc = TypeChecker()
    tc.check(ast)
    
    compiler = EBCCompiler()
    instrs = compiler.compile_ast(ast)
    
    vm = EigenVM()
    vm.execute(instrs)
    return vm
