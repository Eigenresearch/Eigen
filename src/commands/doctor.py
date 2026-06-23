import sys
import os
from src.cli import register_command
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.type_checker import TypeChecker

@register_command("doctor")
def doctor_command(args, workspace_root):
    print("=" * 60)
    print("          EIGEN SYSTEM DIAGNOSTICS (eigen doctor)          ")
    print("=" * 60)
    print(f"OS Platform:          {sys.platform}")
    print(f"Python Version:       {sys.version.split()[0]} (>= 3.10 required)")
    
    try:
        import PyInstaller
        print(f"PyInstaller:          AVAILABLE ({PyInstaller.__version__})")
    except ImportError:
        print("PyInstaller:          NOT INSTALLED")
        
    print(f"Workspace Root:       {workspace_root}")
    stdlib_path = os.path.join(workspace_root, "stdlib")
    print(f"Stdlib path:          {stdlib_path} ({'Exists' if os.path.isdir(stdlib_path) else 'NOT FOUND'})")
    
    try:
        lexer = Lexer("eigen 1.0 let x: int = 10 print x")
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        tc = TypeChecker()
        tc.check(ast)
        print("Compiler frontend:    HEALTHY")
    except Exception as e:
        print(f"Compiler frontend:    ERROR ({e})")
        
    print("=" * 60)
