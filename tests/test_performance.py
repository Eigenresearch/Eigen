import unittest
import time
from src.lexer import Lexer
from src.parser import Parser
from src.type_checker import TypeChecker
from src.ebc_compiler import EBCCompiler
from src.vm import EigenVM

class TestPerformance(unittest.TestCase):
    def test_vm_execution_performance(self):
        source = """
        eigen 1.0
        let sum: int = 0
        let arr: array<int> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in arr {
            sum += x
        }
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        tc = TypeChecker()
        tc.check(ast)
        compiler = EBCCompiler()
        instrs = compiler.compile_ast(ast)
        
        vm = EigenVM()
        
        start_time = time.perf_counter()
        vm.execute(instrs)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Relaxed performance assertion for virtualized CI/CD environments
        self.assertLess(duration_ms, 500.0)
        self.assertEqual(vm.lookup_var("sum"), 55)

if __name__ == "__main__":
    unittest.main()
