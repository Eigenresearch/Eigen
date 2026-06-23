import unittest
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.type_checker import TypeChecker
from src.backend.ebc_compiler import EBCCompiler
from src.backend.vm import EigenVM

class TestPhase2(unittest.TestCase):
    def test_factorial_recursion(self):
        # Classical recursive factorial function executing on VM
        source = """
        eigen 1.0
        func fact(n: int) -> int {
            if n == 0 {
                return 1
            }
            return n * fact(n - 1)
        }
        let result: int = fact(5)
        assert result == 120
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        tc = TypeChecker()
        tc.check(ast)
        
        compiler = EBCCompiler()
        instrs = compiler.compile_ast(ast)
        
        vm = EigenVM()
        vm.execute(instrs)
        
        # Verify local frame values
        self.assertEqual(vm.lookup_var("result"), 120)

    def test_struct_manipulation(self):
        source = """
        eigen 1.0
        struct Pair {
            first: int,
            second: float
        }
        let p: Pair = Pair { first: 42, second: 3.14 }
        let val1: int = p.first
        let val2: float = p.second
        p.first = 100
        let val3: int = p.first
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        tc = TypeChecker()
        tc.check(ast)
        
        compiler = EBCCompiler()
        instrs = compiler.compile_ast(ast)
        
        vm = EigenVM()
        vm.execute(instrs)
        
        self.assertEqual(vm.lookup_var("val1"), 42)
        self.assertEqual(vm.lookup_var("val2"), 3.14)
        self.assertEqual(vm.lookup_var("val3"), 100)

    def test_loops_and_arrays(self):
        source = """
        eigen 1.0
        let arr: array<int> = [10, 20, 30]
        let sum: int = 0
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
        vm.execute(instrs)
        
        self.assertEqual(vm.lookup_var("sum"), 60)

    def test_exceptions_try_catch(self):
        source = """
        eigen 1.0
        let caught: string = "none"
        try {
            throw "error_val"
        } catch (e) {
            caught = e
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
        vm.execute(instrs)
        
        self.assertEqual(vm.lookup_var("caught"), "error_val")

    def test_quantum_noise_simulation(self):
        source = """
        eigen 1.0
        qubit q
        cbit c
        H q
        noise depolarizing(0.1) q
        noise bitflip(0.2) q
        measure q -> c
        """
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        tc = TypeChecker()
        tc.check(ast)
        
        compiler = EBCCompiler()
        instrs = compiler.compile_ast(ast)
        
        vm = EigenVM()
        vm.execute(instrs)
        
        # Verify qubit was allocated and measured
        c_val = vm.lookup_var("c")
        self.assertIn(c_val, (0, 1))

if __name__ == "__main__":
    unittest.main()
