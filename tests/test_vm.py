import unittest
from src.bytecode import Opcode, Instruction
from src.ebc_compiler import (
    EBCCompiler, Label, WhileNode, TryCatchNode, ThrowNode,
    StructAllocNode, StructGetNode, StructSetNode,
    MapAllocNode, MapGetNode, MapSetNode,
    ArrayAllocNode, ArrayGetNode, ArraySetNode
)
from src.vm import EigenVM, VMRef
from src.ast import (
    ProgramNode, LetNode, VarDeclNode, LiteralNode, VarRefNode,
    BinaryOpNode, GateNode, MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, QFuncCallNode
)
from src.ir_graph import EQIRGraph

class TestEigenVMAndCompiler(unittest.TestCase):

    def setUp(self):
        self.compiler = EBCCompiler()
        self.vm = EigenVM()

    def test_arithmetic_and_comparisons(self):
        # Program: assert (2 + 3 * 4) == 14
        # and: (5 > 3) and (2 <= 2)
        # AST: LetNode(res, "int", BinaryOpNode("+", LiteralNode(2, "int"), BinaryOpNode("*", LiteralNode(3, "int"), LiteralNode(4, "int"))))
        expr = BinaryOpNode("+", LiteralNode(2, "int"), BinaryOpNode("*", LiteralNode(3, "int"), LiteralNode(4, "int")))
        let_node = LetNode("res", "int", expr)
        
        program = ProgramNode(1.0, None, [], [let_node])
        instructions = self.compiler.compile_ast(program)
        
        self.vm.execute(instructions)
        self.assertEqual(self.vm.lookup_var("res"), 14)

        # Let's test comparisons: 5 > 3 -> True, 2 <= 2 -> True, and them -> True
        comp_expr = BinaryOpNode("and",
            BinaryOpNode(">", LiteralNode(5, "int"), LiteralNode(3, "int")),
            BinaryOpNode("<=", LiteralNode(2, "int"), LiteralNode(2, "int"))
        )
        let_comp = LetNode("comp_res", "bool", comp_expr)
        program = ProgramNode(1.0, None, [], [let_comp])
        instructions = self.compiler.compile_ast(program)
        self.vm.execute(instructions)
        self.assertTrue(self.vm.lookup_var("comp_res"))

    def test_control_flow_if(self):
        # Program:
        # let x: int = 10
        # if x == 10 {
        #     let y: int = 20
        # }
        # Note: IfNode condition is compared: condition_left, op, condition_right
        let_x = LetNode("x", "int", LiteralNode(10, "int"))
        if_node = IfNode(VarRefNode("x"), "==", LiteralNode(10, "int"), [
            LetNode("y", "int", LiteralNode(20, "int"))
        ])
        program = ProgramNode(1.0, None, [], [let_x, if_node])
        instructions = self.compiler.compile_ast(program)
        
        self.vm.execute(instructions)
        self.assertEqual(self.vm.lookup_var("y"), 20)

        # Now test false condition:
        if_node_false = IfNode(VarRefNode("x"), "==", LiteralNode(5, "int"), [
            LetNode("z", "int", LiteralNode(99, "int"))
        ])
        program = ProgramNode(1.0, None, [], [let_x, if_node_false])
        instructions = self.compiler.compile_ast(program)
        self.vm.execute(instructions)
        # lookup_var returns name if not found
        self.assertEqual(self.vm.lookup_var("z"), "z")

    def test_loops_while(self):
        # Program:
        # let i: int = 0
        # while i < 5 {
        #     let i: int = i + 1
        # }
        let_i = LetNode("i", "int", LiteralNode(0, "int"))
        while_node = WhileNode(
            BinaryOpNode("<", VarRefNode("i"), LiteralNode(5, "int")),
            [LetNode("i", "int", BinaryOpNode("+", VarRefNode("i"), LiteralNode(1, "int")))]
        )
        program = ProgramNode(1.0, None, [], [let_i, while_node])
        instructions = self.compiler.compile_ast(program)
        
        self.vm.execute(instructions)
        self.assertEqual(self.vm.lookup_var("i"), 5)

    def test_functions_call_ret(self):
        # Program:
        # qfunc add_one(int val) {
        #     let res: int = val + 1
        #     return
        # }
        # Note: In our current compiler structure, qfunc decls are resolved to jump targets,
        # and ReturnNode returns None. But let's check function scoping.
        # Call: add_one(x)
        from src.ast import QFuncDeclNode
        func_decl = QFuncDeclNode("add_one", [("val", "int")], [
            LetNode("res", "int", BinaryOpNode("+", VarRefNode("val"), LiteralNode(1, "int"))),
            ReturnNode()
        ])
        let_x = LetNode("x", "int", LiteralNode(10, "int"))
        call_node = QFuncCallNode("add_one", ["x"])
        
        program = ProgramNode(1.0, None, [], [func_decl, let_x, call_node])
        instructions = self.compiler.compile_ast(program)
        
        self.vm.execute(instructions)
        # Note: the VM operand stack should contain the return value from the call (which is None)
        self.assertEqual(self.vm.operand_stack[-1], None)

    def test_structs(self):
        # Program:
        # let s = StructAllocNode(["x", "y"], [10, 20])
        # let val = StructGetNode(s, "y")
        # StructSetNode(s, "x", 30)
        # let val2 = StructGetNode(s, "x")
        struct_alloc = StructAllocNode(["x", "y"], [LiteralNode(10, "int"), LiteralNode(20, "int")])
        let_s = LetNode("s", "struct", struct_alloc)
        get_y = LetNode("val", "int", StructGetNode(VarRefNode("s"), "y"))
        set_x = StructSetNode(VarRefNode("s"), "x", LiteralNode(30, "int"))
        get_x = LetNode("val2", "int", StructGetNode(VarRefNode("s"), "x"))

        program = ProgramNode(1.0, None, [], [let_s, get_y, set_x, get_x])
        instructions = self.compiler.compile_ast(program)
        
        self.vm.execute(instructions)
        
        # Verify heap reference
        s_ref = self.vm.lookup_var("s")
        self.assertIsInstance(s_ref, VMRef)
        self.assertEqual(self.vm.heap[s_ref.ref_id].obj_type, "struct")
        self.assertEqual(self.vm.lookup_var("val"), 20)
        self.assertEqual(self.vm.lookup_var("val2"), 30)

    def test_maps(self):
        # Program:
        # let m = MapAllocNode(["a", "b"], [100, 200])
        # let val = MapGetNode(m, "b")
        # MapSetNode(m, "a", 300)
        # let val2 = MapGetNode(m, "a")
        map_alloc = MapAllocNode(
            [LiteralNode("a", "str"), LiteralNode("b", "str")],
            [LiteralNode(100, "int"), LiteralNode(200, "int")]
        )
        let_m = LetNode("m", "map", map_alloc)
        get_b = LetNode("val", "int", MapGetNode(VarRefNode("m"), LiteralNode("b", "str")))
        set_a = MapSetNode(VarRefNode("m"), LiteralNode("a", "str"), LiteralNode(300, "int"))
        get_a = LetNode("val2", "int", MapGetNode(VarRefNode("m"), LiteralNode("a", "str")))

        program = ProgramNode(1.0, None, [], [let_m, get_b, set_a, get_a])
        instructions = self.compiler.compile_ast(program)
        
        self.vm.execute(instructions)
        self.assertEqual(self.vm.lookup_var("val"), 200)
        self.assertEqual(self.vm.lookup_var("val2"), 300)

    def test_arrays(self):
        # Program:
        # let arr = ArrayAllocNode([10, 20, 30])
        # let val = ArrayGetNode(arr, 1)
        # ArraySetNode(arr, 0, 99)
        # let val2 = ArrayGetNode(arr, 0)
        array_alloc = ArrayAllocNode([LiteralNode(10, "int"), LiteralNode(20, "int"), LiteralNode(30, "int")])
        let_arr = LetNode("arr", "array", array_alloc)
        get_1 = LetNode("val", "int", ArrayGetNode(VarRefNode("arr"), LiteralNode(1, "int")))
        set_0 = ArraySetNode(VarRefNode("arr"), LiteralNode(0, "int"), LiteralNode(99, "int"))
        get_0 = LetNode("val2", "int", ArrayGetNode(VarRefNode("arr"), LiteralNode(0, "int")))

        program = ProgramNode(1.0, None, [], [let_arr, get_1, set_0, get_0])
        instructions = self.compiler.compile_ast(program)
        
        self.vm.execute(instructions)
        self.assertEqual(self.vm.lookup_var("val"), 20)
        self.assertEqual(self.vm.lookup_var("val2"), 99)

    def test_try_catch_exception_handling(self):
        # Program:
        # try {
        #     throw "test_error"
        # } catch (err) {
        #     let caught = err
        # }
        try_catch = TryCatchNode(
            [ThrowNode(LiteralNode("test_error", "str"))],
            "err",
            [LetNode("caught", "str", VarRefNode("err"))]
        )
        program = ProgramNode(1.0, None, [], [try_catch])
        instructions = self.compiler.compile_ast(program)
        
        self.vm.execute(instructions)
        self.assertEqual(self.vm.lookup_var("caught"), "test_error")

    def test_uncaught_exception_traceback(self):
        # Program:
        # throw "fatal"
        program = ProgramNode(1.0, None, [], [ThrowNode(LiteralNode("fatal", "str"))])
        instructions = self.compiler.compile_ast(program)
        
        with self.assertRaises(RuntimeError) as context:
            self.vm.execute(instructions)
        
        self.assertIn("Uncaught Exception: fatal", str(context.exception))
        self.assertIn("Stack Trace:", str(context.exception))

    def test_quantum_operations(self):
        # Create a simple bell state circuit and compile via EQIR
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('ALLOC', targets=['q1'])
        graph.add_operation('GATE', gate_name='H', targets=['q0'])
        graph.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q1'])
        graph.add_operation('MEASURE', targets=['q0'], cbit_name='c0')
        graph.add_operation('MEASURE', targets=['q1'], cbit_name='c1')

        instructions = self.compiler.compile_eqir(graph)
        self.vm.execute(instructions)

        # Check cbits in vm globals (either 0 or 1, and they must be equal due to entanglement!)
        c0 = self.vm.lookup_var("c0")
        c1 = self.vm.lookup_var("c1")
        self.assertIn(c0, (0, 1))
        self.assertIn(c1, (0, 1))
        self.assertEqual(c0, c1)


if __name__ == "__main__":
    unittest.main()
