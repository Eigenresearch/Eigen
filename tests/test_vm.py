import unittest
from src.backend.bytecode import Opcode, Instruction
from src.backend.ebc_compiler import (
    EBCCompiler, WhileNode, TryCatchNode, ThrowNode,
    StructAllocNode, StructGetNode, StructSetNode,
    MapAllocNode, MapGetNode, MapSetNode,
    ArrayAllocNode, ArrayGetNode, ArraySetNode
)
from src.backend.vm import EigenVM, VMRef, UndefinedVariableError
from src.frontend.ast import (
    ProgramNode, LetNode, LiteralNode, VarRefNode,
    BinaryOpNode, IfNode, ReturnNode, QFuncCallNode
)
from src.ir.ir_graph import EQIRGraph

class TestEigenVMAndCompiler(unittest.TestCase):

    def setUp(self):
        self.compiler = EBCCompiler()
        self.vm = EigenVM()

    def test_arithmetic_and_comparisons(self):
        # Program: assert (2 + 3 * 4) == 14
        # and: (5 > 3) and (2 <= 2)
        # AST: LetNode(res, "int", BinaryOpNode("+", LiteralNode(2, "int"),
        #     BinaryOpNode("*", LiteralNode(3, "int"), LiteralNode(4, "int"))))
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
        with self.assertRaises(UndefinedVariableError):
            self.vm.lookup_var("z")

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
        from src.frontend.ast import QFuncDeclNode
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

    def test_division_by_zero_check(self):
        program = ProgramNode(1.0, None, [], [
            LetNode("div_zero", "int", BinaryOpNode("/", LiteralNode(10, "int"), LiteralNode(0, "int")))
        ])
        instructions = self.compiler.compile_ast(program)
        
        with self.assertRaises(RuntimeError) as context:
            self.vm.execute(instructions)
            
        self.assertIn("DivisionByZeroError: Division by zero.", str(context.exception))

    def test_stack_overflow_check(self):
        from src.frontend.ast import QFuncDeclNode
        func_decl = QFuncDeclNode("recurse", [], [
            QFuncCallNode("recurse", [])
        ])
        call_node = QFuncCallNode("recurse", [])
        
        program = ProgramNode(1.0, None, [], [func_decl, call_node])
        instructions = self.compiler.compile_ast(program)
        
        with self.assertRaises(RuntimeError) as context:
            self.vm.execute(instructions)
            
        self.assertIn("StackOverflowError: Maximum recursion depth (1000) exceeded.", str(context.exception))

    def test_jit_global_caching_and_lru(self):
        # Audit §1.1 BUG #5 / §2.3: the JIT cache is now per-instance, not class-
        # level. This test verifies the per-instance cache still accumulates
        # compiled blocks (after enough iterations to cross the hot threshold)
        # and that LRU evicts oldest entries when the cache is full.


        # 2. Create a basic loop program to trigger JIT compilation
        # let i: int = 0
        # while i < 15 {
        #     let i: int = i + 1
        # }
        let_i = LetNode("i", "int", LiteralNode(0, "int"))
        while_node = WhileNode(
            BinaryOpNode("<", VarRefNode("i"), LiteralNode(15, "int")),
            [LetNode("i", "int", BinaryOpNode("+", VarRefNode("i"), LiteralNode(1, "int")))]
        )
        program = ProgramNode(1.0, None, [], [let_i, while_node])
        instructions = self.compiler.compile_ast(program)

        # Verify execution and check cache contents.
        import src.backend.vm as vm_module
        old_native = vm_module.native
        vm_module.native = None
        try:
            vm1 = EigenVM()
            vm1._try_fast_loop = lambda instructions: False
            vm1._try_fast_array_loop = lambda instructions: False
            vm1.execute(instructions)
            self.assertEqual(vm1.lookup_var("i"), 15)

            # The block containing the loop body should have crossed the JIT's
            # hot_threshold and been compiled into the per-instance cache.
            self.assertGreater(len(vm1.jit.cache.cache), 0)

            # Per-instance exec counts must have grown during the run.
            self.assertGreater(len(vm1.jit.exec_counts), 0)

            # 3. Create a second VM instance and run again: its per-instance
            # cache starts empty and is not contaminated by the first VM.
            vm2 = EigenVM()
            vm2._try_fast_loop = lambda instructions: False
            vm2._try_fast_array_loop = lambda instructions: False
            vm2.execute(instructions)
            self.assertEqual(vm2.lookup_var("i"), 15)
        finally:
            vm_module.native = old_native

        # The two instances' caches are independent: vm2's first run had to
        # warm up its own exec-counts, then compile its own block(s).
        self.assertGreater(len(vm2.jit.cache.cache), 0)

        # 4. Test LRU eviction: set maxsize to 2 and add 3 elements
        from src.jit.jit_compiler import LRUCache
        lru = LRUCache(maxsize=2)
        lru.put("a", 1)
        lru.put("b", 2)
        # Access "a" to make "b" least recently used
        lru.get("a")
        # Put "c" -> "b" should be evicted
        lru.put("c", 3)

        self.assertEqual(lru.get("a"), 1)
        self.assertIsNone(lru.get("b"))
        self.assertEqual(lru.get("c"), 3)

    def test_jit_v2_constant_folding(self):
        from src.jit.jit_compiler import JITCompiler
        jit = JITCompiler(self.vm)
        
        # 2 + 3 * 4 -> should fold 3 * 4 to 12, then 2 + 12 to 14
        instrs = [
            Instruction(Opcode.LOAD_CONST, 2),
            Instruction(Opcode.LOAD_CONST, 3),
            Instruction(Opcode.LOAD_CONST, 4),
            Instruction(Opcode.MUL, None),
            Instruction(Opcode.ADD, None)
        ]
        
        folded = jit.fold_constants(instrs)
        self.assertEqual(len(folded), 1)
        self.assertEqual(folded[0].opcode, Opcode.LOAD_CONST)
        self.assertEqual(folded[0].arg, 14)

    def test_jit_v2_inlining(self):
        from src.jit.jit_compiler import JITCompiler
        
        # Define a simple function: func add_one(x) { return x + 1 }
        # EBC:
        # func_add_one:
        #   ENTER_FRAME
        #   STORE_VAR x
        #   LOAD_VAR x
        #   LOAD_CONST 1
        #   ADD
        #   RET
        instructions = [
            Instruction(Opcode.JMP, 7), # Jump to main_start
            Instruction(Opcode.ENTER_FRAME, None),
            Instruction(Opcode.STORE_VAR, "x"),
            Instruction(Opcode.LOAD_VAR, "x"),
            Instruction(Opcode.LOAD_CONST, 1),
            Instruction(Opcode.ADD, None),
            Instruction(Opcode.RET, None),
            # main_start:
            Instruction(Opcode.LOAD_CONST, 5),
            Instruction(Opcode.CALL, (1, "add_one", 1)),
            Instruction(Opcode.STORE_VAR, "res"),
            Instruction(Opcode.HALT, None)
        ]
        
        self.vm.instructions = instructions
        jit = JITCompiler(self.vm)
        
        # Test inlining target 1 (entry of add_one call)
        inlined = jit.inline_function(1, 1, "add_one")
        self.assertIsNotNone(inlined)
        
        # Should contain parameter stores and body, without ENTER_FRAME and RET
        opcodes = [inst.opcode for inst in inlined]
        self.assertEqual(opcodes, [Opcode.STORE_VAR, Opcode.LOAD_VAR, Opcode.LOAD_CONST, Opcode.ADD])
        
        # Verify variables are renamed/namespaced
        self.assertTrue(inlined[0].arg.startswith("x_add_one_inline_"))

    def test_jit_v2_type_guards_and_deopt(self):
        from src.jit.jit_compiler import JITCompiler
        
        # Loop program that modifies type of 'val':
        # val = 5 (int)
        # val = 3.14 (float)
        # JIT should compile while 'val' is int, then deopt/fallback when it becomes float.
        # We simulate this behavior:
        instructions = [
            Instruction(Opcode.LOAD_VAR, "val"),
            Instruction(Opcode.LOAD_CONST, 1),
            Instruction(Opcode.ADD, None),
            Instruction(Opcode.STORE_VAR, "res"),
            Instruction(Opcode.HALT, None)
        ]
        
        self.vm.instructions = instructions
        self.vm.globals = {"val": 5}
        
        jit = JITCompiler(self.vm)
        compiled = jit.compile_block(instructions[:-1]) # compile LOAD_VAR, LOAD_CONST, ADD, STORE_VAR
        self.assertIsNotNone(compiled)
        
        # Execute once with int val -> res should be 6
        stack = []
        globals_map = {"val": 5}
        def lookup_var(name):
            return globals_map[name]
            
        res = compiled(stack, globals_map, globals_map, lookup_var, self.vm)
        self.assertFalse(res) # completed without change of control flow (vm.ip update)
        self.assertEqual(globals_map.get("res"), 6)
        
        # Now change val to float 3.14 -> JIT guard should trigger and return True (deopt)
        globals_map["val"] = 3.14
        res = compiled(stack, globals_map, globals_map, lookup_var, self.vm)
        self.assertTrue(res) # guard failed -> returned True to VM to fallback

if __name__ == "__main__":
    unittest.main()

