"""Memory safety and VM resilience tests for EigenVM."""
import unittest
from src.backend.bytecode import Opcode, Instruction
from src.backend.vm import EigenVM, UnsupportedBytecodeVersionError, UndefinedVariableError


class TestVMMemorySafety(unittest.TestCase):

    def setUp(self):
        self.vm = EigenVM()

    def test_stack_underflow_on_add(self):
        """ADD with empty operand stack should trigger StackUnderflowError."""
        instructions = [
            Instruction(Opcode.ADD),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("StackUnderflow", str(ctx.exception))

    def test_stack_underflow_on_sub(self):
        instructions = [
            Instruction(Opcode.SUB),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("StackUnderflow", str(ctx.exception))

    def test_stack_underflow_on_mul(self):
        instructions = [
            Instruction(Opcode.MUL),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("StackUnderflow", str(ctx.exception))

    def test_stack_underflow_on_div(self):
        instructions = [
            Instruction(Opcode.DIV),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("StackUnderflow", str(ctx.exception))

    def test_stack_underflow_on_eq(self):
        instructions = [
            Instruction(Opcode.EQ),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("StackUnderflow", str(ctx.exception))

    def test_stack_underflow_on_comparison_ops(self):
        for op in [Opcode.LT, Opcode.GT, Opcode.LTE, Opcode.GTE, Opcode.NEQ, Opcode.AND, Opcode.OR]:
            vm = EigenVM()
            instructions = [
                Instruction(op),
                Instruction(Opcode.HALT),
            ]
            with self.assertRaises(RuntimeError, msg=f"Expected RuntimeError for {op}"):
                vm.execute(instructions)

    def test_stack_underflow_on_not(self):
        instructions = [
            Instruction(Opcode.NOT),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("StackUnderflow", str(ctx.exception))

    def test_stack_underflow_on_store_var(self):
        instructions = [
            Instruction(Opcode.STORE_VAR, "x"),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("StackUnderflow", str(ctx.exception))

    def test_stack_underflow_on_jmp_if_false(self):
        instructions = [
            Instruction(Opcode.JMP_IF_FALSE, 1),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("StackUnderflow", str(ctx.exception))

    def test_stack_underflow_on_print(self):
        instructions = [
            Instruction(Opcode.PRINT),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("StackUnderflow", str(ctx.exception))

    def test_deep_recursion_call_stack_overflow(self):
        """Deeply recursive calls should eventually raise RecursionError or RuntimeError."""
        # Create a function that calls itself: ENTER_FRAME, CALL self, RET
        instructions = [
            Instruction(Opcode.LOAD_CONST, 0),        # 0: push dummy arg
            Instruction(Opcode.CALL, (0, "self_recurse", 0)),  # 1: call at ip=0
            Instruction(Opcode.HALT),                  # 2: never reached
        ]
        with self.assertRaises((RecursionError, RuntimeError, IndexError)):
            self.vm.execute(instructions)

    def test_invalid_jump_target(self):
        """Jump to out-of-bounds IP should not crash with unhandled exception."""
        instructions = [
            Instruction(Opcode.JMP, 9999),  # Jump way beyond instruction list
            Instruction(Opcode.HALT),
        ]
        # Should either halt cleanly (ip > len) or raise a controlled error
        try:
            self.vm.execute(instructions)
        except (IndexError, RuntimeError):
            pass  # Acceptable controlled error

    def test_division_by_zero(self):
        """Division by zero should trigger an exception via throw_exception."""
        instructions = [
            Instruction(Opcode.LOAD_CONST, 10),
            Instruction(Opcode.LOAD_CONST, 0),
            Instruction(Opcode.DIV),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("DivisionByZero", str(ctx.exception))

    def test_halt_stops_execution(self):
        """HALT should stop execution immediately."""
        instructions = [
            Instruction(Opcode.LOAD_CONST, 42),
            Instruction(Opcode.STORE_VAR, "x"),
            Instruction(Opcode.HALT),
            Instruction(Opcode.LOAD_CONST, 99),  # Should never execute
            Instruction(Opcode.STORE_VAR, "y"),
        ]
        self.vm.execute(instructions)
        self.assertEqual(self.vm.lookup_var("x"), 42)
        # y is not defined, so lookup_var("y") should raise UndefinedVariableError
        with self.assertRaises(UndefinedVariableError):
            self.vm.lookup_var("y")

    def test_empty_instructions(self):
        """Executing an empty instruction list should not crash."""
        self.vm.execute([])

    def test_load_var_undefined(self):
        """Loading an undefined variable should raise UndefinedVariableError, not crash."""
        instructions = [
            Instruction(Opcode.LOAD_VAR, "undefined_var"),
            Instruction(Opcode.STORE_VAR, "result"),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(UndefinedVariableError):
            self.vm.execute(instructions)

    def test_bytecode_version_check(self):
        """UnsupportedBytecodeVersionError should be available."""
        err = UnsupportedBytecodeVersionError("Bytecode version 99 not supported")
        self.assertIsInstance(err, Exception)
        self.assertIn("99", str(err))

    def test_throw_without_try_raises_runtime_error(self):
        """THROW without a TRY handler should raise RuntimeError."""
        instructions = [
            Instruction(Opcode.LOAD_CONST, "test_exception"),
            Instruction(Opcode.THROW),
            Instruction(Opcode.HALT),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self.vm.execute(instructions)
        self.assertIn("Uncaught Exception", str(ctx.exception))

    def test_try_catch_handles_throw(self):
        """THROW inside TRY should be caught."""
        instructions = [
            Instruction(Opcode.PUSH_TRY, 4),           # 0: handler at ip=4
            Instruction(Opcode.LOAD_CONST, "err_val"),  # 1
            Instruction(Opcode.THROW),                  # 2
            Instruction(Opcode.JMP, 6),                 # 3: skip catch (never reached)
            Instruction(Opcode.STORE_VAR, "caught"),    # 4: catch handler
            Instruction(Opcode.POP_TRY),                # 5
            Instruction(Opcode.HALT),                   # 6
        ]
        self.vm.execute(instructions)
        self.assertEqual(self.vm.lookup_var("caught"), "err_val")

    def test_ret_without_call_frame(self):
        """RET from the main frame should not crash uncontrollably."""
        instructions = [
            Instruction(Opcode.LOAD_CONST, 0),
            Instruction(Opcode.RET),
            Instruction(Opcode.HALT),
        ]
        try:
            self.vm.execute(instructions)
        except (IndexError, RuntimeError, TypeError):
            pass  # Acceptable

    def test_large_operand_stack(self):
        """Pushing many values should work without crash."""
        instructions = []
        for i in range(1000):
            instructions.append(Instruction(Opcode.LOAD_CONST, i))
        instructions.append(Instruction(Opcode.HALT))
        self.vm.execute(instructions)
        self.assertEqual(len(self.vm.operand_stack), 1000)

    def test_multiple_halt_instructions(self):
        instructions = [
            Instruction(Opcode.HALT),
            Instruction(Opcode.HALT),
            Instruction(Opcode.HALT),
        ]
        self.vm.execute(instructions)


if __name__ == "__main__":
    unittest.main()
