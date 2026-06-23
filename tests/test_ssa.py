import unittest
from src.backend.bytecode import Opcode, Instruction
from src.ir.ssa.ssa_builder import SSABuilder

class TestSSAIRBuilder(unittest.TestCase):
    def test_simple_ssa_rename(self):
        # Let's mock a sequence of bytecode representing:
        # let x: int = 1
        # x = 2
        # print x
        instructions = [
            Instruction(Opcode.LOAD_CONST, 1),
            Instruction(Opcode.STORE_VAR, "x"),
            Instruction(Opcode.LOAD_CONST, 2),
            Instruction(Opcode.STORE_VAR, "x"),
            Instruction(Opcode.LOAD_VAR, "x"),
            Instruction(Opcode.PRINT),
            Instruction(Opcode.HALT)
        ]
        
        builder = SSABuilder()
        blocks, ssa_repr = builder.build_ssa(instructions)
        
        # Verify renaming
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        # x should be renamed to x_1 and x_2
        self.assertEqual(block.instructions[1].arg, "x_1")
        self.assertEqual(block.instructions[3].arg, "x_2")
        self.assertEqual(block.instructions[4].arg, "x_2")

    def test_conditional_branch_phi(self):
        # Mock control flow with branches
        # let x: int = 0
        # if cond {
        #     x = 10
        # }
        # print x
        # Basic blocks:
        # B0:
        #   x = 0
        #   cond = LOAD_VAR "c"
        #   JMP_IF_FALSE B2 (idx 8)
        # B1 (if branch):
        #   x = 10
        #   JMP B2 (idx 8)
        # B2 (merge):
        #   print x
        #   HALT
        instructions = [
            # Block 0
            Instruction(Opcode.LOAD_CONST, 0),
            Instruction(Opcode.STORE_VAR, "x"),
            Instruction(Opcode.LOAD_VAR, "c"),
            # index 3: JMP_IF_FALSE to Block 2
            Instruction(Opcode.JMP_IF_FALSE, 6),
            # Block 1 (idx 4)
            Instruction(Opcode.LOAD_CONST, 10),
            Instruction(Opcode.STORE_VAR, "x"),
            # Block 2 (idx 6)
            Instruction(Opcode.LOAD_VAR, "x"),
            Instruction(Opcode.PRINT),
            Instruction(Opcode.HALT)
        ]
        
        builder = SSABuilder()
        blocks, ssa_repr = builder.build_ssa(instructions)
        
        # There should be 3 basic blocks
        self.assertEqual(len(blocks), 3)
        # The merge block (Block 2) should have a phi node for x
        merge_block = blocks[2]
        self.assertEqual(len(merge_block.phi_nodes), 1)
        phi_var = list(merge_block.phi_nodes.keys())[0]
        self.assertTrue(phi_var.startswith("x_"))
        
        # The variables in the phi node list should match predecessor outputs
        phi_entries = merge_block.phi_nodes[phi_var]
        # Should have entries from Block 0 (x_1) and Block 1 (x_2)
        pred_blocks = {pred_id for pred_id, val in phi_entries}
        self.assertEqual(pred_blocks, {0, 1})

    def test_constant_propagation_and_folding(self):
        from src.ir.ssa.optimizer import optimize_ebc
        instructions = [
            Instruction(Opcode.LOAD_CONST, 10),
            Instruction(Opcode.STORE_VAR, "x"),
            Instruction(Opcode.LOAD_CONST, 20),
            Instruction(Opcode.STORE_VAR, "y"),
            Instruction(Opcode.LOAD_VAR, "x"),
            Instruction(Opcode.LOAD_VAR, "y"),
            Instruction(Opcode.ADD),
            Instruction(Opcode.STORE_VAR, "z"),
            Instruction(Opcode.LOAD_VAR, "z"),
            Instruction(Opcode.PRINT),
            Instruction(Opcode.HALT)
        ]
        opt_instrs = optimize_ebc(instructions)
        # Should fold 10 + 20 = 30 and load it directly
        has_load_30 = any(inst.opcode == Opcode.LOAD_CONST and inst.arg == 30 for inst in opt_instrs)
        self.assertTrue(has_load_30, f"Optimized instructions: {opt_instrs}")

    def test_copy_propagation(self):
        from src.ir.ssa.optimizer import optimize_ebc
        instructions = [
            Instruction(Opcode.LOAD_VAR, "a"),
            Instruction(Opcode.LOAD_VAR, "b"),
            Instruction(Opcode.ADD),
            Instruction(Opcode.STORE_VAR, "x"),
            Instruction(Opcode.LOAD_VAR, "x"),
            Instruction(Opcode.STORE_VAR, "y"),
            Instruction(Opcode.LOAD_VAR, "y"),
            Instruction(Opcode.PRINT),
            Instruction(Opcode.HALT)
        ]
        opt_instrs = optimize_ebc(instructions)
        # y should be copy-propagated to x
        # So instead of loading y, it should load x (which is x_1)
        load_vars = [inst.arg for inst in opt_instrs if inst.opcode == Opcode.LOAD_VAR]
        self.assertIn("x_1", load_vars)
        self.assertNotIn("y_1", load_vars)


    def test_dead_store_elimination(self):
        from src.ir.ssa.optimizer import optimize_ebc
        instructions = [
            Instruction(Opcode.LOAD_VAR, "external_val"),
            Instruction(Opcode.STORE_VAR, "x"), # x is never loaded
            Instruction(Opcode.LOAD_CONST, 100),
            Instruction(Opcode.STORE_VAR, "y"), # y is loaded
            Instruction(Opcode.LOAD_VAR, "y"),
            Instruction(Opcode.PRINT),
            Instruction(Opcode.HALT)
        ]
        opt_instrs = optimize_ebc(instructions)
        # x_1 store and load should be eliminated since x is never read
        store_vars = [inst.arg for inst in opt_instrs if inst.opcode == Opcode.STORE_VAR]
        self.assertNotIn("x_1", store_vars)


    def test_common_subexpression_elimination(self):
        from src.ir.ssa.optimizer import optimize_ebc
        instructions = [
            Instruction(Opcode.LOAD_VAR, "a"),
            Instruction(Opcode.LOAD_VAR, "b"),
            Instruction(Opcode.ADD),
            Instruction(Opcode.STORE_VAR, "t1"),
            # Duplicate computation
            Instruction(Opcode.LOAD_VAR, "a"),
            Instruction(Opcode.LOAD_VAR, "b"),
            Instruction(Opcode.ADD),
            Instruction(Opcode.STORE_VAR, "t2"),
            Instruction(Opcode.HALT)
        ]
        opt_instrs = optimize_ebc(instructions)
        # The duplicate ADD should be replaced by loading t1 (or t1_1)
        # So we should see a store of t2_1 immediately after loading t1_1
        # Let's check instructions
        opcodes = [inst.opcode for inst in opt_instrs]
        self.assertEqual(opcodes.count(Opcode.ADD), 1)

    def test_sccp_branch_elimination(self):
        from src.ir.ssa.optimizer import optimize_ebc
        instructions = [
            Instruction(Opcode.LOAD_CONST, False),
            # False, so jump is always taken
            Instruction(Opcode.JMP_IF_FALSE, 4),
            Instruction(Opcode.LOAD_CONST, 10),
            Instruction(Opcode.STORE_VAR, "x"),
            # Target (index 4)
            Instruction(Opcode.LOAD_CONST, 20),
            Instruction(Opcode.STORE_VAR, "y"),
            Instruction(Opcode.HALT)
        ]
        opt_instrs = optimize_ebc(instructions)
        # JMP_IF_FALSE should be simplified to JMP unconditional or eliminated
        opcodes = [inst.opcode for inst in opt_instrs]
        self.assertNotIn(Opcode.JMP_IF_FALSE, opcodes)


if __name__ == "__main__":
    unittest.main()

