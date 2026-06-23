import unittest
import sys
import io
from src.backend.bytecode import Opcode, Instruction
from src.debugger.debugger import EigenDebugger

class TestEigenDebugger(unittest.TestCase):
    def test_debugger_repl_commands(self):
        # A simple bytecode sequence:
        # let x = 5
        # print x
        # halt
        instructions = [
            Instruction(Opcode.LOAD_CONST, 5, line=2),
            Instruction(Opcode.STORE_VAR, "x", line=2),
            Instruction(Opcode.LOAD_VAR, "x", line=3),
            Instruction(Opcode.PRINT, line=3),
            Instruction(Opcode.HALT, line=4)
        ]
        
        dbg = EigenDebugger()
        # Mock sys.stdin with commands:
        # break 3 -> continue -> locals -> step -> step -> exit/quit
        commands = [
            "break 3",
            "continue",
            "locals",
            "step",
            "step",
            "quit"
        ]
        
        sys.stdin = io.StringIO("\n".join(commands) + "\n")
        
        # We catch SystemExit because the quit command exits
        with self.assertRaises(SystemExit):
            dbg.debug(instructions)
            
        # Verify breakpoint was hit and variable x was stored
        self.assertEqual(dbg.vm.lookup_var("x"), 5)

if __name__ == "__main__":
    unittest.main()
