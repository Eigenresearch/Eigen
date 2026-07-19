import sys
import os
import atheris

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

with atheris.instrument_imports():
    from src.frontend.lexer import Lexer
    from src.frontend.parser import Parser
    from src.backend.vm import EigenVM
    from src.backend.bytecode import Instruction, Opcode

def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    
    # 1. Lexer/Parser Fuzzing
    input_str = fdp.ConsumeUnicodeNoSurrogates(sys.maxsize)
    try:
        tokens = Lexer(input_str).tokenize()
        if tokens:
            parser = Parser(tokens)
            parser.parse()
    except Exception:
        # Controlled exceptions are fine, we look for crashes/hangs
        pass
        
    # 2. VM Fuzzing (Instruction execution)
    # This is a bit more complex, but we can try executing random opcodes
    try:
        # Generate 5-10 random instructions
        instructions = []
        for _ in range(fdp.ConsumeIntInRange(1, 10)):
            opcode_val = fdp.ConsumeIntInRange(0, 100) # Assuming < 100 opcodes
            try:
                opcode = Opcode(opcode_val)
            except ValueError:
                continue
            
            # Simple instructions without complex operands for basic smoke fuzzing
            instructions.append(Instruction(opcode=opcode, operands=[]))
            
        if instructions:
            vm = EigenVM(opt_level=0)
            # Run with a short timeout/limit to prevent infinite loops
            vm.execute(instructions)
    except Exception:
        pass

def main():
    # Safe entrypoint: check if we have enough args for atheris or provide help
    if len(sys.argv) < 2 and "-help" not in sys.argv:
        print("Eigen Atheris Fuzzing Harness")
        print("Usage: python tests/fuzz/atheris_harness.py <corpus_dir> [atheris_flags]")
        print("Example: python tests/fuzz/atheris_harness.py ./corpus -max_total_time=60")
        sys.exit(0)
        
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()

if __name__ == "__main__":
    main()
