import sys
from src.backend.bytecode import Opcode, Instruction

class JITCompiler:
    def __init__(self, vm):
        self.vm = vm
        self.hot_threshold = 10
        self.exec_counts = {}  # ip -> count
        self.compiled_blocks = {}  # ip -> callable function

    def check_and_compile(self, ip: int, instructions: list[Instruction]) -> callable:
        if ip in self.compiled_blocks:
            return self.compiled_blocks[ip]
            
        self.exec_counts[ip] = self.exec_counts.get(ip, 0) + 1
        if self.exec_counts[ip] >= self.hot_threshold:
            # Detect basic block starting at ip
            block = self.trace_basic_block(ip, instructions)
            if len(block) > 3:  # Only compile blocks of reasonable size
                compiled_func = self.compile_block(block)
                self.compiled_blocks[ip] = compiled_func
                return compiled_func
        return None

    def trace_basic_block(self, start_ip: int, instructions: list[Instruction]) -> list[Instruction]:
        block = []
        ip = start_ip
        while ip < len(instructions):
            inst = instructions[ip]
            block.append(inst)
            # Basic block ends on JMPs, RETs, HALT, or if the next instruction is a jump target
            # For simplicity, end block on control flow instructions
            if inst.opcode in (Opcode.JMP, Opcode.JMP_IF_FALSE, Opcode.JMP_IF_TRUE, Opcode.RET, Opcode.HALT, Opcode.CALL):
                break
            ip += 1
        return block

    def compile_block(self, block: list[Instruction]) -> callable:
        from src.jit.native_codegen import generate_python_source
        source = generate_python_source(block)
        
        # Compile source to bytecode
        local_vars = {}
        try:
            code_obj = compile(source, '<jit_block>', 'exec')
            exec(code_obj, globals(), local_vars)
            return local_vars['compiled_block']
        except Exception as e:
            # Fallback on compilation failure
            return None
