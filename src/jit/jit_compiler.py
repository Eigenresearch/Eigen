import hashlib
import json
from collections import OrderedDict
from src.backend.bytecode import Opcode, Instruction

class LRUCache:
    def __init__(self, maxsize=1024):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
            
    def clear(self):
        self.cache.clear()


def get_function_hash(instructions_segment: list[Instruction]) -> str:
    # Serialize the instructions to a stable representation
    serialized = []
    for inst in instructions_segment:
        serialized.append((inst.opcode, inst.arg))
    try:
        data_str = json.dumps(serialized, sort_keys=True)
    except Exception:
        # Fallback if there are any non-serializable objects (e.g. complex numbers)
        def custom_serializer(obj):
            if isinstance(obj, complex):
                return {"__complex__": True, "real": obj.real, "imag": obj.imag}
            if isinstance(obj, (list, tuple)):
                return [custom_serializer(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): custom_serializer(v) for k, v in obj.items()}
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return repr(obj)
        data_str = json.dumps(custom_serializer(serialized), sort_keys=True)
        
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()


class JITCompiler:
    GLOBAL_CACHE = LRUCache(maxsize=1024)
    GLOBAL_EXEC_COUNTS = {}
    GLOBAL_EXEC_COUNTS_MAX = 4096
    DEFAULT_HOT_THRESHOLD = 3
    _inline_counter = 0

    def __init__(self, vm, hot_threshold: int = None, exec_counts_max: int = None):
        self.vm = vm
        self.hot_threshold = hot_threshold if hot_threshold is not None else JITCompiler.DEFAULT_HOT_THRESHOLD
        self.exec_counts_max = exec_counts_max if exec_counts_max is not None else JITCompiler.GLOBAL_EXEC_COUNTS_MAX
        self.current_instructions = None
        self.ip_to_key = {}

    def analyze_program(self, instructions: list[Instruction]):
        self.current_instructions = instructions
        self.ip_to_key = {}
        if not instructions:
            return

        # Find entry points
        entry_points = [0]
        if instructions[0].opcode == Opcode.JMP and isinstance(instructions[0].arg, int):
            entry_points.append(instructions[0].arg)
        for idx, inst in enumerate(instructions):
            if inst.opcode == Opcode.ENTER_FRAME:
                entry_points.append(idx)
        entry_points = sorted(list(set(entry_points)))

        for i in range(len(entry_points)):
            func_start = entry_points[i]
            func_end = entry_points[i+1] if i + 1 < len(entry_points) else len(instructions)
            
            # Extract function instructions
            func_instrs = instructions[func_start:func_end]
            func_hash = get_function_hash(func_instrs)
            
            # Map every instruction within this function to its (func_hash, block_id)
            for ip in range(func_start, func_end):
                self.ip_to_key[ip] = (func_hash, ip - func_start)

    def check_and_compile(self, ip: int, instructions: list[Instruction]) -> callable:
        if self.current_instructions is not instructions:
            self.analyze_program(instructions)

        key = self.ip_to_key.get(ip)
        if not key:
            return None

        # Check global LRU cache
        compiled_func = self.GLOBAL_CACHE.get(key)
        if compiled_func:
            return compiled_func
            
        # Update global execution counts
        count = self.GLOBAL_EXEC_COUNTS.get(key, 0) + 1
        if len(self.GLOBAL_EXEC_COUNTS) < self.GLOBAL_EXEC_COUNTS_MAX or key in self.GLOBAL_EXEC_COUNTS:
            self.GLOBAL_EXEC_COUNTS[key] = count
        if self.GLOBAL_EXEC_COUNTS[key] >= self.hot_threshold:
            # Detect basic block starting at ip
            block = self.trace_basic_block(ip, instructions)
            if len(block) >= 2:  # Only compile blocks of reasonable size (superinstructions reduce count)
                compiled_func = self.compile_block(block)
                if compiled_func:
                    self.GLOBAL_CACHE.put(key, compiled_func)
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
        optimized_block = self.optimize_block(block)
        from src.jit.native_codegen import generate_python_source
        source = generate_python_source(optimized_block, self.vm)
        
        # Compile source to bytecode
        local_vars = {}
        try:
            code_obj = compile(source, '<jit_block>', 'exec')
            safe_globals = {
                "__builtins__": {},
                "type": type,
                "repr": repr,
                "bool": bool,
                "int": int,
                "float": float,
                "str": str,
                "len": len,
                "abs": abs,
                "range": range,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
            }
            exec(code_obj, safe_globals, local_vars)
            return local_vars['compiled_block']
        except Exception as e:
            # Fallback on compilation failure
            return None

    def optimize_block(self, block: list[Instruction]) -> list[Instruction]:
        # 1. Inlining
        inlined = []
        for inst in block:
            if inst.opcode == Opcode.CALL:
                if isinstance(inst.arg, tuple) and len(inst.arg) >= 3:
                    func_target, func_name, num_args = inst.arg
                    if isinstance(func_target, int):
                        inlined_func_instrs = self.inline_function(func_target, num_args, func_name)
                        if inlined_func_instrs is not None:
                            inlined.extend(inlined_func_instrs)
                            continue
            inlined.append(inst)
            
        # 2. Constant Folding
        folded = self.fold_constants(inlined)
        return folded

    def inline_function(self, func_target: int, num_args: int, func_name: str) -> list[Instruction]:
        func_instrs = []
        ip = func_target
        if ip >= len(self.vm.instructions):
            return None
            
        first = self.vm.instructions[ip]
        if first.opcode == Opcode.ENTER_FRAME:
            ip += 1
            
        has_jumps = False
        while ip < len(self.vm.instructions):
            inst = self.vm.instructions[ip]
            if inst.opcode == Opcode.RET:
                break
            if inst.opcode in (Opcode.JMP, Opcode.JMP_IF_FALSE, Opcode.JMP_IF_TRUE, Opcode.CALL, Opcode.HALT):
                has_jumps = True
                break
            func_instrs.append(inst)
            ip += 1
            
        if has_jumps or ip >= len(self.vm.instructions):
            return None
            
        JITCompiler._inline_counter += 1
        suffix = f"_{func_name}_inline_{JITCompiler._inline_counter}"
        renamed_instrs = []
        for inst in func_instrs:
            if inst.opcode in (Opcode.LOAD_VAR, Opcode.STORE_VAR):
                new_inst = Instruction(inst.opcode, f"{inst.arg}{suffix}")
                new_inst.line = inst.line
                renamed_instrs.append(new_inst)
            else:
                renamed_instrs.append(inst)
                
        return renamed_instrs

    def fold_constants(self, instrs: list[Instruction]) -> list[Instruction]:
        changed = True
        current = list(instrs)
        while changed:
            changed = False
            folded = []
            i = 0
            while i < len(current):
                if i + 2 < len(current):
                    inst1 = current[i]
                    inst2 = current[i+1]
                    inst3 = current[i+2]
                    if inst1.opcode == Opcode.LOAD_CONST and inst2.opcode == Opcode.LOAD_CONST:
                        folded_val = None
                        if inst3.opcode == Opcode.ADD:
                            folded_val = inst1.arg + inst2.arg
                        elif inst3.opcode == Opcode.SUB:
                            folded_val = inst1.arg - inst2.arg
                        elif inst3.opcode == Opcode.MUL:
                            folded_val = inst1.arg * inst2.arg
                        elif inst3.opcode == Opcode.DIV:
                            if inst2.arg != 0:
                                folded_val = inst1.arg / inst2.arg
                        elif inst3.opcode == Opcode.EQ:
                            folded_val = inst1.arg == inst2.arg
                        elif inst3.opcode == Opcode.NEQ:
                            folded_val = inst1.arg != inst2.arg
                        elif inst3.opcode == Opcode.LT:
                            folded_val = inst1.arg < inst2.arg
                        elif inst3.opcode == Opcode.GT:
                            folded_val = inst1.arg > inst2.arg
                        elif inst3.opcode == Opcode.LTE:
                            folded_val = inst1.arg <= inst2.arg
                        elif inst3.opcode == Opcode.GTE:
                            folded_val = inst1.arg >= inst2.arg
                        elif inst3.opcode == Opcode.MOD:
                            if inst2.arg != 0:
                                folded_val = inst1.arg % inst2.arg
                        elif inst3.opcode == Opcode.POW:
                            folded_val = inst1.arg ** inst2.arg
                        elif inst3.opcode == Opcode.BIT_AND:
                            folded_val = inst1.arg & inst2.arg
                        elif inst3.opcode == Opcode.BIT_OR:
                            folded_val = inst1.arg | inst2.arg
                        elif inst3.opcode == Opcode.BIT_XOR:
                            folded_val = inst1.arg ^ inst2.arg
                        elif inst3.opcode == Opcode.SHL:
                            folded_val = inst1.arg << inst2.arg
                        elif inst3.opcode == Opcode.SHR:
                            folded_val = inst1.arg >> inst2.arg
                            
                        if folded_val is not None:
                            new_inst = Instruction(Opcode.LOAD_CONST, folded_val)
                            new_inst.line = inst1.line
                            folded.append(new_inst)
                            i += 3
                            changed = True
                            continue
                if i + 1 < len(current):
                    inst1 = current[i]
                    inst2 = current[i+1]
                    if inst1.opcode == Opcode.LOAD_CONST:
                        folded_val = None
                        if inst2.opcode == Opcode.NOT:
                            folded_val = not inst1.arg
                        elif inst2.opcode == Opcode.BIT_NOT:
                            folded_val = ~inst1.arg
                        if folded_val is not None:
                            new_inst = Instruction(Opcode.LOAD_CONST, folded_val)
                            new_inst.line = inst1.line
                            folded.append(new_inst)
                            i += 2
                            changed = True
                            continue
                folded.append(current[i])
                i += 1
            current = folded
        return current
