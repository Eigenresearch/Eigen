from src.backend.bytecode import Opcode, Instruction
from src.simulator import QuantumSimulator
import random

try:
    import eigen_native as native
except ImportError:
    native = None

class UnsupportedBytecodeVersionError(Exception):
    pass

class VMRef:
    def __init__(self, ref_id: int):
        self.ref_id = ref_id

    def __repr__(self) -> str:
        return f"Ref({self.ref_id})"

    def __eq__(self, other) -> bool:
        if isinstance(other, VMRef):
            return self.ref_id == other.ref_id
        return False

    def __hash__(self) -> int:
        return hash(self.ref_id)


class HeapObject:
    def __init__(self, obj_type: str, data):
        self.obj_type = obj_type  # 'struct', 'map', 'array', 'string'
        self.data = data          # dict, list, or str

    def __repr__(self) -> str:
        return f"HeapObject({self.obj_type}, {self.data})"


class ActivationFrame:
    def __init__(self, return_address: int | None, func_name: str = "main"):
        self.locals = {}
        self.try_stack = []
        self.reset(return_address, func_name)

    def reset(self, return_address: int | None, func_name: str = "main"):
        self.locals.clear()
        self.return_address = return_address
        self.try_stack.clear()
        self.current_line = None
        self.func_name = func_name


class EigenVM:
    def __init__(self, trace_mode: bool = False, noise_model=None, sim_type: str = 'dense', gpu_platform: str = 'none'):
        from src.noise.noise_model import NoiseModel
        self.simulator = QuantumSimulator(sim_type=sim_type, gpu_platform=gpu_platform)
        self.trace_mode = trace_mode
        self.trace_log = []
        self.noise_model = noise_model if noise_model is not None else NoiseModel()
        
        # Trace-Based Adaptive Execution Engine
        from src.jit.jit_compiler import JITCompiler
        self.jit = JITCompiler(self)
        self.jit_enabled = True
        
        # VM registers and stacks
        self.instructions = []
        self.ip = 0
        self.operand_stack = []
        self.call_stack = []
        self.globals = {}
        self.frame_pool = []
        
        # Heap
        self.heap = {}
        self.next_ref_id = 1

        # Dispatch table for bytecode instructions
        self.dispatch_table = {
            Opcode.HALT: self.op_halt,
            Opcode.MUL: self.op_mul,
            Opcode.DIV: self.op_div,
            Opcode.EQ: self.op_eq,
            Opcode.NEQ: self.op_neq,
            Opcode.LT: self.op_lt,
            Opcode.GT: self.op_gt,
            Opcode.LTE: self.op_lte,
            Opcode.GTE: self.op_gte,
            Opcode.AND: self.op_and,
            Opcode.OR: self.op_or,
            Opcode.NOT: self.op_not,
            Opcode.JMP: self.op_jmp,
            Opcode.JMP_IF_FALSE: self.op_jmp_if_false,
            Opcode.JMP_IF_TRUE: self.op_jmp_if_true,
            Opcode.CALL: self.op_call,
            Opcode.RET: self.op_ret,
            Opcode.ENTER_FRAME: self.op_enter_frame,
            Opcode.EXIT_FRAME: self.op_exit_frame,
            Opcode.ALLOC_STRUCT: self.op_alloc_struct,
            Opcode.GET_FIELD: self.op_get_field,
            Opcode.SET_FIELD: self.op_set_field,
            Opcode.ALLOC_MAP: self.op_alloc_map,
            Opcode.ALLOC_ARRAY: self.op_alloc_array,
            Opcode.LEN: self.op_len,
            Opcode.GET_INDEX: self.op_get_index,
            Opcode.SET_INDEX: self.op_set_index,
            Opcode.THROW: self.op_throw,
            Opcode.PUSH_TRY: self.op_push_try,
            Opcode.POP_TRY: self.op_pop_try,
            Opcode.Q_ALLOC: self.op_q_alloc,
            Opcode.Q_GATE: self.op_q_gate,
            Opcode.Q_MEASURE: self.op_q_measure,
            Opcode.Q_NOISE: self.op_q_noise,
            Opcode.Q_TRACE: self.op_q_trace,
            Opcode.PRINT: self.op_print,
            Opcode.SPAWN: self.op_spawn,
            Opcode.JOIN: self.op_join,
        }

    def log_trace(self, msg: str):
        self.trace_log.append(msg)
        if self.trace_mode:
            print(f"[TRACE] {msg}")

    def format_amplitudes(self) -> str:
        amps = self.simulator.get_amplitudes_dict()
        parts = []
        for state, amp in sorted(amps.items()):
            prob = abs(amp) ** 2
            real = amp.real
            imag = amp.imag
            if abs(imag) < 1e-9:
                amp_str = f"{real:.5f}"
            elif abs(real) < 1e-9:
                amp_str = f"{imag:.5f}i"
            else:
                sign = "+" if imag >= 0 else "-"
                amp_str = f"({real:.5f} {sign} {abs(imag):.5f}i)"
            parts.append(f"{amp_str} * |{state}> (prob={prob * 100:.1f}%)")
        return " + ".join(parts)

    def lookup_var(self, name: str):
        if not isinstance(name, str):
            return name
            
        import re
        if re.search(r'[\s\(\)\=\!\+\-\*\/\<\>]', name):
            if not re.match(r'^[a-zA-Z0-9_\s\(\)\=\!\+\-\*\/\<\>\.\,j]+$', name):
                return name
            
            def get_val(word):
                if word in ('True', 'False', 'None'):
                    return word
                if self.call_stack:
                    frame = self.call_stack[-1]
                    if word in frame.locals:
                        val = frame.locals[word]
                        if isinstance(val, bool):
                            return str(val)
                        # Avoid nested quotes
                        if isinstance(val, str) and (val.startswith("'") or val.startswith('"')):
                            return val
                        return repr(val)
                if word in self.globals:
                    val = self.globals[word]
                    if isinstance(val, bool):
                        return str(val)
                    if isinstance(val, str) and (val.startswith("'") or val.startswith('"')):
                        return val
                    return repr(val)
                if word.isidentifier():
                    return '0'
                return word

            subbed = re.sub(r'[a-zA-Z_][a-zA-Z0-9_]*', lambda m: get_val(m.group(0)), name)
            try:
                return eval(subbed, {"__builtins__": None}, {})
            except Exception:
                return name

        if self.call_stack:
            frame = self.call_stack[-1]
            if name in frame.locals:
                return frame.locals[name]
        if name in self.globals:
            return self.globals[name]
        return name

    def throw_exception(self, val):
        handler_found = False
        handler_frame_idx = -1
        for idx in range(len(self.call_stack) - 1, -1, -1):
            if self.call_stack[idx].try_stack:
                handler_found = True
                handler_frame_idx = idx
                break

        if handler_found:
            while len(self.call_stack) > handler_frame_idx + 1:
                self.call_stack.pop()

            frame = self.call_stack[-1]
            handler_ip, saved_stack_depth = frame.try_stack.pop()

            while len(self.operand_stack) > saved_stack_depth:
                self.operand_stack.pop()

            self.operand_stack.append(val)
            self.ip = handler_ip
        else:
            trace_lines = [f"Uncaught Exception: {val}", "Stack Trace:"]
            for frame in reversed(self.call_stack):
                line_info = f", line {frame.current_line}" if frame.current_line is not None else ""
                trace_lines.append(f"  at {frame.func_name} (ip {self.ip - 1}{line_info})")
            trace_str = "\n".join(trace_lines)
            raise RuntimeError(trace_str)

    def allocate_heap(self, obj_type: str, data) -> VMRef:
        ref_id = self.next_ref_id
        self.next_ref_id += 1
        self.heap[ref_id] = HeapObject(obj_type, data)
        return VMRef(ref_id)

    def get_frame(self, return_address, func_name):
        if self.frame_pool:
            frame = self.frame_pool.pop()
            frame.reset(return_address, func_name)
            return frame
        return ActivationFrame(return_address, func_name)

    def recycle_frame(self, frame):
        self.frame_pool.append(frame)

    # Opcode handlers for dispatch
    def op_halt(self, arg):
        return True

    def op_mul(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a * b)

    def op_div(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        if b == 0:
            self.throw_exception("DivisionByZeroError: Division by zero.")
            return
        self.operand_stack.append(a / b)

    def op_eq(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a == b)

    def op_neq(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a != b)

    def op_lt(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a < b)

    def op_gt(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a > b)

    def op_lte(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a <= b)

    def op_gte(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a >= b)

    def op_and(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(bool(a) and bool(b))

    def op_or(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(bool(a) or bool(b))

    def op_not(self, arg):
        a = self.operand_stack.pop()
        self.operand_stack.append(not a)

    def op_jmp(self, arg):
        self.ip = arg

    def op_jmp_if_false(self, arg):
        cond = self.operand_stack.pop()
        if not cond:
            self.ip = arg

    def op_jmp_if_true(self, arg):
        cond = self.operand_stack.pop()
        if cond:
            self.ip = arg

    def op_call(self, arg):
        func_target, func_name, num_args = arg
        
        args = []
        pop = self.operand_stack.pop
        for _ in range(num_args):
            args.append(pop())
        args.reverse()

        # Direct standard library function mappings
        std_mapping = {
            "sin": "std.math.sin", "cos": "std.math.cos", "tan": "std.math.tan",
            "sqrt": "std.math.sqrt", "log": "std.math.log", "exp": "std.math.exp", "abs": "std.math.abs",
            "mean": "std.stats.mean", "variance": "std.stats.variance",
            "rand_float": "std.random.rand_float", "rand_int": "std.random.rand_int",
            "append_int": "std.collections.append_int", "remove_at": "std.collections.remove_at",
            "read_file": "std.io.read_file", "write_file": "std.io.write_file", "print_format": "std.io.print_format",
            "now": "std.time.now", "sleep": "std.time.sleep",
            "concat": "std.string.concat", "format_int": "std.string.format_int"
        }

        # Check standard library redirection
        target_name = func_name
        if func_name in std_mapping:
            target_name = std_mapping[func_name]
            func_target = target_name

        # Handle native standard library functions
        if isinstance(func_target, str) or (func_target is None and target_name and target_name.startswith("std.")):
            name = func_target if isinstance(func_target, str) else target_name
            res = self.execute_native_stdlib(name, args)
            self.operand_stack.append(res)
            return

        if len(self.call_stack) >= 1000:
            self.throw_exception("StackOverflowError: Maximum recursion depth (1000) exceeded.")
            return

        new_frame = self.get_frame(self.ip, func_name)
        self.call_stack.append(new_frame)

        append = self.operand_stack.append
        for a in args:
            append(a)

        self.ip = func_target

    def op_ret(self, arg):
        if not self.call_stack:
            self.throw_exception("StackUnderflowError: Call stack is empty on RET.")
            return
        val = self.operand_stack.pop()
        old_frame = self.call_stack.pop()
        self.ip = old_frame.return_address
        self.operand_stack.append(val)
        self.recycle_frame(old_frame)

    def op_enter_frame(self, arg):
        pass

    def op_exit_frame(self, arg):
        pass

    def op_alloc_struct(self, arg):
        field_names = arg
        field_vals = []
        pop = self.operand_stack.pop
        for _ in range(len(field_names)):
            field_vals.append(pop())
        field_vals.reverse()

        data = {name: val for name, val in zip(field_names, field_vals)}
        ref = self.allocate_heap("struct", data)
        self.operand_stack.append(ref)

    def op_get_field(self, arg):
        field_name = arg
        ref = self.operand_stack.pop()
        if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
            self.throw_exception(f"NullPointerReference: Cannot access field '{field_name}' on non-struct {ref}")
            return
        obj = self.heap[ref.ref_id]
        if obj.obj_type != "struct":
            self.throw_exception(f"TypeError: Attempted field access on non-struct heap object of type {obj.obj_type}")
            return
        if field_name not in obj.data:
            self.throw_exception(f"AttributeError: Struct has no field '{field_name}'")
            return
        self.operand_stack.append(obj.data[field_name])

    def op_set_field(self, arg):
        field_name = arg
        val = self.operand_stack.pop()
        ref = self.operand_stack.pop()
        if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
            self.throw_exception(f"NullPointerReference: Cannot set field '{field_name}' on non-struct {ref}")
            return
        obj = self.heap[ref.ref_id]
        if obj.obj_type != "struct":
            self.throw_exception(f"TypeError: Attempted field set on non-struct heap object of type {obj.obj_type}")
            return
        obj.data[field_name] = val

    def op_alloc_map(self, arg):
        num_pairs = arg
        pairs = []
        pop = self.operand_stack.pop
        for _ in range(num_pairs):
            val = pop()
            key = pop()
            pairs.append((key, val))
        
        data = {}
        for key, val in reversed(pairs):
            data[key] = val
        ref = self.allocate_heap("map", data)
        self.operand_stack.append(ref)

    def op_alloc_array(self, arg):
        num_elems = arg
        elems = []
        pop = self.operand_stack.pop
        for _ in range(num_elems):
            elems.append(pop())
        elems.reverse()
        ref = self.allocate_heap("array", elems)
        self.operand_stack.append(ref)

    def op_len(self, arg):
        ref = self.operand_stack.pop()
        if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
            self.throw_exception(f"NullPointerReference: Cannot perform LEN on {ref}")
            return
        obj = self.heap[ref.ref_id]
        self.operand_stack.append(len(obj.data))

    def op_get_index(self, arg):
        idx = self.operand_stack.pop()
        ref = self.operand_stack.pop()
        if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
            self.throw_exception(f"NullPointerReference: Cannot perform GET_INDEX on {ref}")
            return
        obj = self.heap[ref.ref_id]
        if obj.obj_type == "array":
            if not isinstance(idx, int) or idx < 0 or idx >= len(obj.data):
                self.throw_exception(f"IndexError: Array index {idx} out of range (length {len(obj.data)})")
                return
            self.operand_stack.append(obj.data[idx])
        elif obj.obj_type == "map":
            if idx not in obj.data:
                self.throw_exception(f"KeyError: Map does not contain key {idx}")
                return
            self.operand_stack.append(obj.data[idx])
        else:
            self.throw_exception(f"TypeError: Index access not supported on type {obj.obj_type}")

    def op_set_index(self, arg):
        val = self.operand_stack.pop()
        idx = self.operand_stack.pop()
        ref = self.operand_stack.pop()
        if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
            self.throw_exception(f"NullPointerReference: Cannot perform SET_INDEX on {ref}")
            return
        obj = self.heap[ref.ref_id]
        if obj.obj_type == "array":
            if not isinstance(idx, int) or idx < 0 or idx >= len(obj.data):
                self.throw_exception(f"IndexError: Array index {idx} out of range (length {len(obj.data)})")
                return
            obj.data[idx] = val
        elif obj.obj_type == "map":
            obj.data[idx] = val
        else:
            self.throw_exception(f"TypeError: Index set not supported on type {obj.obj_type}")

    def op_throw(self, arg):
        val = self.operand_stack.pop()
        self.throw_exception(val)

    def op_push_try(self, arg):
        if self.call_stack:
            self.call_stack[-1].try_stack.append((arg, len(self.operand_stack)))

    def op_pop_try(self, arg):
        if self.call_stack and self.call_stack[-1].try_stack:
            self.call_stack[-1].try_stack.pop()

    def op_q_alloc(self, arg):
        qname = self.lookup_var(arg)
        self.simulator.allocate_qubit(qname)
        self.log_trace(f"Allocated qubit: '{qname}'")

    def op_q_gate(self, arg):
        gate_name, targets = arg
        angles = []
        if gate_name in ("RX", "RY", "RZ"):
            angles.append(self.operand_stack.pop())
        
        resolved_targets = [self.lookup_var(t) for t in targets]

        if gate_name == 'H':
            self.simulator.H(resolved_targets[0])
        elif gate_name == 'X':
            self.simulator.X(resolved_targets[0])
        elif gate_name == 'Y':
            self.simulator.Y(resolved_targets[0])
        elif gate_name == 'Z':
            self.simulator.Z(resolved_targets[0])
        elif gate_name == 'S':
            self.simulator.S(resolved_targets[0])
        elif gate_name == 'T':
            self.simulator.T(resolved_targets[0])
        elif gate_name == 'RX':
            self.simulator.RX(resolved_targets[0], angles[0])
        elif gate_name == 'RY':
            self.simulator.RY(resolved_targets[0], angles[0])
        elif gate_name == 'RZ':
            self.simulator.RZ(resolved_targets[0], angles[0])
        elif gate_name == 'CNOT':
            self.simulator.CNOT(resolved_targets[0], resolved_targets[1])
        elif gate_name == 'CZ':
            self.simulator.CZ(resolved_targets[0], resolved_targets[1])
        elif gate_name == 'SWAP':
            self.simulator.SWAP(resolved_targets[0], resolved_targets[1])
        else:
            self.throw_exception(f"UnknownGateException: {gate_name}")
            return

        # Apply global gate noise if active
        for target in resolved_targets:
            self.noise_model.apply_gate_noise(self.simulator, target)

        args_str = f"({', '.join(map(str, angles))})" if angles else ""
        self.log_trace(f"Applied gate: {gate_name}{args_str} on {', '.join(resolved_targets)}")
        self.log_trace(f"  Current Quantum State: {self.format_amplitudes()}")

    def op_q_measure(self, arg):
        qubit_name, cbit_name = arg
        resolved_q = self.lookup_var(qubit_name)
        outcome = self.simulator.measure(resolved_q)
        outcome = self.noise_model.apply_readout_noise(outcome)
        
        if self.call_stack:
            self.call_stack[-1].locals[cbit_name] = outcome
        else:
            self.globals[cbit_name] = outcome
            
        self.log_trace(f"Measured qubit '{resolved_q}' -> stored in cbit '{cbit_name}' (value: {outcome})")
        self.log_trace(f"  Current Quantum State: {self.format_amplitudes()}")

    def op_q_noise(self, arg):
        noise_type, targets = arg
        p = self.operand_stack.pop()
        resolved_targets = [self.lookup_var(t) for t in targets]
        
        for target in resolved_targets:
            r = random.random()
            if noise_type == "bitflip":
                if r < p:
                    self.simulator.X(target)
                    self.log_trace(f"Applied bitflip noise (X) on '{target}'")
            elif noise_type == "depolarizing":
                if r < p:
                    r_dep = random.random()
                    if r_dep < 1/3:
                        self.simulator.X(target)
                        self.log_trace(f"Applied depolarizing noise (X) on '{target}'")
                    elif r_dep < 2/3:
                        self.simulator.Y(target)
                        self.log_trace(f"Applied depolarizing noise (Y) on '{target}'")
                    else:
                        self.simulator.Z(target)
                        self.log_trace(f"Applied depolarizing noise (Z) on '{target}'")

    def op_q_trace(self, arg):
        print(f"[TRACE DIRECTIVE] Quantum State: {self.format_amplitudes()}")

    def op_print(self, arg):
        val = self.operand_stack.pop()
        print(f"[PRINT DIRECTIVE] {val}")

    def op_spawn(self, arg):
        func_target, func_name, num_args = arg
        args = []
        for _ in range(num_args):
            args.append(self.operand_stack.pop())
        args.reverse()

        # Store task info for later JOIN execution
        if not hasattr(self, '_pending_tasks'):
            self._pending_tasks = []
        self._pending_tasks.append((func_target, func_name, args))

    def op_join(self, arg):
        num_tasks = arg
        if not hasattr(self, '_pending_tasks') or not self._pending_tasks:
            return

        import concurrent.futures
        tasks = self._pending_tasks[:num_tasks]
        self._pending_tasks = self._pending_tasks[num_tasks:]

        def run_task(func_target, func_name, args):
            if isinstance(func_target, str) or (func_target is None and func_name and func_name.startswith("std.")):
                name = func_target if isinstance(func_target, str) else func_name
                return self.execute_native_stdlib(name, args)
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
            futures = []
            for func_target, func_name, args in tasks:
                futures.append(executor.submit(run_task, func_target, func_name, args))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.throw_exception(f"ParallelTaskError: {e}")

    def execute(self, instructions: list[Instruction]):
        if native is not None and hasattr(native, 'execute_bytecode_native'):
            supported = {"LOAD_CONST", "STORE_VAR", "LOAD_VAR", "ADD", "SUB", "MUL", "DIV", "EQ", "NEQ", "JMP", "JMP_IF_FALSE", "PRINT", "HALT"}
            if all(inst.opcode in supported for inst in instructions):
                py_instrs = [(inst.opcode, inst.arg) for inst in instructions]
                self.globals = native.execute_bytecode_native(py_instrs, self.globals)
                return

        self.instructions = instructions
        self.ip = 0
        self.operand_stack = []
        self.call_stack = [self.get_frame(None, "main")]
        self.globals = {}
        self.heap = {}
        self.next_ref_id = 1

        self.log_trace("Starting execution of Eigen VM bytecode")

        # Localize hot properties and stack operations
        dispatch = self.dispatch_table
        pop = self.operand_stack.pop
        append = self.operand_stack.append

        while self.ip < len(self.instructions):
            if self.jit_enabled:
                compiled_func = self.jit.check_and_compile(self.ip, self.instructions)
                if compiled_func:
                    try:
                        res = compiled_func(self.operand_stack, self.globals, self.globals, self.lookup_var, self)
                        if res:
                            continue
                    except Exception as e:
                        # Fallback: execute normally or raise
                        raise e

            instr = self.instructions[self.ip]
            self.ip += 1

            if self.call_stack and instr.line is not None:
                self.call_stack[-1].current_line = instr.line

            opcode = instr.opcode
            arg = instr.arg

            try:
                # Fast-path for extremely common operations
                if opcode == Opcode.LOAD_CONST:
                    append(arg)
                elif opcode == Opcode.LOAD_VAR:
                    append(self.lookup_var(arg))
                elif opcode == Opcode.STORE_VAR:
                    val = pop()
                    if self.call_stack:
                        self.call_stack[-1].locals[arg] = val
                    else:
                        self.globals[arg] = val
                elif opcode == Opcode.ADD:
                    b = pop()
                    a = pop()
                    append(a + b)
                elif opcode == Opcode.SUB:
                    b = pop()
                    a = pop()
                    append(a - b)
                else:
                    # Table dispatch
                    if dispatch[opcode](arg):
                        break
            except IndexError:
                self.throw_exception("StackUnderflowError: Operand stack underflow.")

        self.log_trace("Finished execution of Eigen VM bytecode")

    def execute_native_stdlib(self, func_name: str, args: list) -> any:
        import importlib.util
        import sys
        import os
        
        parts = func_name.split('.')
        if len(parts) < 3:
            raise ValueError(f"Invalid stdlib function call: {func_name}")
            
        module_subname = parts[1]
        func_subname = parts[2]
        
        module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "stdlib", "native", f"{module_subname}.py"))
        if not os.path.isfile(module_path):
            raise FileNotFoundError(f"Native stdlib module {module_subname} not found at {module_path}")
            
        module_qualname = f"stdlib.native.{module_subname}"
        if module_qualname in sys.modules:
            mod = sys.modules[module_qualname]
        else:
            spec = importlib.util.spec_from_file_location(module_qualname, module_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_qualname] = mod
            spec.loader.exec_module(mod)
            
        if not hasattr(mod, func_subname):
            if func_subname == 'abs' and hasattr(mod, 'abs_val'):
                func_subname = 'abs_val'
            else:
                raise AttributeError(f"Module {module_subname} has no function {func_subname}")
            
        native_fn = getattr(mod, func_subname)
        
        # Unpack VMRef arguments
        unpacked_args = []
        for a in args:
            if isinstance(a, VMRef) and a.ref_id in self.heap:
                obj = self.heap[a.ref_id]
                unpacked_args.append(obj.data)
            else:
                unpacked_args.append(a)
                
        # Call
        res = native_fn(*unpacked_args)
        
        # Pack results
        if isinstance(res, str):
            return self.allocate_heap('string', res)
        elif isinstance(res, list):
            return self.allocate_heap('array', res)
        elif isinstance(res, dict):
            return self.allocate_heap('map', res)
        return res
