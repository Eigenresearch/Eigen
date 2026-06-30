from src.backend.bytecode import Opcode, Instruction, UnsupportedBytecodeVersionError
from src.simulator import QuantumSimulator
import random
import re
import weakref
import threading
import concurrent.futures

VAR_PATTERN = re.compile(r'^[qc](_?\d+)?(_\d+)?$')
_PATTERN_CACHE = {}
_PATTERN_CACHE_MAX = 4096

try:
    import eigen_native as native
except ImportError:
    native = None

class UndefinedVariableError(Exception):
    pass

class VMRef:
    def __init__(self, ref_id: int, heap_obj=None):
        self.ref_id = ref_id
        self.heap_obj = heap_obj

    def __repr__(self) -> str:
        return f"Ref({self.ref_id})"

    def __eq__(self, other) -> bool:
        if isinstance(other, VMRef):
            return self.ref_id == other.ref_id
        return False

    def __hash__(self) -> int:
        return hash(self.ref_id)


class HeapObject:
    __slots__ = ('obj_type', 'data', '__weakref__')

    def __init__(self, obj_type: str, data):
        self.obj_type = obj_type  # 'struct', 'map', 'array', 'string'
        self.data = data          # dict, list, or str

    def __repr__(self) -> str:
        return f"HeapObject({self.obj_type}, {self.data})"


class ActivationFrame:
    __slots__ = ('locals', 'try_stack', 'return_address', 'current_line', 'func_name')

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
    _STD_MAPPING = {
        "sin": "std.math.sin", "cos": "std.math.cos", "tan": "std.math.tan",
        "sqrt": "std.math.sqrt", "log": "std.math.log", "exp": "std.math.exp", "abs": "std.math.abs",
        "mean": "std.stats.mean", "variance": "std.stats.variance",
        "rand_float": "std.random.rand_float", "rand_int": "std.random.rand_int",
        "append_int": "std.collections.append_int", "remove_at": "std.collections.remove_at",
        "read_file": "std.io.read_file", "write_file": "std.io.write_file", "print_format": "std.io.print_format",
        "now": "std.time.now", "sleep": "std.time.sleep",
        "concat": "std.string.concat", "format_int": "std.string.format_int"
    }

    def __init__(self, trace_mode: bool = False, noise_model=None, sim_type: str = 'dense', gpu_platform: str = 'none', seed: int | None = None, verbose: bool = False, opt_level: int = 3):
        self.rng = random.Random(seed)
        from src.noise.noise_model import NoiseModel
        self.simulator = QuantumSimulator(sim_type=sim_type, gpu_platform=gpu_platform, seed=seed)
        self.trace_mode = trace_mode
        self.trace_log = []
        self.noise_model = noise_model if noise_model is not None else NoiseModel(rng=self.rng)
        if getattr(self.noise_model, 'rng', None) is None:
            self.noise_model.rng = self.rng
        
        self.verbose = verbose
        self.output_stream = None
        
        # Trace-Based Adaptive Execution Engine
        from src.jit.jit_compiler import JITCompiler
        self.jit = JITCompiler(self)
        self.jit_enabled = (opt_level >= 3)
        self.jit_hits = 0
        self.jit_deopts = 0
        
        # VM registers and stacks
        self.instructions = []
        self.ip = 0
        self.operand_stack = []
        self.call_stack = []
        self.try_stack = []
        self.globals = {}
        self.frame_pool = []
        
        # Heap
        self.heap = weakref.WeakValueDictionary()
        self.heap_lock = threading.Lock()
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
            Opcode.MOD: self.op_mod,
            Opcode.POW: self.op_pow,
            Opcode.BIT_AND: self.op_bit_and,
            Opcode.BIT_OR: self.op_bit_or,
            Opcode.BIT_XOR: self.op_bit_xor,
            Opcode.BIT_NOT: self.op_bit_not,
            Opcode.SHL: self.op_shl,
            Opcode.SHR: self.op_shr,
            Opcode.LOAD_CONST_STORE: self.op_load_const_store,
            Opcode.LOAD_VAR_LOAD_CONST_ADD: self.op_load_var_load_const_add,
            Opcode.LOAD_VAR_LOAD_CONST_SUB: self.op_load_var_load_const_sub,
            Opcode.LOAD_VAR_LOAD_CONST_LT: self.op_load_var_load_const_lt,
            Opcode.LOAD_VAR_LOAD_CONST_GT: self.op_load_var_load_const_gt,
            Opcode.LOAD_VAR_LOAD_CONST_LTE: self.op_load_var_load_const_lte,
            Opcode.LOAD_VAR_LOAD_CONST_GTE: self.op_load_var_load_const_gte,
        }

        from src.backend.bytecode import OPCODE_TO_INT
        self.OP_LOAD_CONST_STORE = OPCODE_TO_INT.get(Opcode.LOAD_CONST_STORE)
        self.OP_LOAD_VAR_LOAD_CONST_ADD = OPCODE_TO_INT.get(Opcode.LOAD_VAR_LOAD_CONST_ADD)
        self.OP_LOAD_VAR_LOAD_CONST_SUB = OPCODE_TO_INT.get(Opcode.LOAD_VAR_LOAD_CONST_SUB)
        self.OP_LOAD_VAR_LOAD_CONST_LT = OPCODE_TO_INT.get(Opcode.LOAD_VAR_LOAD_CONST_LT)
        self.OP_LOAD_VAR_LOAD_CONST_GT = OPCODE_TO_INT.get(Opcode.LOAD_VAR_LOAD_CONST_GT)
        self.OP_LOAD_VAR_LOAD_CONST_LTE = OPCODE_TO_INT.get(Opcode.LOAD_VAR_LOAD_CONST_LTE)
        self.OP_LOAD_VAR_LOAD_CONST_GTE = OPCODE_TO_INT.get(Opcode.LOAD_VAR_LOAD_CONST_GTE)
        self.OP_LOAD_CONST = OPCODE_TO_INT.get(Opcode.LOAD_CONST)
        self.OP_LOAD_VAR = OPCODE_TO_INT.get(Opcode.LOAD_VAR)
        self.OP_STORE_VAR = OPCODE_TO_INT.get(Opcode.STORE_VAR)
        self.OP_ADD = OPCODE_TO_INT.get(Opcode.ADD)
        self.OP_SUB = OPCODE_TO_INT.get(Opcode.SUB)
        self.OP_MUL = OPCODE_TO_INT.get(Opcode.MUL)
        self.OP_DIV = OPCODE_TO_INT.get(Opcode.DIV)
        self.OP_EQ = OPCODE_TO_INT.get(Opcode.EQ)
        self.OP_LT = OPCODE_TO_INT.get(Opcode.LT)
        self.OP_GT = OPCODE_TO_INT.get(Opcode.GT)
        self.OP_JMP = OPCODE_TO_INT.get(Opcode.JMP)
        self.OP_JMP_IF_FALSE = OPCODE_TO_INT.get(Opcode.JMP_IF_FALSE)
        self.OP_CALL = OPCODE_TO_INT.get(Opcode.CALL)
        self.OP_RET = OPCODE_TO_INT.get(Opcode.RET)
        self.OP_PRINT = OPCODE_TO_INT.get(Opcode.PRINT)

        self.dispatch_list = [None] * len(OPCODE_TO_INT)
        for op, func in self.dispatch_table.items():
            if op in OPCODE_TO_INT:
                self.dispatch_list[OPCODE_TO_INT[op]] = func

    def log_trace(self, msg: str):
        if self.trace_mode:
            self.trace_log.append(msg)
            if len(self.trace_log) > 10000:
                del self.trace_log[:5000]
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

    def run_compiled_block(self, compiled_func) -> bool:
        locals_map = self.call_stack[-1].locals if self.call_stack else self.globals
        try:
            res = compiled_func(self.operand_stack, locals_map, self.globals, self.lookup_var, self)
            if res:
                self.jit_deopts += 1
            return True
        except IndexError:
            self.jit_deopts += 1
            self.throw_exception("StackUnderflowError: Operand stack underflow.")
            return True

    def lookup_var(self, name: str):
        if not isinstance(name, str):
            return name
            
        frame_locals = None
        if self.call_stack:
            frame = self.call_stack[-1]
            frame_locals = frame.locals
            if name in frame_locals:
                return frame_locals[name]
        if name in self.globals:
            return self.globals[name]
            
        is_lit = _PATTERN_CACHE.get(name)
        if is_lit is None:
            is_lit = bool(VAR_PATTERN.match(name))
            if len(_PATTERN_CACHE) < _PATTERN_CACHE_MAX:
                _PATTERN_CACHE[name] = is_lit
        if is_lit:
            return name
            
        raise UndefinedVariableError(f"Variable '{name}' is not defined.")

    def throw_exception(self, val):
        if self.try_stack:
            handler_ip, saved_stack_depth, saved_call_depth = self.try_stack.pop()
            
            # Pop try-blocks pushed inside call frames we are about to discard
            while self.try_stack and self.try_stack[-1][2] > saved_call_depth:
                self.try_stack.pop()
                
            while len(self.call_stack) > saved_call_depth:
                self.call_stack.pop()
                
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
        with self.heap_lock:
            ref_id = self.next_ref_id
            self.next_ref_id += 1
            obj = HeapObject(obj_type, data)
            self.heap[ref_id] = obj
            return VMRef(ref_id, obj)

    def get_frame(self, return_address, func_name):
        if self.frame_pool:
            frame = self.frame_pool.pop()
            frame.reset(return_address, func_name)
            return frame
        return ActivationFrame(return_address, func_name)

    def recycle_frame(self, frame):
        if len(self.frame_pool) < 64:
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

    def op_mod(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        if b == 0:
            self.throw_exception("DivisionByZeroError: Division by zero.")
            return
        self.operand_stack.append(a % b)

    def op_pow(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a ** b)

    def op_bit_and(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a & b)

    def op_bit_or(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a | b)

    def op_bit_xor(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a ^ b)

    def op_bit_not(self, arg):
        a = self.operand_stack.pop()
        self.operand_stack.append(~a)

    def op_shl(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a << b)

    def op_shr(self, arg):
        b = self.operand_stack.pop()
        a = self.operand_stack.pop()
        self.operand_stack.append(a >> b)

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

    def op_load_const_store(self, arg):
        const_val, var_name = arg
        if self.call_stack:
            self.call_stack[-1].locals[var_name] = const_val
        else:
            self.globals[var_name] = const_val

    def op_load_var_load_const_add(self, arg):
        var_name, const_val = arg
        self.operand_stack.append(self.lookup_var(var_name) + const_val)

    def op_load_var_load_const_sub(self, arg):
        var_name, const_val = arg
        self.operand_stack.append(self.lookup_var(var_name) - const_val)

    def op_load_var_load_const_lt(self, arg):
        var_name, const_val = arg
        self.operand_stack.append(self.lookup_var(var_name) < const_val)

    def op_load_var_load_const_gt(self, arg):
        var_name, const_val = arg
        self.operand_stack.append(self.lookup_var(var_name) > const_val)

    def op_load_var_load_const_lte(self, arg):
        var_name, const_val = arg
        self.operand_stack.append(self.lookup_var(var_name) <= const_val)

    def op_load_var_load_const_gte(self, arg):
        var_name, const_val = arg
        self.operand_stack.append(self.lookup_var(var_name) >= const_val)

    def op_jmp(self, arg):
        target = arg
        if target < self.ip and self.jit_enabled:
            compiled_func = self.jit.check_and_compile(target, self.instructions)
            if compiled_func:
                self.ip = target
                self.run_compiled_block(compiled_func)
                return
        self.ip = target

    def op_jmp_if_false(self, arg):
        cond = self.operand_stack.pop()
        if not cond:
            target = arg
            if target < self.ip and self.jit_enabled:
                compiled_func = self.jit.check_and_compile(target, self.instructions)
                if compiled_func:
                    self.ip = target
                    self.run_compiled_block(compiled_func)
                    return
            self.ip = target

    def op_jmp_if_true(self, arg):
        cond = self.operand_stack.pop()
        if cond:
            target = arg
            if target < self.ip and self.jit_enabled:
                compiled_func = self.jit.check_and_compile(target, self.instructions)
                if compiled_func:
                    self.ip = target
                    self.run_compiled_block(compiled_func)
                    return
            self.ip = target

    def op_call(self, arg):
        func_target, func_name, num_args = arg
        
        args = []
        pop = self.operand_stack.pop
        for _ in range(num_args):
            args.append(pop())
        args.reverse()

        # Check standard library redirection
        target_name = func_name
        if func_name in self._STD_MAPPING:
            target_name = self._STD_MAPPING[func_name]
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
        
        current_depth = len(self.call_stack)
        while self.try_stack and self.try_stack[-1][2] >= current_depth:
            self.try_stack.pop()

        if not self.operand_stack:
            val = None
        else:
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
        self.try_stack.append((arg, len(self.operand_stack), len(self.call_stack)))

    def op_pop_try(self, arg):
        if self.try_stack:
            self.try_stack.pop()

    def op_q_alloc(self, arg):
        qname = self.lookup_var(arg)
        self.simulator.allocate_qubit(qname)
        self.log_trace(f"Allocated qubit: '{qname}'")

    def op_q_gate(self, arg):
        gate_name, targets = arg
        angles = []
        if gate_name in ("RX", "RY", "RZ", "CP", "CRX", "CRY", "CRZ"):
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
        elif gate_name == 'CCX':
            self.simulator.CCX(resolved_targets[0], resolved_targets[1], resolved_targets[2])
        elif gate_name == 'CSWAP':
            self.simulator.CSWAP(resolved_targets[0], resolved_targets[1], resolved_targets[2])
        elif gate_name == 'CP':
            self.simulator.CP(resolved_targets[0], resolved_targets[1], angles[0])
        elif gate_name == 'CRX':
            self.simulator.CRX(resolved_targets[0], resolved_targets[1], angles[0])
        elif gate_name == 'CRY':
            self.simulator.CRY(resolved_targets[0], resolved_targets[1], angles[0])
        elif gate_name == 'CRZ':
            self.simulator.CRZ(resolved_targets[0], resolved_targets[1], angles[0])
        else:
            self.throw_exception(f"UnknownGateException: {gate_name}")
            return

        # Apply global gate noise if active
        for target in resolved_targets:
            self.noise_model.apply_gate_noise(self.simulator, target)

        args_str = f"({', '.join(map(str, angles))})" if angles else ""
        self.log_trace(f"Applied gate: {gate_name}{args_str} on {', '.join(resolved_targets)}")
        if self.trace_mode:
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
        if self.trace_mode:
            self.log_trace(f"  Current Quantum State: {self.format_amplitudes()}")

    def op_q_noise(self, arg):
        noise_type, targets = arg
        p = self.operand_stack.pop()
        resolved_targets = [self.lookup_var(t) for t in targets]
        
        for target in resolved_targets:
            r = self.rng.random()
            if noise_type == "bitflip":
                if r < p:
                    self.simulator.X(target)
                    self.log_trace(f"Applied bitflip noise (X) on '{target}'")
            elif noise_type == "depolarizing":
                if r < p:
                    r_dep = self.rng.random()
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
        output = f"Quantum State: {self.format_amplitudes()}"
        if self.output_stream is not None:
            self.output_stream.write(f"{output}\n")
        else:
            if getattr(self, "verbose", False):
                print(f"[TRACE DIRECTIVE] {output}")
            else:
                print(output)

    def op_print(self, arg):
        val = self.operand_stack.pop()
        if isinstance(val, bool):
            out = "true" if val else "false"
        elif isinstance(val, float):
            if val == int(val) and abs(val) < 1e16:
                out = f"{int(val)}.0"
            elif abs(val) < 1e-4 and val != 0.0:
                out = f"{val:.15f}".rstrip('0').rstrip('.')
            else:
                out = repr(val)
        elif val is None:
            out = "null"
        else:
            out = str(val)
        if self.output_stream is not None:
            self.output_stream.write(f"{out}\n")
        else:
            if getattr(self, "verbose", False):
                print(f"[PRINT DIRECTIVE] {out}")
            else:
                print(out)

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

    def _try_fast_loop(self, instructions) -> bool:
        """Detect simple tight loops and compile them to register-based Python for near-native speed."""
        n = len(instructions)
        if n < 6:
            return False

        jmp_ops = {"JMP", "JMP_IF_FALSE", "JMP_IF_TRUE"}
        back_jump_idx = None
        back_jump_target = None
        for i, inst in enumerate(instructions):
            if inst.opcode in jmp_ops and isinstance(inst.arg, int) and inst.arg < i:
                back_jump_idx = i
                back_jump_target = inst.arg
                break

        if back_jump_idx is None:
            return False

        cond_check_idx = None
        for i in range(back_jump_target, back_jump_idx):
            inst = instructions[i]
            if inst.opcode == "JMP_IF_FALSE" and isinstance(inst.arg, int):
                cond_check_idx = i
                loop_body_start = i + 1
                exit_target = inst.arg
                break

        if cond_check_idx is None:
            return False

        # Only support pure register operations (no stack needed for simple loops)
        # Condition must be: LOAD_VAR var, LOAD_CONST N, CMP_OP, JMP_IF_FALSE
        cond_expr_insts = instructions[back_jump_target:cond_check_idx]
        if len(cond_expr_insts) != 3:
            return False
        if cond_expr_insts[0].opcode != "LOAD_VAR":
            return False
        if cond_expr_insts[1].opcode != "LOAD_CONST":
            return False
        if cond_expr_insts[2].opcode not in ("LT", "GT", "LTE", "GTE", "EQ", "NEQ"):
            return False

        loop_var = cond_expr_insts[0].arg
        limit_val = cond_expr_insts[1].arg
        cmp_op = cond_expr_insts[2].opcode

        # Parse loop body into simple register operations
        # Each instruction must be: LOAD_VAR, LOAD_CONST, ADD/SUB, STORE_VAR
        body_insts = instructions[loop_body_start:back_jump_idx]
        if not body_insts:
            return False

        # Check all body instructions are supported
        supported_body = {"LOAD_VAR", "LOAD_CONST", "ADD", "SUB", "MUL", "DIV",
                          "STORE_VAR", "LOAD_VAR_LOAD_CONST_ADD", "LOAD_VAR_LOAD_CONST_SUB",
                          "LOAD_CONST_STORE", "JMP"}
        for inst in body_insts:
            if inst.opcode not in supported_body:
                return False

        # Collect all variables
        all_vars = set()
        all_vars.add(loop_var)
        for inst in body_insts:
            if inst.opcode in ("LOAD_VAR", "STORE_VAR"):
                all_vars.add(inst.arg)
            elif inst.opcode == "LOAD_CONST_STORE":
                all_vars.add(inst.arg[1])
            elif inst.opcode in ("LOAD_VAR_LOAD_CONST_ADD", "LOAD_VAR_LOAD_CONST_SUB"):
                all_vars.add(inst.arg[0])

        clean = lambda v: "".join(c if c.isalnum() else "_" for c in v)

        # Build register-based Python code (no stack!)
        lines = []
        lines.append("def _fast_loop(_globals, _locals):")
        for v in sorted(all_vars):
            cn = clean(v)
            lines.append(f"    r_{cn} = _locals.get({repr(v)}, _globals.get({repr(v)}, 0))")

        cmp_map = {"LT": "<", "GT": ">", "LTE": "<=", "GTE": ">=", "EQ": "==", "NEQ": "!="}

        # Generate while loop
        lines.append(f"    while r_{clean(loop_var)} {cmp_map[cmp_op]} {repr(limit_val)}:")

        # Translate body instructions to register operations
        # We use a simple expression stack for intermediate values
        stack = []
        for inst in body_insts:
            if inst.opcode == "LOAD_CONST":
                stack.append(repr(inst.arg))
            elif inst.opcode == "LOAD_VAR":
                stack.append(f"r_{clean(inst.arg)}")
            elif inst.opcode == "STORE_VAR":
                val = stack.pop()
                lines.append(f"        r_{clean(inst.arg)} = {val}")
            elif inst.opcode == "ADD":
                b = stack.pop(); a = stack.pop()
                stack.append(f"({a} + {b})")
            elif inst.opcode == "SUB":
                b = stack.pop(); a = stack.pop()
                stack.append(f"({a} - {b})")
            elif inst.opcode == "MUL":
                b = stack.pop(); a = stack.pop()
                stack.append(f"({a} * {b})")
            elif inst.opcode == "DIV":
                b = stack.pop(); a = stack.pop()
                stack.append(f"({a} / {b})")
            elif inst.opcode == "LOAD_VAR_LOAD_CONST_ADD":
                stack.append(f"(r_{clean(inst.arg[0])} + {repr(inst.arg[1])})")
            elif inst.opcode == "LOAD_VAR_LOAD_CONST_SUB":
                stack.append(f"(r_{clean(inst.arg[0])} - {repr(inst.arg[1])})")
            elif inst.opcode == "LOAD_CONST_STORE":
                lines.append(f"        r_{clean(inst.arg[1])} = {repr(inst.arg[0])}")
            elif inst.opcode == "JMP":
                pass

        # Write back variables after loop
        for v in sorted(all_vars):
            cn = clean(v)
            lines.append(f"    _locals[{repr(v)}] = r_{cn}")
            lines.append(f"    _globals[{repr(v)}] = r_{cn}")

        source = "\n".join(lines)

        try:
            code_obj = compile(source, '<fast_loop>', 'exec')
            safe_globals = {
                "__builtins__": {},
                "range": range, "abs": abs, "len": len,
                "int": int, "float": float, "str": str, "bool": bool,
                "repr": repr, "isinstance": isinstance, "hasattr": hasattr,
                "getattr": getattr, "type": type,
            }
            local_vars = {}
            exec(code_obj, safe_globals, local_vars)
            fast_func = local_vars['_fast_loop']

            # Execute pre-loop instructions (initialize variables)
            for i in range(0, back_jump_target):
                inst = instructions[i]
                if inst.opcode == "LOAD_CONST":
                    self.operand_stack.append(inst.arg)
                elif inst.opcode == "STORE_VAR":
                    val = self.operand_stack.pop()
                    if self.call_stack:
                        self.call_stack[-1].locals[inst.arg] = val
                    else:
                        self.globals[inst.arg] = val
                elif inst.opcode == "LOAD_CONST_STORE":
                    cv, vn = inst.arg
                    if self.call_stack:
                        self.call_stack[-1].locals[vn] = cv
                    else:
                        self.globals[vn] = cv
                else:
                    return False

            # Run the fast loop!
            locals_map = self.call_stack[-1].locals if self.call_stack else self.globals
            fast_func(self.globals, locals_map)

            # Execute epilogue
            self.instructions = instructions
            ip = exit_target
            while ip < n:
                inst = instructions[ip]
                ip += 1
                self.ip = ip
                if inst.opcode == "LOAD_VAR":
                    self.operand_stack.append(self.lookup_var(inst.arg))
                elif inst.opcode == "LOAD_CONST":
                    self.operand_stack.append(inst.arg)
                elif inst.opcode == "PRINT":
                    self.op_print(inst.arg)
                elif inst.opcode == "HALT":
                    return True
                elif inst.opcode == "STORE_VAR":
                    val = self.operand_stack.pop()
                    if self.call_stack:
                        self.call_stack[-1].locals[inst.arg] = val
                    else:
                        self.globals[inst.arg] = val
                else:
                    self.ip = ip - 1
                    return False
            return True
        except Exception:
            return False

    def execute(self, instructions: list[Instruction]):
        if getattr(self.simulator, 'sim_type', None) == 'auto':
            from src.backend.bytecode import Opcode
            n_qubits = 0
            n_2q = 0
            for inst in instructions:
                if inst.opcode == Opcode.Q_ALLOC:
                    n_qubits += 1
                elif inst.opcode == Opcode.Q_GATE:
                    if isinstance(inst.arg, tuple) and len(inst.arg) > 0:
                        gate_name = inst.arg[0]
                        if gate_name in ('CNOT', 'CZ', 'SWAP'):
                            n_2q += 1
            if n_qubits <= 16:
                chosen = 'dense'
            else:
                if n_2q < n_qubits * 1.5:
                    chosen = 'mps'
                else:
                    chosen = 'sparse'
            self.simulator.configure_backend(chosen)

        # Native Rust executor bypassed — Python VM with fast-loop JIT is faster for loops
        # and provides better print precision. Native executor kept as fallback for
        # straight-line programs without PRINT/loops if ever needed.
        _skip_native = True
        if not _skip_native and native is not None and hasattr(native, 'execute_bytecode_native'):
            supported = {
                "LOAD_CONST", "STORE_VAR", "LOAD_VAR",
                "ADD", "SUB", "MUL", "DIV",
                "EQ", "NEQ", "LT", "GT", "LTE", "GTE",
                "AND", "OR", "NOT",
                "JMP", "JMP_IF_FALSE", "JMP_IF_TRUE",
                "PRINT", "HALT",
                "MOD", "BIT_AND", "BIT_OR", "BIT_XOR", "BIT_NOT", "SHL", "SHR",
                "LOAD_CONST_STORE",
                "LOAD_VAR_LOAD_CONST_ADD", "LOAD_VAR_LOAD_CONST_SUB",
                "LOAD_VAR_LOAD_CONST_LT", "LOAD_VAR_LOAD_CONST_GT",
                "LOAD_VAR_LOAD_CONST_LTE", "LOAD_VAR_LOAD_CONST_GTE"
            }
            # Check for backward jumps (loops) — JIT is faster for loops
            has_backward_jumps = False
            jmp_opcodes = {"JMP", "JMP_IF_FALSE", "JMP_IF_TRUE"}
            for i, inst in enumerate(instructions):
                if inst.opcode in jmp_opcodes:
                    if isinstance(inst.arg, int) and inst.arg < i:
                        has_backward_jumps = True
                        break
            
            if not has_backward_jumps and all(inst.opcode in supported for inst in instructions):
                py_instrs = [(inst.opcode, inst.arg) for inst in instructions]
                try:
                    self.globals, self.operand_stack = native.execute_bytecode_native(py_instrs, self.globals)
                    return
                except UndefinedVariableError as e:
                    raise e
                except ZeroDivisionError:
                    self.throw_exception("DivisionByZeroError: Division by zero.")
                except Exception as e:
                    pass

        self.instructions = instructions
        self.ip = 0
        self.operand_stack = []
        self.call_stack = [self.get_frame(None, "main")]
        self.try_stack = []
        if not hasattr(self, 'globals') or not isinstance(self.globals, dict):
            self.globals = {}

        self.log_trace("Starting execution of Eigen VM bytecode")

        # Try fast-loop compilation: detect backward jumps and compile entire loop bodies
        if self.jit_enabled:
            if self._try_fast_loop(instructions):
                self.log_trace("Finished execution of Eigen VM bytecode (fast loop)")
                return

        # Pre-extract instruction data into parallel arrays for maximum speed
        n_instrs = len(instructions)
        _ops = [inst.opcode_int for inst in instructions]
        _args = [inst.arg for inst in instructions]
        _lines = [inst.line for inst in instructions]

        # Localize hot properties and stack operations
        dispatch = self.dispatch_list
        pop = self.operand_stack.pop
        append = self.operand_stack.append
        call_stack = self.call_stack
        globals_dict = self.globals
        _lookup = self.lookup_var
        _throw = self.throw_exception

        # Cache opcode constants as local ints for fastest comparison
        OP_LOAD_CONST = self.OP_LOAD_CONST
        OP_LOAD_VAR = self.OP_LOAD_VAR
        OP_STORE_VAR = self.OP_STORE_VAR
        OP_LCS = self.OP_LOAD_CONST_STORE
        OP_LVCA = self.OP_LOAD_VAR_LOAD_CONST_ADD
        OP_LVCS = self.OP_LOAD_VAR_LOAD_CONST_SUB
        OP_LVCLT = self.OP_LOAD_VAR_LOAD_CONST_LT
        OP_LVCGT = self.OP_LOAD_VAR_LOAD_CONST_GT
        OP_LVCLTE = self.OP_LOAD_VAR_LOAD_CONST_LTE
        OP_LVCGTE = self.OP_LOAD_VAR_LOAD_CONST_GTE
        OP_ADD = self.OP_ADD
        OP_SUB = self.OP_SUB
        OP_MUL = self.OP_MUL
        OP_DIV = self.OP_DIV
        OP_EQ = self.OP_EQ
        OP_LT = self.OP_LT
        OP_GT = self.OP_GT
        OP_JMP = self.OP_JMP
        OP_JIF = self.OP_JMP_IF_FALSE
        OP_CALL = self.OP_CALL
        OP_RET = self.OP_RET
        OP_PRINT = self.OP_PRINT
        jit_enabled = self.jit_enabled

        def run_jit(target_ip):
            compiled_func = self.jit.check_and_compile(target_ip, self.instructions)
            if compiled_func:
                self.run_compiled_block(compiled_func)
                return True
            return False

        if jit_enabled:
            if run_jit(0):
                if self.ip >= n_instrs:
                    self.log_trace("Finished execution of Eigen VM bytecode")
                    return

        ip = 0
        while ip < n_instrs:
            op = _ops[ip]
            arg = _args[ip]
            ip += 1
            self.ip = ip

            try:
                if op == OP_LOAD_CONST:
                    append(arg)
                elif op == OP_LOAD_VAR:
                    append(_lookup(arg))
                elif op == OP_STORE_VAR:
                    val = pop()
                    if call_stack:
                        call_stack[-1].locals[arg] = val
                    else:
                        globals_dict[arg] = val
                elif op == OP_LCS:
                    const_val, var_name = arg
                    if call_stack:
                        call_stack[-1].locals[var_name] = const_val
                    else:
                        globals_dict[var_name] = const_val
                elif op == OP_LVCA:
                    append(_lookup(arg[0]) + arg[1])
                elif op == OP_LVCS:
                    append(_lookup(arg[0]) - arg[1])
                elif op == OP_LVCLT:
                    append(_lookup(arg[0]) < arg[1])
                elif op == OP_LVCGT:
                    append(_lookup(arg[0]) > arg[1])
                elif op == OP_LVCLTE:
                    append(_lookup(arg[0]) <= arg[1])
                elif op == OP_LVCGTE:
                    append(_lookup(arg[0]) >= arg[1])
                elif op == OP_ADD:
                    b = pop(); a = pop(); append(a + b)
                elif op == OP_SUB:
                    b = pop(); a = pop(); append(a - b)
                elif op == OP_MUL:
                    b = pop(); a = pop(); append(a * b)
                elif op == OP_DIV:
                    b = pop(); a = pop()
                    if b == 0:
                        _throw("DivisionByZeroError: Division by zero.")
                    else:
                        append(a / b)
                elif op == OP_EQ:
                    b = pop(); a = pop(); append(a == b)
                elif op == OP_LT:
                    b = pop(); a = pop(); append(a < b)
                elif op == OP_GT:
                    b = pop(); a = pop(); append(a > b)
                elif op == OP_JMP:
                    if arg < ip and jit_enabled:
                        ip = arg; self.ip = ip
                        if run_jit(arg):
                            ip = self.ip
                            continue
                    ip = arg; self.ip = ip
                elif op == OP_JIF:
                    cond = pop()
                    if not cond:
                        if arg < ip and jit_enabled:
                            ip = arg; self.ip = ip
                            if run_jit(arg):
                                ip = self.ip
                                continue
                        ip = arg; self.ip = ip
                elif op == OP_CALL:
                    self.op_call(arg)
                    ip = self.ip
                elif op == OP_RET:
                    self.op_ret(arg)
                    ip = self.ip
                elif op == OP_PRINT:
                    self.op_print(arg)
                else:
                    if op < 0 or op >= len(dispatch):
                        _throw(f"InvalidOpcodeError: Invalid opcode {op} at IP {ip - 1}.")
                        break
                    op_func = dispatch[op]
                    if op_func is None:
                        _throw(f"InvalidOpcodeError: Unhandled or invalid opcode {op} at IP {ip - 1}.")
                        break
                    if op_func(arg):
                        break
                    ip = self.ip
            except IndexError:
                _throw("StackUnderflowError: Operand stack underflow.")
                ip = self.ip

        self.ip = ip

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
