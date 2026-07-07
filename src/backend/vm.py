from src.backend.bytecode import Opcode, Instruction, UnsupportedBytecodeVersionError
from src.simulator import QuantumSimulator
from src.backend.vm_optimizations import (
    InlineCache, HotLoopDetector, ObjectPool, FrameCache, CacheEntry,
)
from src.debugger.dap_server import DebugSession
import random
import re
import weakref
import threading
import concurrent.futures
import time as _time
import hashlib as _hashlib


def _hash_program_fallback(vm, instructions) -> str:
    """§6.4 — when the caller did not pass an explicit ``program_hash``
    to ``execute(audit=...)``, compute a stable SHA-256 over the source
    text of the supplied instructions so the audit log still groups
    consecutive runs of the same program. Falls back to hashing the
    raw ``Instruction`` reprs when no source text is available."""
    h = _hashlib.sha256()
    for inst in instructions:
        h.update(repr(inst).encode("utf-8", errors="replace"))
        h.update(b"|")
    return h.hexdigest()

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

    def __init__(self, trace_mode: bool = False, noise_model=None, sim_type: str = 'dense', gpu_platform: str = 'none', seed: int | None = None, verbose: bool = False, opt_level: int = 3, deterministic: bool = False, max_instruction_count: int | None = None, max_operand_stack_depth: int = 1 << 20, instruction_timeout_s: float | None = None, dispatch_mode: str = 'fast'):
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

        # === sol.md P0 determinism mode (§2.1) and VM hardening (§7.1) =====
        # deterministic=True guarantees that two runs of the same bytecode
        # produce byte-identical output: the RNG is seeded (caller must pass
        # ``seed`` too), the audit/profile output sorting is stable, and any
        # future non-deterministic source gates are rejected at execute() time.
        self.deterministic = deterministic
        # sol.md §1.1 — dispatch-mode switch. ``'fast'`` keeps the
        # inline if/elif chain (default for max throughput); ``'table'``
        # routes every opcode through `dispatch_table[op](arg)` — the
        # architectural form the roadmap mandates. Both modes use the
        # SAME handler methods (`op_*`) for correctness parity.
        if dispatch_mode not in ('fast', 'table'):
            raise ValueError(
                f"dispatch_mode must be 'fast' or 'table', got "
                f"{dispatch_mode!r}")
        self.dispatch_mode = dispatch_mode
        self.instruction_count = 0
        self.max_instruction_count = max_instruction_count  # None == unbounded
        self.max_operand_stack_depth = max_operand_stack_depth
        # Wall-clock deadline for execute(); None means no timeout. Set from
        # instruction_timeout_s at the start of the loop.
        self.instruction_timeout_s = instruction_timeout_s
        self._execute_deadline = None

        # §6.1 — Thread-safety state lock. RLock so re-entrant calls from
        # the same thread (e.g. op_call dispatching back into execute()
        # for a recursive frame, or a stdlib callback) don't deadlock.
        # Locks cover the whole execute() body so concurrent threads
        # sharing one VM instance serialize. For true parallelism
        # callers should use `execute_parallel`, which spawns fresh
        # VMs (one per shot) with zero shared mutable state.
        self._state_lock = threading.RLock()

        # Capture ctor args so `execute_parallel` can rebuild peer VMs
        # with byte-identical configuration. Updating this dict if the
        # caller mutates the VM after construction (e.g. swaps
        # `self.noise_model`) is the caller's responsibility — we
        # snapshot at __init__ time only.
        self._ctor_args = {
            "trace_mode": trace_mode,
            "sim_type": sim_type,
            "gpu_platform": gpu_platform,
            "seed": seed,
            "verbose": verbose,
            "opt_level": opt_level,
            "deterministic": deterministic,
            "max_instruction_count": max_instruction_count,
            "max_operand_stack_depth": max_operand_stack_depth,
            "instruction_timeout_s": instruction_timeout_s,
            "dispatch_mode": dispatch_mode,
        }

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
        
        # §1.1 VM Acceleration — integrated optimization components
        self._inline_cache = InlineCache()
        self._hot_loop_detector = HotLoopDetector(threshold=100)
        self._object_pool = ObjectPool()
        self._frame_cache = FrameCache()
        # §10.1 — Debug adapter protocol session
        self._debug_session = DebugSession()
        self._debug_mode = False
        
        # Heap
        self.heap = weakref.WeakValueDictionary()
        self.heap_lock = threading.Lock()
        self.next_ref_id = 1

        # Dispatch table for bytecode instructions
        # sol.md §1.1 — every hot-path opcode now has a dedicated
        # dispatch-table handler so the interpreter can run in pure
        # table-driven mode (`dispatch_mode='table'`). The if/elif
        # fast-path remains the default (`dispatch_mode='fast'`); both
        # paths use the SAME handlers, so behaviour is identical.
        self.dispatch_table = {
            Opcode.HALT: self.op_halt,
            Opcode.LOAD_CONST: self.op_load_const,
            Opcode.LOAD_VAR: self.op_load_var,
            Opcode.STORE_VAR: self.op_store_var,
            Opcode.ADD: self.op_add,
            Opcode.SUB: self.op_sub,
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

        # Native-Python recursion cache (see src/jit/recursive_codegen.py).
        # When ``op_call`` sees a name it recognizes here, the function is
        # invoked directly via a Python callable instead of going through
        # frame-push / bytecode dispatch / frame-pop. For pure numeric
        # recursion (e.g. fib(20)) this brings per-call cost down from
        # ~3.7us (VM dispatch) to ~0.14us (native Python), matching CPython.
        self.recursive_funcs: dict = {}
        self.recursive_calls_native: int = 0

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
        
        # §1.1 — Inline cache fast path: check cached source first
        entry = self._inline_cache._cache.get(name)
        if entry is not None:
            if entry.source == "frame" and self.call_stack:
                fl = self.call_stack[-1].locals
                if name in fl:
                    return fl[name]
            elif entry.source == "globals":
                if name in self.globals:
                    return self.globals[name]
            elif entry.source == "literal":
                return name
            # Cache miss — fall through to full lookup

        frame_locals = None
        if self.call_stack:
            frame = self.call_stack[-1]
            frame_locals = frame.locals
            if name in frame_locals:
                if len(self._inline_cache._cache) <= self._inline_cache._max_size:
                    self._inline_cache._cache[name] = CacheEntry(source="frame")
                return frame_locals[name]
        if name in self.globals:
            if len(self._inline_cache._cache) <= self._inline_cache._max_size:
                self._inline_cache._cache[name] = CacheEntry(source="globals")
            return self.globals[name]

        is_lit = _PATTERN_CACHE.get(name)
        if is_lit is None:
            is_lit = bool(VAR_PATTERN.match(name))
            if len(_PATTERN_CACHE) < _PATTERN_CACHE_MAX:
                _PATTERN_CACHE[name] = is_lit
        if is_lit:
            if len(self._inline_cache._cache) <= self._inline_cache._max_size:
                self._inline_cache._cache[name] = CacheEntry(source="literal")
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

    # === §10.1 — Debug Adapter Protocol Integration ====================
    def enable_debug(self):
        """Enable debug mode — VM will check breakpoints on each instruction."""
        self._debug_mode = True

    def disable_debug(self):
        self._debug_mode = False

    @property
    def debug_session(self) -> DebugSession:
        return self._debug_session

    def set_breakpoint(self, source: str, line: int):
        return self._debug_session.set_breakpoint(source, line)

    def _check_debug_pause(self, ip: int, line: int | None):
        """Check if execution should pause for debugging.
        Only called when _debug_mode is True."""
        if not self._debug_mode:
            return
        source = ""  # Could be enhanced to track source file
        call_depth = len(self.call_stack)
        if self._debug_session.should_pause(source, line or 0, call_depth):
            # Update debug frame with current state
            frame = self.call_stack[-1] if self.call_stack else None
            self._debug_session.update_frame(
                func_name=frame.func_name if frame else "main",
                line=line or 0,
                ip=ip,
                locals_dict=frame.locals if frame else self.globals,
                operand_stack=list(self.operand_stack),
            )
            # In a real DAP server, we would block here and wait for
            # the client to send a "continue" or "step" request.
            # For now, we just record the pause.
            self._debug_session.paused = True

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

    # === sol.md §1.1 VM Table-Driven Dispatch ============================
    # High-frequency opcodes promoted into `dispatch_table` so the
    # entire interpreter loop can run through `dispatch[op](arg)` with
    # no `if/elif` chain. The fast-path inline check (still the
    # default execution mode for max throughput) is now matched by
    # dedicated handlers — both paths produce identical results.
    def op_load_const(self, arg):
        """PUSH a constant onto the operand stack.
        Handler for `Opcode.LOAD_CONST`."""
        self.operand_stack.append(arg)

    def op_load_var(self, arg):
        """PUSH the value of variable `arg` (a str name).
        Handler for `Opcode.LOAD_VAR`. Resolves via `lookup_var`
        (which honours lexical scope + global fallback)."""
        self.operand_stack.append(self.lookup_var(arg))

    def op_store_var(self, arg):
        """POP one value and bind it to `arg` (a str variable name).
        Handler for `Opcode.STORE_VAR`. Writes to the current frame's
        local scope when a frame is active, otherwise to globals."""
        val = self.operand_stack.pop()
        if self.call_stack:
            self.call_stack[-1].locals[arg] = val
        else:
            self.globals[arg] = val

    def op_add(self, arg):
        """POP b, POP a, PUSH a + b. Handler for `Opcode.ADD`."""
        stack = self.operand_stack
        b = stack.pop()
        a = stack.pop()
        stack.append(a + b)

    def op_sub(self, arg):
        """POP b, POP a, PUSH a - b. Handler for `Opcode.SUB`."""
        stack = self.operand_stack
        b = stack.pop()
        a = stack.pop()
        stack.append(a - b)

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

        # Check standard library redirection first; stdlib needs args as a
        # list because ``execute_native_stdlib`` makes native Python calls.
        target_name = func_name
        if func_name in self._STD_MAPPING:
            target_name = self._STD_MAPPING[func_name]
            func_target = target_name

        is_stdlib_string = isinstance(func_target, str) and func_target.startswith("std.")
        is_missing_stdlib = (func_target is None and target_name and target_name.startswith("std."))
        if is_stdlib_string or is_missing_stdlib:
            # Stdlib path: pop args into a list, reverse, dispatch.
            args = []
            pop = self.operand_stack.pop
            for _ in range(num_args):
                args.append(pop())
            args.reverse()
            name = func_target if isinstance(func_target, str) else target_name
            res = self.execute_native_stdlib(name, args)
            self.operand_stack.append(res)
            return

        # Native-Python recursion fast path. The recursive_funcs registry is
        # populated by the run/CLI layer via ``compile_recursive_functions``.
        # We pop the args (rightmost-first on the stack) and invoke the Python
        # callable directly, bypassing ActivationFrame + bytecode dispatch.
        # This is safe because qualifying funcs are guaranteed pure numeric
        # (no quantum/heap/IO/struct/array ops) by the purity gate in the
        # recursive codegen module.
        if func_name in self.recursive_funcs and num_args >= 0:
            pop = self.operand_stack.pop
            args = [pop() for _ in range(num_args)]
            args.reverse()
            try:
                result = self.recursive_funcs[func_name](*args)
            except RecursionError:
                # Fall back through VM dispatch so the standard
                # StackOverflowError path (depth>=1000) fires once more
                # rather than raising a raw Python RecursionError.
                self.recursive_funcs.pop(func_name, None)
            else:
                self.operand_stack.append(result)
                self.recursive_calls_native += 1
                return

        # User-defined function path (audit §1.1 BUG #2/#8): the caller pushed
        # args in left-to-right order; the function body's STORE_VAR sequence
        # pops them in *reverse* order (last param first). The args are
        # therefore *already* on the operand stack in the layout the function
        # body expects — no pop-and-repush is needed. Removing this dance
        # eliminates O(calls) Python-list allocations per program; for fib(20)
        # that's ~21,000 list creations + reverses bypassed.
        if len(self.call_stack) >= 1000:
            self.throw_exception("StackOverflowError: Maximum recursion depth (1000) exceeded.")
            return

        new_frame = self.get_frame(self.ip, func_name)
        self.call_stack.append(new_frame)
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
        # §1.1 — Use ObjectPool to reduce list allocation churn
        elems = self._object_pool.borrow()
        pop = self.operand_stack.pop
        for _ in range(num_elems):
            elems.append(pop())
        elems.reverse()
        ref = self.allocate_heap("array", list(elems))  # copy for heap
        self._object_pool.release(elems)  # return borrowed list to pool
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

    def _try_fast_array_loop(self, instructions) -> bool:
        """
        Audit §1.1 BUG #1: the simple counter Fast-Loop JIT at ``_try_fast_loop``
        rejects ``for x in arr { body }`` patterns because (a) the loop condition
        uses ``LEN`` (4 ops, not the required 3) and (b) the body uses
        ``GET_INDEX`` which is not in the ``supported_body`` set. For the
        ``arrays.eig`` benchmark (sum of 10 numbers) this caused the loop to fall
        back to a fully interpret-and-deopt path: ~17 ms vs the quantum sim's
        ~0.8 ms for the same data size.

        This handler recognises the EBC-compiler lowering of ``for x in arr``
        specifically and rewrites it to a native Python
        ``for x_val in <underlying_python_list>`` loop — iteration over a list
        is one order of magnitude faster than the interpreter dispatch path.

        Matching pattern (after the function-skip JMP at IP 0):

            LOAD_VAR arr_ref_temp
            STORE_VAR _compiler_temp_1
            LOAD_CONST_STORE (0, _compiler_temp_2)
          <loop_start>:
            LOAD_VAR   _compiler_temp_2   # idx (counter)
            LOAD_VAR   _compiler_temp_1   # arr_ref
            LEN
            <CMP>
            JMP_IF_FALSE <exit_target>
            LOAD_VAR   _compiler_temp_1   # reloaded for GET_INDEX
            LOAD_VAR   _compiler_temp_2
            GET_INDEX
            STORE_VAR  x                  # iter var assignment
            <body using LOAD_VAR/STORE_VAR/ADD/SUB/.../MUL/... only>
            LOAD_VAR_LOAD_CONST_ADD (_compiler_temp_2, 1)   # idx + 1
            STORE_VAR  _compiler_temp_2
            JMP        <loop_start>       # backward jump
          <exit_target>:
            <post-loop opcodes — typically LOAD_VAR/PRINT/assert/HALT>

        Returns True if it ran the loop to completion and ran the epilogue.
        Returns False otherwise — caller falls back to the counter-based
        ``_try_fast_loop`` or the slow interpreter.
        """
        n = len(instructions)
        if n < 14:
            return False

        # 1. Find the *first* backward JMP (the loop back-jump).
        back_jump_idx = None
        back_jump_target = None
        for i, inst in enumerate(instructions):
            if inst.opcode == "JMP" and isinstance(inst.arg, int) and inst.arg < i:
                back_jump_idx = i
                back_jump_target = inst.arg
                break
        if back_jump_idx is None:
            return False

        # 2. Within the loop body, find the JMP_IF_FALSE that exits the loop.
        cond_check_idx = None
        exit_target = None
        for i in range(back_jump_target, back_jump_idx):
            inst = instructions[i]
            if inst.opcode == "JMP_IF_FALSE" and isinstance(inst.arg, int):
                cond_check_idx = i
                exit_target = inst.arg
                break
        if cond_check_idx is None:
            return False
        # exit_target must land *after* the back-jump (forward jump out of loop).
        if exit_target <= back_jump_idx:
            return False

        # 3. The condition expression: LOAD_VAR idx, LOAD_VAR arr_ref, LEN, CMP.
        cond_insts = instructions[back_jump_target:cond_check_idx]
        if len(cond_insts) != 4:
            return False
        if cond_insts[0].opcode != "LOAD_VAR" or not isinstance(cond_insts[0].arg, str):
            return False
        if cond_insts[1].opcode != "LOAD_VAR" or not isinstance(cond_insts[1].arg, str):
            return False
        if cond_insts[2].opcode != "LEN":
            return False
        if cond_insts[3].opcode not in ("LT", "LTE", "GT", "GTE", "EQ", "NEQ"):
            return False

        idx_var = cond_insts[0].arg
        arr_ref_var = cond_insts[1].arg
        cmp_op = cond_insts[3].opcode

        # `for x in arr` semantics require iterating over every element of the
        # array; loops with LTE/GT/GTE/EQ/NEQ against len(arr) *might* still
        # iterate (e.g. LTE with == len) but the pattern is rare and ambiguous;
        # accept only LT/LTE with the standard `idx < len(arr)` form.
        if cmp_op not in ("LT", "LTE"):
            return False

        body_start = cond_check_idx + 1
        body_end = back_jump_idx  # back-JMP itself is excluded from the body

        # Need at least: GET_INDEX(4) + increment + STORE_VAR idx (2 minimum).
        if body_end - body_start < 7:
            return False

        # 4. First 4 body ops must be: LOAD_VAR arr_ref, LOAD_VAR idx, GET_INDEX, STORE_VAR x.
        b0 = instructions[body_start]
        b1 = instructions[body_start + 1]
        b2 = instructions[body_start + 2]
        b3 = instructions[body_start + 3]
        if b0.opcode != "LOAD_VAR" or b0.arg != arr_ref_var:
            return False
        if b1.opcode != "LOAD_VAR" or b1.arg != idx_var:
            return False
        if b2.opcode != "GET_INDEX":
            return False
        if b3.opcode != "STORE_VAR" or not isinstance(b3.arg, str):
            return False
        iter_var = b3.arg

        # 5. Last body op = STORE_VAR idx_var (the increment's store-back).
        #    Second-to-last = the increment itself. We accept the
        #    LOAD_VAR_LOAD_CONST_ADD superinstr (the common emitter case) and
        #    fall back to the three-op LOAD_VAR/LOAD_CONST/ADD form.
        increment_const = None
        if (instructions[body_end - 2].opcode == "LOAD_VAR_LOAD_CONST_ADD"
                and isinstance(instructions[body_end - 2].arg, tuple)
                and instructions[body_end - 2].arg[0] == idx_var):
            store_inst = instructions[body_end - 1]
            if store_inst.opcode != "STORE_VAR" or store_inst.arg != idx_var:
                return False
            increment_const = instructions[body_end - 2].arg[1]
            body_insts_start = body_start + 4
            body_insts_end = body_end - 2
        elif body_end - body_start >= 9:
            l1 = instructions[body_end - 4]
            l2 = instructions[body_end - 3]
            addn = instructions[body_end - 2]
            store_2 = instructions[body_end - 1]
            if (l1.opcode != "LOAD_VAR" or l1.arg != idx_var
                    or l2.opcode != "LOAD_CONST"
                    or addn.opcode != "ADD"
                    or store_2.opcode != "STORE_VAR" or store_2.arg != idx_var):
                return False
            increment_const = l2.arg
            body_insts_start = body_start + 4
            body_insts_end = body_end - 4
        else:
            return False
        if increment_const != 1:
            return False

        body_insts = instructions[body_insts_start:body_insts_end]

        # 6. Validate that the body uses only ops we can lower to Python
        #    register-form. NOTABLY excluded: GET_INDEX (other than the head
        #    read we handle), SET_INDEX (body would mutate arr — Python's
        #    iteration would skip the new entry), LEN, CALL (recursion), RET,
        #    THROW, GET_FIELD, SET_FIELD, ALLOC_*, etc.
        supported_body = {
            "LOAD_VAR", "LOAD_CONST", "ADD", "SUB", "MUL", "DIV", "STORE_VAR",
            "LOAD_VAR_LOAD_CONST_ADD", "LOAD_VAR_LOAD_CONST_SUB", "LOAD_CONST_STORE",
            "JMP", "MOD", "POW", "BIT_AND", "BIT_OR", "BIT_XOR", "SHL", "SHR",
            "EQ", "NEQ", "LT", "GT", "LTE", "GTE",
        }
        for inst in body_insts:
            if inst.opcode not in supported_body:
                return False

        # 7. Collect all variables touched by the loop (so we can load them
        #    once into Python local ``r_*`` registers and write them back to
        #    globals/locals once at the end — avoiding dict lookups per iter).
        all_vars = {iter_var, idx_var, arr_ref_var}
        for inst in body_insts:
            if inst.opcode in ("LOAD_VAR", "STORE_VAR"):
                all_vars.add(inst.arg)
            elif inst.opcode == "LOAD_CONST_STORE":
                all_vars.add(inst.arg[1])
            elif inst.opcode in ("LOAD_VAR_LOAD_CONST_ADD", "LOAD_VAR_LOAD_CONST_SUB"):
                all_vars.add(inst.arg[0])

        clean = lambda v: "".join(c if c.isalnum() else "_" for c in v)

        # 8. Build the Python source.
        lines = []
        lines.append("def _fast_array_loop(_globals, _locals, _heap_resolve):")
        lines.append(f"    _arr_ref = _locals.get({arr_ref_var!r}, _globals.get({arr_ref_var!r}, None))")
        lines.append("    _arr_data = _heap_resolve(_arr_ref)")
        lines.append("    if _arr_data is None:")
        lines.append("        return False")
        for v in sorted(all_vars):
            cn = clean(v)
            lines.append(f"    r_{cn} = _locals.get({v!r}, _globals.get({v!r}, 0))")
        iter_cn = clean(iter_var)
        lines.append(f"    for _iter_val in _arr_data:")
        lines.append(f"        r_{iter_cn} = _iter_val")

        stack = []
        _binop = {
            "ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "MOD": "%", "POW": "**",
            "EQ": "==", "NEQ": "!=", "LT": "<", "GT": ">", "LTE": "<=", "GTE": ">=",
            "BIT_AND": "&", "BIT_OR": "|", "BIT_XOR": "^", "SHL": "<<", "SHR": ">>",
        }
        for inst in body_insts:
            if inst.opcode == "LOAD_CONST":
                stack.append(repr(inst.arg))
            elif inst.opcode == "LOAD_VAR":
                stack.append(f"r_{clean(inst.arg)}")
            elif inst.opcode == "STORE_VAR":
                if not stack:
                    return False
                val = stack.pop()
                lines.append(f"        r_{clean(inst.arg)} = {val}")
            elif inst.opcode in _binop:
                if len(stack) < 2:
                    return False
                b = stack.pop(); a = stack.pop()
                stack.append(f"({a} {_binop[inst.opcode]} {b})")
            elif inst.opcode == "LOAD_VAR_LOAD_CONST_ADD":
                vn, k = inst.arg
                stack.append(f"(r_{clean(vn)} + {k!r})")
            elif inst.opcode == "LOAD_VAR_LOAD_CONST_SUB":
                vn, k = inst.arg
                stack.append(f"(r_{clean(vn)} - {k!r})")
            elif inst.opcode == "LOAD_CONST_STORE":
                cv, vn = inst.arg
                lines.append(f"        r_{clean(vn)} = {cv!r}")
            elif inst.opcode == "JMP":
                # The compiler emits a JMP only for `continue`'s forward jump
                # inside the loop. We treat it as a no-op inside the Python
                # for-loop body; this is correct *iff* the only body-JMPs are
                # forward jumps within the loop. We do not currently detect
                # break/continue semantics — they're outside `supported_body`
                # (BREAK/CONTINUE are separate opcodes).
                pass
            else:
                return False

        if stack:
            # Body leaves intermediate values on the stack — bail to the
            # interpreter for safety.
            return False

        # Write back *all* locals at the end. The idx_var is set to the final
        # length so subsequent code that reads it sees the same value the
        # original interpreter would have left.
        for v in sorted(all_vars):
            if v == idx_var:
                continue
            cn = clean(v)
            lines.append(f"    _locals[{v!r}] = r_{cn}")
            lines.append(f"    _globals[{v!r}] = r_{cn}")
        idx_cn = clean(idx_var)
        lines.append(f"    _locals[{idx_var!r}] = len(_arr_data)")
        lines.append(f"    _globals[{idx_var!r}] = len(_arr_data)")
        lines.append("    return True")

        source = "\n".join(lines)

        try:
            code_obj = compile(source, "<fast_array_loop>", "exec")
            # Audit §3: hardened sandbox — no type/getattr/hasattr/isinstance.
            safe_globals = {
                "__builtins__": {
                    "bool": bool, "int": int, "float": float, "str": str,
                    "len": len, "abs": abs, "range": range, "repr": repr,
                },
            }
            local_vars = {}
            exec(code_obj, safe_globals, local_vars)
            fast_func = local_vars["_fast_array_loop"]
        except Exception:
            return False

        # 9. Execute the prologue (instructions before the loop start). Any
        #    opcode outside the supported set bails — keep us robust against
        #    future compiler changes.
        prologue_ok = self._exec_prologue_for_fast_array_loop(back_jump_target, instructions)
        if not prologue_ok:
            return False

        # 10. Heap-resolver: walks a VMRef to its underlying Python list.
        def resolve_arr_data(ref):
            if not isinstance(ref, VMRef):
                return None
            if ref.ref_id not in self.heap:
                return None
            obj = self.heap[ref.ref_id]
            if obj.obj_type != "array":
                return None
            return obj.data

        locals_map = self.call_stack[-1].locals if self.call_stack else self.globals
        success = fast_func(self.globals, locals_map, resolve_arr_data)
        if not success:
            return False

        # Audit BUG #6: increment the unused-but-tracked jit_hits counter so
        # the resource profile / monitoring reports a non-zero JIT count.
        self.jit_hits += 1

        # 11. Run the epilogue (instructions from exit_target to end). Bail
        #     to the interpreter if any opcode falls outside our supported
        #     set; this ensures we never silently drop a side-effect.
        return self._exec_epilogue_for_fast_array_loop(exit_target, instructions)

    def _exec_prologue_for_fast_array_loop(self, up_to_idx: int, instructions) -> bool:
        """Run the prologue instructions for the array-loop JIT. Returns True
        when all prologue ops were handled; False if the prologue contains
        something we don't yet handle (caller bails to the slow interpreter)."""
        for i in range(0, up_to_idx):
            inst = instructions[i]
            try:
                if inst.opcode == "LOAD_CONST":
                    self.operand_stack.append(inst.arg)
                elif inst.opcode == "STORE_VAR":
                    val = self.operand_stack.pop()
                    if self.call_stack:
                        self.call_stack[-1].locals[inst.arg] = val
                    else:
                        self.globals[inst.arg] = val
                elif inst.opcode == "ALLOC_ARRAY":
                    num_elems = inst.arg
                    elems = []
                    for _ in range(num_elems):
                        elems.append(self.operand_stack.pop())
                    elems.reverse()
                    ref = self.allocate_heap("array", elems)
                    self.operand_stack.append(ref)
                elif inst.opcode == "LOAD_CONST_STORE":
                    cv, vn = inst.arg
                    if self.call_stack:
                        self.call_stack[-1].locals[vn] = cv
                    else:
                        self.globals[vn] = cv
                elif inst.opcode == "LOAD_VAR":
                    self.operand_stack.append(self.lookup_var(inst.arg))
                elif inst.opcode == "LOAD_VAR_LOAD_CONST_ADD":
                    vn, k = inst.arg
                    self.operand_stack.append(self.lookup_var(vn) + k)
                elif inst.opcode == "LOAD_VAR_LOAD_CONST_SUB":
                    vn, k = inst.arg
                    self.operand_stack.append(self.lookup_var(vn) - k)
                elif inst.opcode == "ADD":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a + b)
                elif inst.opcode == "SUB":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a - b)
                elif inst.opcode == "MUL":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a * b)
                elif inst.opcode == "DIV":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    if b == 0:
                        self.throw_exception("DivisionByZeroError: Division by zero.")
                        return False
                    self.operand_stack.append(a / b)
                elif inst.opcode == "JMP":
                    pass  # the function-skip jump at IP 0
                else:
                    return False
            except IndexError:
                return False
        return True

    def _exec_epilogue_for_fast_array_loop(self, exit_target: int, instructions) -> bool:
        """Run the instructions after the loop body. Return True iff we
        reached HALT (or natural end) with only supported epilogue ops."""
        n = len(instructions)
        self.instructions = instructions
        ip = exit_target
        while ip < n:
            inst = instructions[ip]
            ip += 1
            self.ip = ip
            try:
                if inst.opcode == "LOAD_VAR":
                    self.operand_stack.append(self.lookup_var(inst.arg))
                elif inst.opcode == "LOAD_CONST":
                    self.operand_stack.append(inst.arg)
                elif inst.opcode == "PRINT":
                    self.op_print(inst.arg)
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
                elif inst.opcode == "LOAD_VAR_LOAD_CONST_ADD":
                    vn, k = inst.arg
                    self.operand_stack.append(self.lookup_var(vn) + k)
                elif inst.opcode == "LOAD_VAR_LOAD_CONST_SUB":
                    vn, k = inst.arg
                    self.operand_stack.append(self.lookup_var(vn) - k)
                elif inst.opcode == "LOAD_VAR_LOAD_CONST_LT":
                    vn, k = inst.arg
                    self.operand_stack.append(self.lookup_var(vn) < k)
                elif inst.opcode == "LOAD_VAR_LOAD_CONST_LTE":
                    vn, k = inst.arg
                    self.operand_stack.append(self.lookup_var(vn) <= k)
                elif inst.opcode == "LOAD_VAR_LOAD_CONST_GT":
                    vn, k = inst.arg
                    self.operand_stack.append(self.lookup_var(vn) > k)
                elif inst.opcode == "LOAD_VAR_LOAD_CONST_GTE":
                    vn, k = inst.arg
                    self.operand_stack.append(self.lookup_var(vn) >= k)
                elif inst.opcode == "ADD":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a + b)
                elif inst.opcode == "SUB":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a - b)
                elif inst.opcode == "MUL":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a * b)
                elif inst.opcode == "EQ":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a == b)
                elif inst.opcode == "NEQ":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a != b)
                elif inst.opcode == "LT":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a < b)
                elif inst.opcode == "GT":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a > b)
                elif inst.opcode == "LTE":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a <= b)
                elif inst.opcode == "GTE":
                    b = self.operand_stack.pop(); a = self.operand_stack.pop()
                    self.operand_stack.append(a >= b)
                elif inst.opcode == "JMP_IF_FALSE":
                    cond = self.operand_stack.pop()
                    if not cond:
                        ip = inst.arg
                    self.ip = ip
                elif inst.opcode == "JMP_IF_TRUE":
                    cond = self.operand_stack.pop()
                    if cond:
                        ip = inst.arg
                    self.ip = ip
                elif inst.opcode == "JMP":
                    ip = inst.arg
                    self.ip = ip
                elif inst.opcode == "HALT":
                    return True
                else:
                    self.ip = ip - 1
                    return False
            except IndexError:
                self.ip = ip - 1
                return False
        return True

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
            # Audit §3: hardened sandbox. The fast-loop generated source uses
            # only dict.get + arithmetic + comparisons, so we drop the
            # introspection primitives (type/getattr/hasattr/isinstance) that
            # previously enabled escapes like type(x).__subclasses__().
            safe_globals = {
                "__builtins__": {
                    "bool": bool,
                    "int": int,
                    "float": float,
                    "str": str,
                    "len": len,
                    "abs": abs,
                    "range": range,
                    "repr": repr,
                },
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

    def execute(self, instructions: list[Instruction],
                audit: 'AuditTrail | None' = None,
                program_hash: str | None = None):
        # §6.1 — serialize concurrent calls on the same VM instance.
        # RLock allows re-entry from the same thread (e.g. native stdlib
        # callbacks that recurse into execute); cross-thread calls
        # will block here until the holder finishes.
        #
        # §6.4 — when `audit` is non-None, we wrap the inner call in an
        # `AuditTrail.record(...)` so the trail captures (program_hash,
        # parameters, outcome, duration). We compute the duration via
        # `time.monotonic_ns()` so it's stable across wall-clock NTP
        # adjustments. The audit is opt-in: callers that don't pass
        # `audit=` continue to use the existing, non-audited code path.
        if audit is None:
            with self._state_lock:
                return self._execute_locked(instructions)
        # Audited path: build the entry around _execute_locked.
        started = _time.monotonic_ns()
        outcome_str = "success"
        err: BaseException | None = None
        try:
            with self._state_lock:
                rv = self._execute_locked(instructions)
        except BaseException as e:
            err = e
            outcome_str = "failure"
            raise
        finally:
            wall = _time.monotonic_ns() - started
            # Don't fail the audited op because of an audit-write
            # failure; record() swallows OSError by design.
            audit.record(
                program_hash=program_hash or
                              _hash_program_fallback(self, instructions),
                seed=self._ctor_args.get("seed"),
                sim_type=self._ctor_args.get("sim_type"),
                gpu_platform=self._ctor_args.get("gpu_platform"),
                deterministic=self.deterministic,
                noise_type=(getattr(self.noise_model, "noise_type", None)
                            if self.noise_model is not None else None),
                noise_prob=(getattr(self.noise_model, "noise_prob", None)
                            if self.noise_model is not None else None),
                num_instructions=len(instructions),
                outcome=outcome_str,
                error=err,
                started_at_ns=started,
                wall_clock_ns=wall,
            )
        return rv

    def execute_parallel(self, instructions: list[Instruction],
                          shots: int = 8, threads: int | None = None):
        """Run `instructions` on `shots` fresh VM instances in parallel.

        Each shot gets a brand-new `EigenVM` constructed with the
        parent VM's `_ctor_args` plus a per-shot seed offset (`seed + i
        + 1`) when the parent had a fixed seed; otherwise the new VM
        is seeded stochastically. The parent VM's state is never
        touched. Returns a list of `dict(sub_vm.globals)` snapshots —
        one per shot — in shot order.

        Args:
            instructions: bytecode to execute on each shot VM.
            shots: number of independent VMs to spawn.
            threads: max worker threads. Defaults to min(shots, 8).
        """
        if shots <= 0:
            return []
        # Pull ctor args from the snapshot; we don't propagate the live
        # noise_model because that object carries an rng that's stateful
        # and per-instance — fresh VMs build their own from the seed.
        ctor_args = dict(self._ctor_args)
        # If parent has a noise config, we want the same noise model
        # parameters on each shot, but a fresh rng. The simplest way is
        # to clone the noise config by re-constructing one with the
        # same args; since NoiseModel exposes its constructor params
        # as attributes, we can copy them through.
        try:
            nm = self.noise_model
            if nm is not None and getattr(nm, "noise_type", None) is not None:
                from src.noise.noise_model import NoiseModel
                ctor_args["noise_model"] = NoiseModel(
                    noise_type=nm.noise_type,
                    noise_prob=nm.noise_prob,
                )
        except Exception:
            # Fall back to no noise model on the sub-VMs; better than
            # crashing in the parallel path.
            ctor_args.pop("noise_model", None)

        # Snapshot of the parent's recursive_funcs so JIT'd functions
        # are available in sub-VMs without recompiling.
        recursive_funcs = dict(getattr(self, "recursive_funcs", {}) or {})

        def _shot(i):
            args = dict(ctor_args)
            base_seed = args.get("seed")
            if base_seed is not None and not self.deterministic:
                args["seed"] = base_seed + i + 1
            sub_vm = EigenVM(**args)
            if recursive_funcs:
                sub_vm.recursive_funcs = dict(recursive_funcs)
            sub_vm.execute(instructions)
            # Merge top-level locals into globals so callers see a
            # single, self-contained shot dictionary — this matters
            # because op_q_measure stores cbits in
            # `call_stack[-1].locals` (when a frame exists), not in
            # globals; for shot-based quantum experiments that IS the
            # outcome we care about.
            combined = dict(sub_vm.globals)
            if sub_vm.call_stack:
                for k, v in sub_vm.call_stack[-1].locals.items():
                    if k not in combined:
                        combined[k] = v
            return combined

        threads = min(threads or min(shots, 8), shots)
        threads = max(1, threads)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [pool.submit(_shot, i) for i in range(shots)]
            return [f.result() for f in futures]

    def _execute_locked(self, instructions: list[Instruction]):
        # sol.md §1.1 — pure table-driven dispatch when requested. The
        # `auto` backend configuration + audit + limits + JIT setup
        # below still apply through the shared preamble; we just
        # switch the inner interpreter loop.
        if self.dispatch_mode == 'table':
            self._execute_dispatch_table(instructions)
            return
        if getattr(self.simulator, 'sim_type', None) == 'auto':
            from src.backend.bytecode import Opcode
            from src.backend.gate_registry import CLIFFORD_GATES
            from src.backend.sim_selector import select_from_counts
            n_qubits = 0
            n_2q = 0
            n_gates = 0
            is_all_clifford = True
            for inst in instructions:
                if inst.opcode == Opcode.Q_ALLOC:
                    n_qubits += 1
                elif inst.opcode == Opcode.Q_GATE:
                    n_gates += 1
                    if isinstance(inst.arg, tuple) and len(inst.arg) > 0:
                        gate_name = inst.arg[0]
                        if gate_name in ('CNOT', 'CZ', 'SWAP'):
                            n_2q += 1
                        if is_all_clifford and gate_name not in CLIFFORD_GATES:
                            is_all_clifford = False
            noise_active = bool(self.noise_model and self.noise_model.noise_prob > 0)
            report = select_from_counts(
                n_qubits=n_qubits,
                n_2q_gates=n_2q,
                n_gates=n_gates,
                is_all_clifford=is_all_clifford,
                noise_active=noise_active,
            )
            chosen = report.chosen
            if self.verbose:
                print(f"[Auto Backend] Selected '{chosen}': {report.reason}")
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
            # Audit §1.1 BUG #1: try the array-iteration variant first (it is
            # a specific pattern the counter JIT below cannot handle because of
            # LEN/GET_INDEX). If it recognizes the pattern and runs, we're done.
            if self._try_fast_array_loop(instructions):
                self.log_trace("Finished execution of Eigen VM bytecode (fast array loop)")
                return
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

        # === sol.md P0 VM hardening (§7.1) ====================================
        # Pre-bake all the limit state into local fast-path variables so the
        # checks themselves stay O(1) and never enter the hot path through
        # attribute lookups.
        self.instruction_count = 0
        max_inst = self.max_instruction_count
        max_stack = self.max_operand_stack_depth
        deterministic = self.deterministic
        if self.instruction_timeout_s is not None:
            import time as _time_mod
            _deadline = _time_mod.monotonic() + self.instruction_timeout_s
            _time_now = _time_mod.monotonic
        else:
            _deadline = None
            _time_now = None

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
        _debug_mode = self._debug_mode  # §10.1 — cache debug flag
        while ip < n_instrs:
            op = _ops[ip]
            arg = _args[ip]
            ip += 1
            self.ip = ip

            # §10.1 — Debug breakpoint check (only when debug mode is on)
            if _debug_mode:
                self._check_debug_pause(ip, _lines[ip - 1])

            # sol.md P0 VM hardening §7.1 — limits checks.
            # Use bitmask-aware cheap checks. The instruction-count check uses
            # a modulo so it only does the comparison every 4096 ops.
            cnt = self.instruction_count + 1
            if max_inst is not None and cnt >= max_inst:
                _throw("TimeoutError: maximum instruction count reached; "
                      "use --max-instructions to extend the limit.")
                break
            if (cnt & 0xfff) == 0:
                if len(self.operand_stack) > max_stack:
                    _throw("StackOverflowError: operand stack exceeded configured "
                          "maximum depth (max_operand_stack_depth).")
                    break
                if _deadline is not None and _time_now() >= _deadline:
                    _throw("TimeoutError: instruction execution time limit exceeded.")
                    break
            self.instruction_count = cnt

            try:
                if op == OP_LOAD_CONST:
                    append(arg)
                elif op == OP_LOAD_VAR:
                    append(_lookup(arg))
                elif op == OP_STORE_VAR:
                    val = pop()
                    # §1.1 — Use FrameCache to avoid call_stack[-1].locals lookup
                    fl = self._frame_cache.get(call_stack)
                    if fl is not None:
                        fl[arg] = val
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
                    if not (0 <= arg < n_instrs):
                        _throw(f"InvalidJumpError: JMP target {arg} out of [0, {n_instrs}).")
                    elif arg < ip and jit_enabled:
                        # §1.1 — Hot loop detection for JIT triggering
                        self._hot_loop_detector.record_branch(arg, ip)
                        ip = arg; self.ip = ip
                        if run_jit(arg):
                            ip = self.ip
                            continue
                    ip = arg; self.ip = ip
                elif op == OP_JIF:
                    cond = pop()
                    if not cond:
                        if not (0 <= arg < n_instrs):
                            _throw(f"InvalidJumpError: JMP_IF_FALSE target {arg} out of [0, {n_instrs}).")
                        elif arg < ip and jit_enabled:
                            # §1.1 — Hot loop detection
                            self._hot_loop_detector.record_branch(arg, ip)
                            ip = arg; self.ip = ip
                            if run_jit(arg):
                                ip = self.ip
                                continue
                        ip = arg; self.ip = ip
                elif op == OP_CALL:
                    self._frame_cache.invalidate()  # §1.1 — frame changes on CALL
                    self.op_call(arg)
                    ip = self.ip
                elif op == OP_RET:
                    self._frame_cache.invalidate()  # §1.1 — frame changes on RET
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

    # === sol.md §1.1 VM Table-Driven Dispatch ==============================
    def _execute_dispatch_table(self, instructions: list[Instruction]):
        """Pure table-driven interpreter loop — sol.md §1.1 architectural
        form. Every opcode routes through `dispatch_table[op](arg)`; no
        `if/elif` chain. This is slower than the fast-path loop in
        `_execute_locked` (which hand-caches locals + branches), but
        it provides:
          * Architectural correctness parity — both modes dispatch
            through the SAME `op_*` handler methods.
          * A reference for future refactors — easy to reason about
            execution because every opcode has a single source-of
            truth entry point.
          * A sanity-check bench target: tests can switch modes and
            assert identical results, surfacing any subtle divergence.

        The limit checks (max instruction count, max stack depth,
        opcode timeout) and audit/auto-backend selection are preserved
        from the fast-path loop, so runtime guarantees hold in both
        modes.
        """
        # Backend auto-selection (same as fast-path).
        if getattr(self.simulator, 'sim_type', None) == 'auto':
            from src.backend.bytecode import Opcode
            from src.backend.gate_registry import CLIFFORD_GATES
            from src.backend.sim_selector import select_from_counts
            n_qubits = 0
            n_2q = 0
            n_gates = 0
            is_all_clifford = True
            for inst in instructions:
                if inst.opcode == Opcode.Q_ALLOC:
                    n_qubits += 1
                elif inst.opcode == Opcode.Q_GATE:
                    n_gates += 1
                    if isinstance(inst.arg, tuple) and len(inst.arg) > 0:
                        gate_name = inst.arg[0]
                        if gate_name in ('CNOT', 'CZ', 'SWAP'):
                            n_2q += 1
                        if is_all_clifford and gate_name not in CLIFFORD_GATES:
                            is_all_clifford = False
            noise_active = bool(self.noise_model and self.noise_model.noise_prob > 0)
            report = select_from_counts(
                n_qubits=n_qubits,
                n_2q_gates=n_2q,
                n_gates=n_gates,
                is_all_clifford=is_all_clifford,
                noise_active=noise_active,
            )
            chosen = report.chosen
            if self.verbose:
                print(f"[Auto Backend] Selected '{chosen}': {report.reason}")
            self.simulator.configure_backend(chosen)

        # Initialise execution state (same as fast-path).
        self.instructions = instructions
        self.ip = 0
        self.operand_stack = []
        self.call_stack = [self.get_frame(None, "main")]
        self.try_stack = []
        if not hasattr(self, 'globals') or not isinstance(self.globals, dict):
            self.globals = {}

        self.log_trace("Starting execution of Eigen VM bytecode (table dispatch)")

        n_instrs = len(instructions)
        dispatch = self.dispatch_list
        _throw = self.throw_exception

        # Limit-check state (mirror the fast-path preamble).
        self.instruction_count = 0
        max_inst = self.max_instruction_count
        max_stack = self.max_operand_stack_depth
        if self.instruction_timeout_s is not None:
            import time as _time_mod
            _deadline = _time_mod.monotonic() + self.instruction_timeout_s
            _time_now = _time_mod.monotonic
        else:
            _deadline = None
            _time_now = None

        try:
            # The hot interpreter loop is intentionally plain. Each op
            # runs through `dispatch_table[op](arg)`; the returned
            # truthiness tells the loop whether to stop (HALT /
            # hard error via throw_exception breaks) or continue.
            ip = 0
            while ip < n_instrs:
                inst = instructions[ip]
                op = inst.opcode_int
                arg = inst.arg
                ip += 1
                self.ip = ip

                # Limit checks.
                cnt = self.instruction_count + 1
                if max_inst is not None and cnt >= max_inst:
                    _throw("TimeoutError: maximum instruction count reached; "
                           "use --max-instructions to extend the limit.")
                    break
                if (cnt & 0xfff) == 0:
                    if len(self.operand_stack) > max_stack:
                        _throw("StackOverflowError: operand stack exceeded "
                               "configured maximum depth "
                               "(max_operand_stack_depth).")
                        break
                    if _deadline is not None and _time_now() >= _deadline:
                        _throw("TimeoutError: instruction execution time "
                               "limit exceeded.")
                        break
                self.instruction_count = cnt

                if op < 0 or op >= len(dispatch):
                    _throw(f"InvalidOpcodeError: Invalid opcode {op} at "
                           f"IP {ip - 1}. This may be a bytecode from a "
                           f"newer version — upgrade your Eigen runtime.")
                    break
                handler = dispatch[op]
                if handler is None:
                    _throw(f"InvalidOpcodeError: Unhandled opcode {op} at "
                           f"IP {ip - 1}. This opcode may be from a newer "
                           f"bytecode minor version — upgrade your Eigen "
                           f"runtime to execute it.")
                    break
                if handler(arg):
                    # HALT or hard-error path: handler returned truthy.
                    break
                # Maintain the same IP-rewind semantics as the
                # fast-path loop: a handler may set self.ip to a new
                # target (JMP / RET / CALL etc.). After the call, we
                # pick up the new IP from `self.ip`.
                ip = self.ip
        except IndexError:
            _throw("StackUnderflowError: Operand stack underflow.")

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
