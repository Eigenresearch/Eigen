import hashlib
import threading
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
    # §3.1 — streaming stable hash for bytecode segments.
    # Instead of json.dumps() which creates a full in-memory copy of the
    # serialized bytecode, we update the hasher iteratively.
    hasher = hashlib.sha256()
    for inst in instructions_segment:
        # Instruction(opcode, arg)
        op_data = str(inst.opcode).encode('utf-8')
        arg_data = repr(inst.arg).encode('utf-8')
        hasher.update(op_data)
        hasher.update(b':')
        hasher.update(arg_data)
        hasher.update(b'|')
    return hasher.hexdigest()


def _build_sandbox_globals() -> dict:
    """Build the restrictive globals dict for ``exec`` of JIT-compiled blocks.

    Audit §3: the previous sandbox used ``{"__builtins__": {}} + {type, getattr,
    hasattr, isinstance, ...}``. The four introspection primitives
    ``type``/``getattr``/``hasattr``/``isinstance`` are well-known sandbox-escape
    vectors — they let a generated block walk the MRO via
    ``type(obj).__subclasses__()`` and reach ``os`` / builtins. We drop them
    entirely. The generated fast-loop/JIT code (see ``native_codegen.py``) uses
    only direct attribute access (``vm.op_x(arg)``, ``x.__class__.__name__``),
    dict/list/arithmetic ops, and ``len``/``bool``/``int``/``float``/``str`` for
    literal coercion. None of these need the four escape primitives.

    CAVEAT: this is a defense in depth, not OS-level isolation. The audit
    explicitly notes that the only *complete* defence against untrusted code is
    a separate process / container / microVM, or to skip Python ``exec``
    entirely and keep JIT within the bytecode VM. Use this in conjunction with
    subprocess isolation when running untrusted ``.eig`` files (see
    ``tests/test_aot.py`` for the same pattern).
    """
    return {
        "__builtins__": {
            # Coercion / numeric base types only.
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "len": len,
            "abs": abs,
            "range": range,
            "repr": repr,
        },
        # No `type`, `getattr`, `hasattr`, `isinstance` -- introspection escape
        # primitives removed (audit §3).
    }


class JITVMProxy:
    """A restricted proxy for the VM instance to be used within the JIT sandbox.
    
    Audit §3.1: instead of passing the full EigenVM instance to ``exec``'d code,
    we pass this proxy which only exposes the necessary opcode handlers and 
    state needed for JIT execution. This prevents the JIT'd block from 
    accessing sensitive VM internals or reaching into the host system.
    """
    def __init__(self, vm):
        self._vm = vm
        # Expose only safe attributes
        self.output_stream = vm.output_stream
        self.verbose = getattr(vm, 'verbose', False)
        self.instructions = vm.instructions # Needed for some checks
        
    @property
    def ip(self):
        return self._vm.ip
    
    @ip.setter
    def ip(self, value):
        self._vm.ip = value
        
    @property
    def call_stack(self):
        return self._vm.call_stack
        
    def throw_exception(self, msg):
        return self._vm.throw_exception(msg)
        
    def op_ret(self, arg):
        return self._vm.op_ret(arg)
        
    def op_call(self, arg):
        return self._vm.op_call(arg)
        
    def __getattr__(self, name):
        if name.startswith("op_"):
            return getattr(self._vm, name)
        raise AttributeError(f"JITVMProxy has no attribute {name!r}")


class JITCompiler:
    """Per-VM JIT compiler with hot-trace detection.

    Audit §1.1 BUG #5 / §2.3: the previous design stored ``GLOBAL_CACHE`` and
    ``GLOBAL_EXEC_COUNTS`` as class attributes, which (a) leaks state across
    distinct ``EigenVM`` instances and (b) means two programs with structurally
    identical bytecode share a compiled artifact — a latent cache-collision
    hazard analogous to BUG-C06. The cache and exec-counts are now per-instance,
    not shared across ``EigenVM`` instantiations.
    """

    DEFAULT_HOT_THRESHOLD = 3
    DEFAULT_CALL_HOT_THRESHOLD = 8  # recursion: higher threshold than loops
    DEFAULT_EXEC_COUNTS_MAX = 4096
    _inline_counter = 0  # name-suffix counter only, no security implication

    def __init__(self, vm, hot_threshold: int = None, call_hot_threshold: int = None, exec_counts_max: int = None):
        self.vm = vm
        self.hot_threshold = hot_threshold if hot_threshold is not None else JITCompiler.DEFAULT_HOT_THRESHOLD
        self.call_hot_threshold = (
            call_hot_threshold if call_hot_threshold is not None else JITCompiler.DEFAULT_CALL_HOT_THRESHOLD
        )
        self.exec_counts_max = (
            exec_counts_max if exec_counts_max is not None else JITCompiler.DEFAULT_EXEC_COUNTS_MAX
        )
        # Per-instance cache (audit BUG #5 / §2.3).
        self.cache = LRUCache(maxsize=1024)
        self.exec_counts = {}
        self.current_instructions = None
        self.ip_to_key = {}
        # Hot-counter specifically for CALL sites — recursion trigger (BUG #3).
        # Keyed by the called function's entry IP.
        self.call_counts = {}
        self._lock = threading.Lock()

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

        # Check per-instance LRU cache first
        with self._lock:
            compiled_func = self.cache.get(key)
            if compiled_func:
                return compiled_func

            # Update execution counts (cap dict size; preserve hot entries)
            count = self.exec_counts.get(key, 0) + 1
            if len(self.exec_counts) < self.exec_counts_max or key in self.exec_counts:
                self.exec_counts[key] = count
        if count < self.hot_threshold:
            return None

        # Detect basic block starting at ip and try to compile it
        block = self.trace_basic_block(ip, instructions)
        if len(block) >= 2:  # Only compile blocks of reasonable size (superinstructions reduce count)
            compiled_func = self.compile_block(block)
            if compiled_func:
                with self._lock:
                    self.cache.put(key, compiled_func)
                return compiled_func
        return None

    def check_and_compile_call(self, func_target: int, instructions: list[Instruction]) -> callable:
        """Hot-counter trigger for recursive call sites (audit BUG #3).

        Returns the compiled entry block if the function `func_target` has been
        entered `call_hot_threshold` times, else None. The threshold is higher
        than the back-jump threshold (8 vs 3) because the only safe recursion
        JIT is the entry-block compile (we cannot currently inline bodies that
        contain CALL/JMP_IF_FALSE per ``inline_function``).
        """
        if (
            self.current_instructions is not instructions
            and instructions is not None
        ):
            self.analyze_program(instructions)

        with self._lock:
            count = self.call_counts.get(func_target, 0) + 1
            self.call_counts[func_target] = count
        if count < self.call_hot_threshold:
            return None
        compiled = self.check_and_compile(func_target, instructions)
        return compiled

    def trace_basic_block(self, start_ip: int, instructions: list[Instruction]) -> list[Instruction]:
        block = []
        ip = start_ip
        while ip < len(instructions):
            inst = instructions[ip]
            block.append(inst)
            # Basic block ends on JMPs, RETs, HALT, or if the next instruction is a jump target
            # For simplicity, end block on control flow instructions
            if inst.opcode in (
                Opcode.JMP, Opcode.JMP_IF_FALSE, Opcode.JMP_IF_TRUE,
                Opcode.RET, Opcode.HALT, Opcode.CALL
            ):
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
            # Audit §3: hardened sandbox — no type/getattr/hasattr/isinstance.
            safe_globals = _build_sandbox_globals()
            exec(code_obj, safe_globals, local_vars)
            return local_vars['compiled_block']
        except Exception:
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
