from src.bytecode import Opcode, Instruction
from src.simulator import QuantumSimulator
import random

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
        self.locals = {}           # var_name -> value
        self.return_address = return_address
        self.try_stack = []        # list of (handler_ip, saved_stack_depth)
        self.current_line = None
        self.func_name = func_name


class EigenVM:
    def __init__(self, trace_mode: bool = False):
        self.simulator = QuantumSimulator()
        self.trace_mode = trace_mode
        self.trace_log = []
        
        # VM registers and stacks
        self.instructions = []
        self.ip = 0
        self.operand_stack = []
        self.call_stack = []
        self.globals = {}
        
        # Heap
        self.heap = {}
        self.next_ref_id = 1

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

    def execute(self, instructions: list[Instruction]):
        self.instructions = instructions
        self.ip = 0
        self.operand_stack = []
        self.call_stack = [ActivationFrame(return_address=None, func_name="main")]
        self.globals = {}
        self.heap = {}
        self.next_ref_id = 1

        self.log_trace("Starting execution of Eigen VM bytecode")

        while self.ip < len(self.instructions):
            instr = self.instructions[self.ip]
            self.ip += 1

            if self.call_stack and instr.line is not None:
                self.call_stack[-1].current_line = instr.line

            opcode = instr.opcode
            arg = instr.arg

            if opcode == Opcode.HALT:
                break

            elif opcode == Opcode.LOAD_CONST:
                self.operand_stack.append(arg)

            elif opcode == Opcode.LOAD_VAR:
                self.operand_stack.append(self.lookup_var(arg))

            elif opcode == Opcode.STORE_VAR:
                val = self.operand_stack.pop()
                if self.call_stack:
                    self.call_stack[-1].locals[arg] = val
                else:
                    self.globals[arg] = val

            elif opcode == Opcode.ADD:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a + b)

            elif opcode == Opcode.SUB:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a - b)

            elif opcode == Opcode.MUL:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a * b)

            elif opcode == Opcode.DIV:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a / b)

            elif opcode == Opcode.EQ:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a == b)

            elif opcode == Opcode.NEQ:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a != b)

            elif opcode == Opcode.LT:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a < b)

            elif opcode == Opcode.GT:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a > b)

            elif opcode == Opcode.LTE:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a <= b)

            elif opcode == Opcode.GTE:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(a >= b)

            elif opcode == Opcode.AND:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(bool(a) and bool(b))

            elif opcode == Opcode.OR:
                b = self.operand_stack.pop()
                a = self.operand_stack.pop()
                self.operand_stack.append(bool(a) or bool(b))

            elif opcode == Opcode.NOT:
                a = self.operand_stack.pop()
                self.operand_stack.append(not a)

            elif opcode == Opcode.JMP:
                self.ip = arg

            elif opcode == Opcode.JMP_IF_FALSE:
                cond = self.operand_stack.pop()
                if not cond:
                    self.ip = arg

            elif opcode == Opcode.JMP_IF_TRUE:
                cond = self.operand_stack.pop()
                if cond:
                    self.ip = arg

            elif opcode == Opcode.CALL:
                func_target, func_name, num_args = arg
                args = []
                for _ in range(num_args):
                    args.append(self.operand_stack.pop())
                args.reverse()

                new_frame = ActivationFrame(return_address=self.ip, func_name=func_name)
                self.call_stack.append(new_frame)

                for a in args:
                    self.operand_stack.append(a)

                self.ip = func_target

            elif opcode == Opcode.RET:
                val = self.operand_stack.pop()
                old_frame = self.call_stack.pop()
                self.ip = old_frame.return_address
                self.operand_stack.append(val)

            elif opcode == Opcode.ENTER_FRAME:
                pass

            elif opcode == Opcode.EXIT_FRAME:
                pass

            elif opcode == Opcode.ALLOC_STRUCT:
                field_names = arg
                field_vals = []
                for _ in range(len(field_names)):
                    field_vals.append(self.operand_stack.pop())
                field_vals.reverse()

                data = {name: val for name, val in zip(field_names, field_vals)}
                ref = self.allocate_heap("struct", data)
                self.operand_stack.append(ref)

            elif opcode == Opcode.GET_FIELD:
                field_name = arg
                ref = self.operand_stack.pop()
                if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
                    self.throw_exception(f"NullPointerReference: Cannot access field '{field_name}' on non-struct {ref}")
                    continue
                obj = self.heap[ref.ref_id]
                if obj.obj_type != "struct":
                    self.throw_exception(f"TypeError: Attempted field access on non-struct heap object of type {obj.obj_type}")
                    continue
                if field_name not in obj.data:
                    self.throw_exception(f"AttributeError: Struct has no field '{field_name}'")
                    continue
                self.operand_stack.append(obj.data[field_name])

            elif opcode == Opcode.SET_FIELD:
                field_name = arg
                val = self.operand_stack.pop()
                ref = self.operand_stack.pop()
                if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
                    self.throw_exception(f"NullPointerReference: Cannot set field '{field_name}' on non-struct {ref}")
                    continue
                obj = self.heap[ref.ref_id]
                if obj.obj_type != "struct":
                    self.throw_exception(f"TypeError: Attempted field set on non-struct heap object of type {obj.obj_type}")
                    continue
                obj.data[field_name] = val

            elif opcode == Opcode.ALLOC_MAP:
                num_pairs = arg
                pairs = []
                for _ in range(num_pairs):
                    val = self.operand_stack.pop()
                    key = self.operand_stack.pop()
                    pairs.append((key, val))
                
                data = {}
                for key, val in reversed(pairs):
                    data[key] = val
                ref = self.allocate_heap("map", data)
                self.operand_stack.append(ref)

            elif opcode == Opcode.ALLOC_ARRAY:
                num_elems = arg
                elems = []
                for _ in range(num_elems):
                    elems.append(self.operand_stack.pop())
                elems.reverse()
                ref = self.allocate_heap("array", elems)
                self.operand_stack.append(ref)

            elif opcode == Opcode.LEN:
                ref = self.operand_stack.pop()
                if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
                    self.throw_exception(f"NullPointerReference: Cannot perform LEN on {ref}")
                    continue
                obj = self.heap[ref.ref_id]
                self.operand_stack.append(len(obj.data))

            elif opcode == Opcode.GET_INDEX:
                idx = self.operand_stack.pop()
                ref = self.operand_stack.pop()
                if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
                    self.throw_exception(f"NullPointerReference: Cannot perform GET_INDEX on {ref}")
                    continue
                obj = self.heap[ref.ref_id]
                if obj.obj_type == "array":
                    if not isinstance(idx, int) or idx < 0 or idx >= len(obj.data):
                        self.throw_exception(f"IndexError: Array index {idx} out of range (length {len(obj.data)})")
                        continue
                    self.operand_stack.append(obj.data[idx])
                elif obj.obj_type == "map":
                    if idx not in obj.data:
                        self.throw_exception(f"KeyError: Map does not contain key {idx}")
                        continue
                    self.operand_stack.append(obj.data[idx])
                else:
                    self.throw_exception(f"TypeError: Index access not supported on type {obj.obj_type}")

            elif opcode == Opcode.SET_INDEX:
                val = self.operand_stack.pop()
                idx = self.operand_stack.pop()
                ref = self.operand_stack.pop()
                if not isinstance(ref, VMRef) or ref.ref_id not in self.heap:
                    self.throw_exception(f"NullPointerReference: Cannot perform SET_INDEX on {ref}")
                    continue
                obj = self.heap[ref.ref_id]
                if obj.obj_type == "array":
                    if not isinstance(idx, int) or idx < 0 or idx >= len(obj.data):
                        self.throw_exception(f"IndexError: Array index {idx} out of range (length {len(obj.data)})")
                        continue
                    obj.data[idx] = val
                elif obj.obj_type == "map":
                    obj.data[idx] = val
                else:
                    self.throw_exception(f"TypeError: Index set not supported on type {obj.obj_type}")

            elif opcode == Opcode.THROW:
                val = self.operand_stack.pop()
                self.throw_exception(val)

            elif opcode == Opcode.PUSH_TRY:
                if self.call_stack:
                    self.call_stack[-1].try_stack.append((arg, len(self.operand_stack)))

            elif opcode == Opcode.POP_TRY:
                if self.call_stack and self.call_stack[-1].try_stack:
                    self.call_stack[-1].try_stack.pop()

            elif opcode == Opcode.Q_ALLOC:
                qname = self.lookup_var(arg)
                self.simulator.allocate_qubit(qname)
                self.log_trace(f"Allocated qubit: '{qname}'")

            elif opcode == Opcode.Q_GATE:
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
                    continue

                args_str = f"({', '.join(map(str, angles))})" if angles else ""
                self.log_trace(f"Applied gate: {gate_name}{args_str} on {', '.join(resolved_targets)}")
                self.log_trace(f"  Current Quantum State: {self.format_amplitudes()}")

            elif opcode == Opcode.Q_MEASURE:
                qubit_name, cbit_name = arg
                resolved_q = self.lookup_var(qubit_name)
                outcome = self.simulator.measure(resolved_q)
                
                if self.call_stack:
                    self.call_stack[-1].locals[cbit_name] = outcome
                else:
                    self.globals[cbit_name] = outcome
                    
                self.log_trace(f"Measured qubit '{resolved_q}' -> stored in cbit '{cbit_name}' (value: {outcome})")
                self.log_trace(f"  Current Quantum State: {self.format_amplitudes()}")

            elif opcode == Opcode.Q_NOISE:
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

            elif opcode == Opcode.Q_TRACE:
                print(f"[TRACE DIRECTIVE] Quantum State: {self.format_amplitudes()}")

            elif opcode == Opcode.PRINT:
                val = self.operand_stack.pop()
                print(f"[PRINT DIRECTIVE] {val}")

        self.log_trace("Finished execution of Eigen VM bytecode")
