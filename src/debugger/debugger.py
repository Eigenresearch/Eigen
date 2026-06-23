import sys
from src.backend.bytecode import Opcode, Instruction
from src.backend.vm import EigenVM

class EigenDebugger:
    def __init__(self, trace_mode: bool = False):
        self.vm = EigenVM(trace_mode=trace_mode)
        self.breakpoints = set()  # set of line numbers (int)
        self.watchpoints = {}     # var_name -> last_seen_value
        self.step_mode = True

    def add_breakpoint(self, line: int):
        self.breakpoints.add(line)
        print(f"Breakpoint set at line {line}")

    def remove_breakpoint(self, line: int):
        self.breakpoints.discard(line)
        print(f"Breakpoint removed at line {line}")

    def debug(self, instructions: list[Instruction]):
        self.vm.instructions = instructions
        self.vm.ip = 0
        self.vm.operand_stack = []
        self.vm.call_stack = [self.vm.get_frame(None, "main")]
        self.vm.globals = {}
        self.vm.heap = {}
        self.vm.next_ref_id = 1

        print("=" * 50)
        print("          EIGEN INTERACTIVE DEBUGGER          ")
        print("=" * 50)
        print("Type 'help' or 'h' for list of commands.")
        
        while self.vm.ip < len(self.vm.instructions):
            instr = self.vm.instructions[self.vm.ip]
            current_line = instr.line
            
            # Check breakpoints
            if current_line in self.breakpoints:
                print(f"\n[BREAKPOINT] Reached line {current_line}")
                self.step_mode = True

            # Check watchpoints
            for var_name, last_val in list(self.watchpoints.items()):
                curr_val = self.vm.lookup_var(var_name)
                if curr_val != last_val:
                    print(f"\n[WATCH] Value of '{var_name}' changed: {last_val} -> {curr_val}")
                    self.watchpoints[var_name] = curr_val
                    self.step_mode = True

            if self.step_mode:
                self.interactive_loop(instr)

            # Execute single step
            self.execute_single_step(instr)

        print("\nVM Execution finished successfully.")

    def execute_single_step(self, instr: Instruction):
        # We manually run a single step in the VM
        self.vm.ip += 1
        if self.vm.call_stack and instr.line is not None:
            self.vm.call_stack[-1].current_line = instr.line

        opcode = instr.opcode
        arg = instr.arg
        
        # Dispatch
        try:
            if opcode == Opcode.LOAD_CONST:
                self.vm.operand_stack.append(arg)
            elif opcode == Opcode.LOAD_VAR:
                self.vm.operand_stack.append(self.vm.lookup_var(arg))
            elif opcode == Opcode.STORE_VAR:
                val = self.vm.operand_stack.pop()
                if self.vm.call_stack:
                    self.vm.call_stack[-1].locals[arg] = val
                else:
                    self.vm.globals[arg] = val
            elif opcode == Opcode.ADD:
                b = self.vm.operand_stack.pop()
                a = self.vm.operand_stack.pop()
                self.vm.operand_stack.append(a + b)
            elif opcode == Opcode.SUB:
                b = self.vm.operand_stack.pop()
                a = self.vm.operand_stack.pop()
                self.vm.operand_stack.append(a - b)
            else:
                self.vm.dispatch_table[opcode](arg)
        except IndexError:
            self.vm.throw_exception("StackUnderflowError: Operand stack underflow.")
        except Exception as e:
            self.vm.throw_exception(str(e))

    def interactive_loop(self, instr: Instruction):
        line_info = f" (line {instr.line})" if instr.line is not None else ""
        print(f"IP {self.vm.ip}: {instr}{line_info}")
        
        while True:
            sys.stdout.write("eigen-dbg> ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                # EOF
                sys.exit(0)
                
            cmd_parts = line.strip().split()
            if not cmd_parts:
                continue
                
            cmd = cmd_parts[0].lower()
            
            if cmd in ('help', 'h'):
                self.print_help()
            elif cmd in ('step', 's'):
                self.step_mode = True
                break
            elif cmd in ('next', 'n'):
                self.step_mode = True
                break
            elif cmd in ('continue', 'c'):
                self.step_mode = False
                break
            elif cmd in ('break', 'b'):
                if len(cmd_parts) < 2:
                    print("Error: Specify a line number for breakpoint.")
                else:
                    try:
                        self.add_breakpoint(int(cmd_parts[1]))
                    except ValueError:
                        print("Error: Line number must be an integer.")
            elif cmd in ('delete', 'd'):
                if len(cmd_parts) < 2:
                    print("Error: Specify a line number.")
                else:
                    try:
                        self.remove_breakpoint(int(cmd_parts[1]))
                    except ValueError:
                        print("Error: Line number must be an integer.")
            elif cmd in ('locals', 'l'):
                if self.vm.call_stack:
                    print("Locals:", self.vm.call_stack[-1].locals)
                print("Globals:", self.vm.globals)
            elif cmd in ('watch', 'w'):
                if len(cmd_parts) < 2:
                    print("Error: Specify variable name to watch.")
                else:
                    var = cmd_parts[1]
                    val = self.vm.lookup_var(var)
                    self.watchpoints[var] = val
                    print(f"Watchpoint set on '{var}' (current value: {val})")
            elif cmd in ('stack', 't'):
                print("Operand Stack:", self.vm.operand_stack)
                print("Call Stack depth:", len(self.vm.call_stack))
            elif cmd in ('quantum', 'q'):
                print("Quantum state:", self.vm.format_amplitudes())
            elif cmd in ('quit', 'exit'):
                print("Aborting debugger.")
                sys.exit(0)
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' or 'h' for commands.")

    def print_help(self):
        print("Commands:")
        print("  step, s         - Execute current instruction and step to next")
        print("  next, n         - Step to next instruction")
        print("  continue, c     - Continue execution")
        print("  break, b <line> - Set breakpoint at line number")
        print("  delete, d <line>- Delete breakpoint")
        print("  locals, l       - Print local and global variables")
        print("  watch, w <var>  - Set a watchpoint on a variable name")
        print("  stack, t        - Print current stacks")
        print("  quantum, q      - Print current wavefunction state vector")
        print("  quit, exit      - Abort debugger")
