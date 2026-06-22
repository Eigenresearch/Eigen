from src.ir_graph import EQIRGraph, EQIRNode
from src.simulator import QuantumSimulator

class EigenRuntime:
    def __init__(self, trace_mode: bool = False):
        self.simulator = QuantumSimulator()
        self.classical_store = {}  # cbit/int/float name -> value
        self.trace_mode = trace_mode
        self.trace_log = []

    def log_trace(self, msg: str):
        self.trace_log.append(msg)
        if self.trace_mode:
            print(f"[TRACE] {msg}")

    def format_amplitudes(self) -> str:
        amps = self.simulator.get_amplitudes_dict()
        parts = []
        for state, amp in sorted(amps.items()):
            prob = abs(amp) ** 2
            # format complex nicely
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

    def execute(self, graph: EQIRGraph):
        nodes = graph.topological_sort()
        
        self.log_trace("Starting execution of EQIR v1.1 Graph")
        
        for node in nodes:
            # 1. Check classical condition
            if node.condition:
                cbit_name, op, expected_val = node.condition
                actual_val = self.classical_store.get(cbit_name, 0)
                if op == '==':
                    condition_met = (actual_val == expected_val)
                else:
                    condition_met = False
                    
                if not condition_met:
                    self.log_trace(f"Skipping node {node.id} because condition {cbit_name} == {expected_val} failed (actual value is {actual_val})")
                    continue

            # 2. Execute node based on type
            if node.type == 'ALLOC':
                q_name = node.targets[0]
                self.simulator.allocate_qubit(q_name)
                self.log_trace(f"Allocated qubit: '{q_name}'")
                
            elif node.type == 'GATE':
                g_name = node.gate_name
                targets = node.targets
                args = node.args
                
                if g_name == 'H':
                    self.simulator.H(targets[0])
                elif g_name == 'X':
                    self.simulator.X(targets[0])
                elif g_name == 'Y':
                    self.simulator.Y(targets[0])
                elif g_name == 'Z':
                    self.simulator.Z(targets[0])
                elif g_name == 'S':
                    self.simulator.S(targets[0])
                elif g_name == 'T':
                    self.simulator.T(targets[0])
                elif g_name == 'RX':
                    self.simulator.RX(targets[0], args[0])
                elif g_name == 'RY':
                    self.simulator.RY(targets[0], args[0])
                elif g_name == 'RZ':
                    self.simulator.RZ(targets[0], args[0])
                elif g_name == 'CNOT':
                    self.simulator.CNOT(targets[0], targets[1])
                elif g_name == 'CZ':
                    self.simulator.CZ(targets[0], targets[1])
                elif g_name == 'SWAP':
                    self.simulator.SWAP(targets[0], targets[1])
                else:
                    raise ValueError(f"Unknown gate type: {g_name}")
                
                args_str = f"({', '.join(map(str, args))})" if args else ""
                self.log_trace(f"Applied gate: {g_name}{args_str} on {', '.join(targets)}")
                self.log_trace(f"  Current Quantum State: {self.format_amplitudes()}")
                
            elif node.type == 'MEASURE':
                q_name = node.targets[0]
                c_name = node.cbit_name
                outcome = self.simulator.measure(q_name)
                self.classical_store[c_name] = outcome
                self.log_trace(f"Measured qubit '{q_name}' -> stored in cbit '{c_name}' (value: {outcome})")
                self.log_trace(f"  Current Quantum State: {self.format_amplitudes()}")
                
            elif node.type == 'TRACE':
                print(f"[TRACE DIRECTIVE] Quantum State: {self.format_amplitudes()}")
                
            elif node.type == 'PRINT':
                val = node.print_expr
                # If it's a string, it represents a variable reference
                if isinstance(val, str):
                    val = self.classical_store.get(val, f"<undefined variable '{val}'>")
                print(f"[PRINT DIRECTIVE] {val}")
                
            elif node.type == 'ASSERT':
                left_ref, op, right_val = node.assert_cond
                left_val = self.classical_store.get(left_ref, 0)
                if op == '==':
                    assert_ok = (left_val == right_val)
                else:
                    assert_ok = False
                    
                if not assert_ok:
                    raise AssertionError(f"Eigen Assertion Failed: {left_ref} (value: {left_val}) {op} {right_val} at node {node.id}")
                self.log_trace(f"Assertion Passed: {left_ref} ({left_val}) {op} {right_val}")

        self.log_trace("Finished execution of EQIR v1.1 Graph")
