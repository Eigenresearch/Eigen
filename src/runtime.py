import random
import ast
from src.ir.ir_graph import EQIRGraph
from src.simulator import QuantumSimulator

class EigenRuntime:
    def __init__(self, trace_mode: bool = False, noise_model=None, sim_type: str = 'dense', gpu_platform: str = 'none', seed: int | None = None):
        self.rng = random.Random(seed)
        from src.noise.noise_model import NoiseModel
        self.simulator = QuantumSimulator(sim_type=sim_type, gpu_platform=gpu_platform, seed=seed)
        self.classical_store = {}  # cbit/int/float name -> value
        self.trace_mode = trace_mode
        self.trace_log = []
        self.noise_model = noise_model if noise_model is not None else NoiseModel(rng=self.rng)
        if getattr(self.noise_model, 'rng', None) is None:
            self.noise_model.rng = self.rng

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

    def evaluate_classical(self, expr) -> any:
        if not isinstance(expr, str):
            return expr
        if expr in self.classical_store:
            return self.classical_store[expr]
        
        def safe_eval(node, variables):
            if isinstance(node, ast.Expression):
                return safe_eval(node.body, variables)
            elif isinstance(node, ast.Constant):
                # `ast.Constant` covers all numeric/string/bool/None literals
                # since Python 3.8; the legacy `ast.Num` fallback below was
                # dead code that triggered a DeprecationWarning in 3.12 and
                # would break in 3.14 (where the alias is removed entirely).
                return node.value
            elif isinstance(node, ast.Name):
                if node.id in ('True', 'False', 'None'):
                    return {'True': True, 'False': False, 'None': None}[node.id]
                return variables.get(node.id, 0)
            elif isinstance(node, ast.UnaryOp):
                operand = safe_eval(node.operand, variables)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                elif isinstance(node.op, ast.USub):
                    return -operand
                elif isinstance(node.op, ast.Not):
                    return not operand
                elif isinstance(node.op, ast.Invert):
                    return ~operand
                else:
                    raise TypeError(f"Unsupported unary operator: {type(node.op)}")
            elif isinstance(node, ast.BinOp):
                left = safe_eval(node.left, variables)
                right = safe_eval(node.right, variables)
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
                elif isinstance(node.op, ast.Mod):
                    return left % right
                elif isinstance(node.op, ast.Pow):
                    return left ** right
                elif isinstance(node.op, ast.LShift):
                    return left << right
                elif isinstance(node.op, ast.RShift):
                    return left >> right
                elif isinstance(node.op, ast.BitOr):
                    return left | right
                elif isinstance(node.op, ast.BitXor):
                    return left ^ right
                elif isinstance(node.op, ast.BitAnd):
                    return left & right
                elif isinstance(node.op, ast.FloorDiv):
                    return left // right
                else:
                    raise TypeError(f"Unsupported binary operator: {type(node.op)}")
            elif isinstance(node, ast.Compare):
                left = safe_eval(node.left, variables)
                for op, comparator in zip(node.ops, node.comparators):
                    right = safe_eval(comparator, variables)
                    if isinstance(op, ast.Eq):
                        val = (left == right)
                    elif isinstance(op, ast.NotEq):
                        val = (left != right)
                    elif isinstance(op, ast.Lt):
                        val = (left < right)
                    elif isinstance(op, ast.LtE):
                        val = (left <= right)
                    elif isinstance(op, ast.Gt):
                        val = (left > right)
                    elif isinstance(op, ast.GtE):
                        val = (left >= right)
                    else:
                        raise TypeError(f"Unsupported comparison operator: {type(op)}")
                    if not val:
                        return False
                    left = right
                return True
            elif isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    for val_node in node.values:
                        val = safe_eval(val_node, variables)
                        if not val:
                            return False
                    return True
                elif isinstance(node.op, ast.Or):
                    for val_node in node.values:
                        val = safe_eval(val_node, variables)
                        if val:
                            return True
                    return False
                else:
                    raise TypeError(f"Unsupported boolean operator: {type(node.op)}")
            else:
                raise TypeError(f"Unsupported AST node: {type(node)}")

        try:
            tree = ast.parse(expr, mode='eval')
            return safe_eval(tree, self.classical_store)
        except Exception:
            return expr

    def execute(self, graph: EQIRGraph):
        if getattr(self.simulator, 'sim_type', None) == 'auto':
            from src.backend.gate_registry import CLIFFORD_GATES
            from src.backend.sim_selector import select_from_counts
            n_qubits = 0
            n_2q = 0
            n_gates = 0
            is_all_clifford = True
            for node in graph.nodes.values():
                if node.type == 'ALLOC':
                    n_qubits += 1
                elif node.type == 'GATE':
                    n_gates += 1
                    if node.gate_name in ('CNOT', 'CZ', 'SWAP'):
                        n_2q += 1
                    if is_all_clifford and node.gate_name not in CLIFFORD_GATES:
                        is_all_clifford = False
            noise_active = bool(self.noise_model and self.noise_model.noise_prob > 0)
            report = select_from_counts(
                n_qubits=n_qubits,
                n_2q_gates=n_2q,
                n_gates=n_gates,
                is_all_clifford=is_all_clifford,
                noise_active=noise_active,
            )
            self.simulator.configure_backend(report.chosen)

        nodes = graph.topological_sort()
        
        self.log_trace("Starting execution of EQIR v1.1 Graph")
        
        for node in nodes:
            # 1. Check classical condition
            if node.condition:
                cbit_name, op, expected_val = node.condition
                actual_val = self.evaluate_classical(cbit_name)
                exp_val = self.evaluate_classical(expected_val)
                if op == '==':
                    condition_met = (actual_val == exp_val)
                elif op == '!=':
                    condition_met = (actual_val != exp_val)
                elif op == '<':
                    condition_met = (actual_val < exp_val)
                elif op == '<=':
                    condition_met = (actual_val <= exp_val)
                elif op == '>':
                    condition_met = (actual_val > exp_val)
                elif op == '>=':
                    condition_met = (actual_val >= exp_val)
                else:
                    condition_met = False
                    
                if not condition_met:
                    self.log_trace(f"Skipping node {node.id} because condition {cbit_name} {op} {expected_val} failed (actual value is {actual_val})")
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
                elif g_name == 'CCX':
                    self.simulator.CCX(targets[0], targets[1], targets[2])
                elif g_name == 'CSWAP':
                    self.simulator.CSWAP(targets[0], targets[1], targets[2])
                elif g_name == 'CP':
                    self.simulator.CP(targets[0], targets[1], args[0])
                elif g_name == 'CRX':
                    self.simulator.CRX(targets[0], targets[1], args[0])
                elif g_name == 'CRY':
                    self.simulator.CRY(targets[0], targets[1], args[0])
                elif g_name == 'CRZ':
                    self.simulator.CRZ(targets[0], targets[1], args[0])
                else:
                    raise ValueError(f"Unknown gate type: {g_name}")
                
                # Apply global gate noise if active
                for target in targets:
                    self.noise_model.apply_gate_noise(self.simulator, target)
                
                args_str = f"({', '.join(map(str, args))})" if args else ""
                self.log_trace(f"Applied gate: {g_name}{args_str} on {', '.join(targets)}")
                if self.trace_mode:
                    self.log_trace(f"  Current Quantum State: {self.format_amplitudes()}")
                
            elif node.type == 'MEASURE':
                q_name = node.targets[0]
                c_name = node.cbit_name
                outcome = self.simulator.measure(q_name)
                outcome = self.noise_model.apply_readout_noise(outcome)
                self.classical_store[c_name] = outcome
                self.log_trace(f"Measured qubit '{q_name}' -> stored in cbit '{c_name}' (value: {outcome})")
                if self.trace_mode:
                    self.log_trace(f"  Current Quantum State: {self.format_amplitudes()}")
                
            elif node.type == 'TRACE':
                print(f"[TRACE DIRECTIVE] Quantum State: {self.format_amplitudes()}")
                
            elif node.type == 'PRINT':
                val = self.evaluate_classical(node.print_expr)
                print(f"[PRINT DIRECTIVE] {val}")
                
            elif node.type == 'ASSERT':
                left_ref, op, right_val = node.assert_cond
                left_val = self.evaluate_classical(left_ref)
                r_val = self.evaluate_classical(right_val)
                
                if op == '==':
                    assert_ok = (left_val == r_val)
                elif op == '!=':
                    assert_ok = (left_val != r_val)
                elif op == '<':
                    assert_ok = (left_val < r_val)
                elif op == '<=':
                    assert_ok = (left_val <= r_val)
                elif op == '>':
                    assert_ok = (left_val > r_val)
                elif op == '>=':
                    assert_ok = (left_val >= r_val)
                else:
                    assert_ok = False
                    
                if not assert_ok:
                    raise AssertionError(f"Eigen Assertion Failed: {left_ref} (value: {left_val}) {op} {right_val} at node {node.id}")
                self.log_trace(f"Assertion Passed: {left_ref} ({left_val}) {op} {right_val}")

        self.log_trace("Finished execution of EQIR v1.1 Graph")
