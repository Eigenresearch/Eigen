# ZX-Calculus based Equivalence Checker using Z-spiders and H-boxes
import math
import random
from src.zx.zx_graph import ZXGraph
from src.ir.ir_graph import EQIRGraph
from src.simulator import QuantumSimulator
from src.zx.exceptions import IndeterminateEquivalenceError

class Pauli:
    def __init__(self, n: int, phase: int = 0):
        self.n = n
        self.x = [False] * n
        self.z = [False] * n
        self.phase = phase  # 0: +1, 1: +i, 2: -1, 3: -i
        
    def copy(self):
        p = Pauli(self.n, self.phase)
        p.x = list(self.x)
        p.z = list(self.z)
        return p

    def apply_H(self, k: int):
        if self.x[k] and self.z[k]:
            self.phase = (self.phase + 2) % 4
        self.x[k], self.z[k] = self.z[k], self.x[k]
        
    def apply_S(self, k: int):
        if self.x[k]:
            if self.z[k]:
                self.phase = (self.phase + 3) % 4
            else:
                self.phase = (self.phase + 1) % 4
            self.z[k] = not self.z[k]

    def apply_X(self, k: int):
        if self.z[k]:
            self.phase = (self.phase + 2) % 4

    def apply_Y(self, k: int):
        if self.x[k] != self.z[k]:
            self.phase = (self.phase + 2) % 4

    def apply_Z(self, k: int):
        if self.x[k]:
            self.phase = (self.phase + 2) % 4

    def apply_CNOT(self, c: int, t: int):
        self.x[t] = self.x[t] ^ self.x[c]
        self.z[c] = self.z[c] ^ self.z[t]

    def apply_CZ(self, c: int, t: int):
        self.z[t] = self.z[t] ^ self.x[c]
        self.z[c] = self.z[c] ^ self.x[t]

    def apply_SWAP(self, q1: int, q2: int):
        self.x[q1], self.x[q2] = self.x[q2], self.x[q1]
        self.z[q1], self.z[q2] = self.z[q2], self.z[q1]

def apply_RY_pi2(p, k, sign=1):
    old_x = p.x[k]
    old_z = p.z[k]
    if old_x and not old_z:
        p.z[k] = True
        if sign == -1:
            p.phase = (p.phase + 2) % 4
    elif not old_x and old_z:
        p.x[k] = True
        p.z[k] = False
        if sign == 1:
            p.phase = (p.phase + 2) % 4
    elif old_x and old_z:
        p.x[k] = False
        if sign == 1:
            p.phase = (p.phase + 2) % 4

def is_clifford_circuit(graph: EQIRGraph) -> bool:
    clifford_gates = {'H', 'S', 'X', 'Y', 'Z', 'CNOT', 'CZ', 'SWAP'}
    for node in graph.nodes.values():
        if node.type == 'GATE':
            g_name = node.gate_name
            if g_name in clifford_gates:
                continue
            elif g_name in ('RX', 'RY', 'RZ'):
                if not node.args:
                    continue
                angle = node.args[0]
                k = round(angle / (math.pi / 2))
                if not math.isclose(angle, k * math.pi / 2, abs_tol=1e-6):
                    return False
            elif g_name == 'T':
                return False
            else:
                return False
    return True

def check_clifford_equivalence(graph1: EQIRGraph, graph2: EQIRGraph) -> bool:
    qubits1 = set()
    for node in graph1.nodes.values():
        if node.type == 'ALLOC':
            qubits1.add(node.targets[0])
    qubits2 = set()
    for node in graph2.nodes.values():
        if node.type == 'ALLOC':
            qubits2.add(node.targets[0])
            
    qubits = sorted(list(qubits1 | qubits2))
    N = len(qubits)
    if N == 0:
        return True
    qubit_map = {q: idx for idx, q in enumerate(qubits)}
    
    generators = []
    for i in range(N):
        p = Pauli(N)
        p.x[i] = True
        generators.append(p)
    for i in range(N):
        p = Pauli(N)
        p.z[i] = True
        generators.append(p)
        
    def apply_gate(g_name, targets, args):
        t_idxs = [qubit_map[t] for t in targets]
        for p in generators:
            if g_name == 'H':
                p.apply_H(t_idxs[0])
            elif g_name == 'S':
                p.apply_S(t_idxs[0])
            elif g_name == 'X':
                p.apply_X(t_idxs[0])
            elif g_name == 'Y':
                p.apply_Y(t_idxs[0])
            elif g_name == 'Z':
                p.apply_Z(t_idxs[0])
            elif g_name == 'CNOT':
                p.apply_CNOT(t_idxs[0], t_idxs[1])
            elif g_name == 'CZ':
                p.apply_CZ(t_idxs[0], t_idxs[1])
            elif g_name == 'SWAP':
                p.apply_SWAP(t_idxs[0], t_idxs[1])
            elif g_name in ('RX', 'RY', 'RZ'):
                angle = args[0] if args else 0.0
                k = round(angle / (math.pi / 2)) % 4
                if g_name == 'RZ':
                    if k == 1: p.apply_S(t_idxs[0])
                    elif k == 2: p.apply_Z(t_idxs[0])
                    elif k == 3:
                        p.apply_S(t_idxs[0])
                        p.apply_Z(t_idxs[0])
                elif g_name == 'RX':
                    if k == 1:
                        p.apply_H(t_idxs[0])
                        p.apply_S(t_idxs[0])
                        p.apply_H(t_idxs[0])
                    elif k == 2:
                        p.apply_X(t_idxs[0])
                    elif k == 3:
                        p.apply_H(t_idxs[0])
                        p.apply_S(t_idxs[0])
                        p.apply_Z(t_idxs[0])
                        p.apply_H(t_idxs[0])
                elif g_name == 'RY':
                    if k == 1:
                        apply_RY_pi2(p, t_idxs[0], 1)
                    elif k == 2:
                        p.apply_Y(t_idxs[0])
                    elif k == 3:
                        apply_RY_pi2(p, t_idxs[0], -1)

    for node in graph1.topological_sort():
        if node.type == 'GATE':
            apply_gate(node.gate_name, node.targets, node.args)
            
    nodes2 = graph2.topological_sort()
    for node in reversed(nodes2):
        if node.type == 'GATE':
            g_name = node.gate_name
            targets = node.targets
            args = node.args
            
            if g_name == 'S':
                apply_gate('S', targets, args)
                apply_gate('Z', targets, args)
            elif g_name in ('RX', 'RY', 'RZ'):
                neg_args = [-args[0]] if args else []
                apply_gate(g_name, targets, neg_args)
            else:
                apply_gate(g_name, targets, args)
                
    pi = {}
    for i in range(N):
        x_trues = [j for j in range(N) if generators[i].x[j]]
        z_trues = [j for j in range(N) if generators[i].z[j]]
        if len(x_trues) != 1 or z_trues:
            return False
        j = x_trues[0]
        if generators[i].phase != 0:
            return False
            
        x_trues_z = [k for k in range(N) if generators[N+i].x[k]]
        z_trues_z = [k for k in range(N) if generators[N+i].z[k]]
        if x_trues_z or len(z_trues_z) != 1 or z_trues_z[0] != j:
            return False
        if generators[N+i].phase != 0:
            return False
            
        pi[i] = j
        
    return len(set(pi.values())) == N


class ZXEquivalenceChecker:
    def __init__(self):
        pass

    def circuit_to_zx(self, graph: EQIRGraph) -> ZXGraph:
        zx = ZXGraph()
        qubit_wires = {}
        qubit_to_input = {}
        qubit_to_output = {}
        
        nodes = graph.topological_sort()
        
        for node in nodes:
            if node.type == 'ALLOC':
                q = node.targets[0]
                v = zx.add_vertex('Boundary')
                zx.inputs.append(v.id)
                qubit_to_input[q] = v.id
                
                z = zx.add_vertex('Z')
                zx.add_edge(v.id, z.id)
                qubit_wires[q] = z.id
            elif node.type == 'GATE':
                g_name = node.gate_name
                targets = node.targets
                args = node.args
                
                if g_name in ('H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ'):
                    q = targets[0]
                    prev_z_id = qubit_wires[q]
                    
                    if g_name == 'H':
                        h = zx.add_vertex('H')
                        next_z = zx.add_vertex('Z')
                        zx.add_edge(prev_z_id, h.id)
                        zx.add_edge(h.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'X':
                        h1 = zx.add_vertex('H')
                        z = zx.add_vertex('Z', 1.0)
                        h2 = zx.add_vertex('H')
                        next_z = zx.add_vertex('Z')
                        
                        zx.add_edge(prev_z_id, h1.id)
                        zx.add_edge(h1.id, z.id)
                        zx.add_edge(z.id, h2.id)
                        zx.add_edge(h2.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'Y':
                        z_y = zx.add_vertex('Z', 1.0)
                        h1 = zx.add_vertex('H')
                        z_x = zx.add_vertex('Z', 1.0)
                        h2 = zx.add_vertex('H')
                        next_z = zx.add_vertex('Z')
                        
                        zx.add_edge(prev_z_id, z_y.id)
                        zx.add_edge(z_y.id, h1.id)
                        zx.add_edge(h1.id, z_x.id)
                        zx.add_edge(z_x.id, h2.id)
                        zx.add_edge(h2.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'Z':
                        z = zx.add_vertex('Z', 1.0)
                        next_z = zx.add_vertex('Z')
                        zx.add_edge(prev_z_id, z.id)
                        zx.add_edge(z.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'S':
                        z = zx.add_vertex('Z', 0.5)
                        next_z = zx.add_vertex('Z')
                        zx.add_edge(prev_z_id, z.id)
                        zx.add_edge(z.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'T':
                        z = zx.add_vertex('Z', 0.25)
                        next_z = zx.add_vertex('Z')
                        zx.add_edge(prev_z_id, z.id)
                        zx.add_edge(z.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'RX':
                        phase = args[0] / math.pi if args else 0.0
                        h1 = zx.add_vertex('H')
                        z = zx.add_vertex('Z', phase)
                        h2 = zx.add_vertex('H')
                        next_z = zx.add_vertex('Z')
                        
                        zx.add_edge(prev_z_id, h1.id)
                        zx.add_edge(h1.id, z.id)
                        zx.add_edge(z.id, h2.id)
                        zx.add_edge(h2.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'RY':
                        phase = args[0] / math.pi if args else 0.0
                        z_pre = zx.add_vertex('Z', 0.5)
                        h1 = zx.add_vertex('H')
                        z_mid = zx.add_vertex('Z', phase)
                        h2 = zx.add_vertex('H')
                        z_post = zx.add_vertex('Z', -0.5)
                        next_z = zx.add_vertex('Z')

                        zx.add_edge(prev_z_id, z_pre.id)
                        zx.add_edge(z_pre.id, h1.id)
                        zx.add_edge(h1.id, z_mid.id)
                        zx.add_edge(z_mid.id, h2.id)
                        zx.add_edge(h2.id, z_post.id)
                        zx.add_edge(z_post.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'RZ':
                        phase = args[0] / math.pi if args else 0.0
                        z = zx.add_vertex('Z', phase)
                        next_z = zx.add_vertex('Z')
                        zx.add_edge(prev_z_id, z.id)
                        zx.add_edge(z.id, next_z.id)
                        qubit_wires[q] = next_z.id
                        
                elif g_name == 'CNOT':
                    c, t = targets[0], targets[1]
                    c_prev = qubit_wires[c]
                    t_prev = qubit_wires[t]
                    
                    z_c = zx.add_vertex('Z')
                    z_t = zx.add_vertex('Z')
                    h = zx.add_vertex('H')
                    
                    zx.add_edge(c_prev, z_c.id)
                    zx.add_edge(t_prev, z_t.id)
                    zx.add_edge(z_c.id, h.id)
                    zx.add_edge(h.id, z_t.id)
                    
                    next_c = zx.add_vertex('Z')
                    next_t = zx.add_vertex('Z')
                    zx.add_edge(z_c.id, next_c.id)
                    zx.add_edge(z_t.id, next_t.id)
                    
                    qubit_wires[c] = next_c.id
                    qubit_wires[t] = next_t.id
                    
                elif g_name == 'CZ':
                    c, t = targets[0], targets[1]
                    c_prev = qubit_wires[c]
                    t_prev = qubit_wires[t]
                    
                    z_c = zx.add_vertex('Z')
                    z_t = zx.add_vertex('Z')
                    
                    zx.add_edge(c_prev, z_c.id)
                    zx.add_edge(t_prev, z_t.id)
                    zx.add_edge(z_c.id, z_t.id)
                    
                    next_c = zx.add_vertex('Z')
                    next_t = zx.add_vertex('Z')
                    zx.add_edge(z_c.id, next_c.id)
                    zx.add_edge(z_t.id, next_t.id)
                    
                    qubit_wires[c] = next_c.id
                    qubit_wires[t] = next_t.id
                    
                elif g_name == 'SWAP':
                    q1, q2 = targets[0], targets[1]
                    qubit_wires[q1], qubit_wires[q2] = qubit_wires[q2], qubit_wires[q1]

        for q, last_v_id in qubit_wires.items():
            out_v = zx.add_vertex('Boundary')
            zx.add_edge(last_v_id, out_v.id)
            zx.outputs.append(out_v.id)
            qubit_to_output[q] = out_v.id
            
        zx.qubit_to_input = qubit_to_input
        zx.qubit_to_output = qubit_to_output
        return zx

    def simplify(self, zx: ZXGraph):
        from src.zx.local_complementation import LocalComplementer
        from src.zx.pivoting import Pivoter
        lc = LocalComplementer()
        piv = Pivoter()
        
        try:
            import eigen_native as native
        except ImportError:
            native = None
        
        while True:
            if native is not None:
                edges = []
                seen = set()
                for v_id, v in zx.vertices.items():
                    for neighbor_id in v.neighbors:
                        edge = (min(v_id, neighbor_id), max(v_id, neighbor_id))
                        if edge not in seen:
                            seen.add(edge)
                            edges.append(edge)
                unique_edges = native.fast_spider_fusion(edges)
                for v in zx.vertices.values():
                    v.neighbors.clear()
                for u, v in unique_edges:
                    if u in zx.vertices and v in zx.vertices:
                        zx.vertices[u].neighbors.add(v)
                        zx.vertices[v].neighbors.add(u)
            
            changed = False
            
            # 1. Z-spider fusion
            boundary_ids = set(zx.inputs) | set(zx.outputs)
            for v_id in list(zx.vertices.keys()):
                if v_id not in zx.vertices:
                    continue
                v = zx.vertices[v_id]
                if v.type != 'Z':
                    continue
                if v_id in boundary_ids:
                    continue
                for neighbor_id in list(v.neighbors):
                    if neighbor_id not in zx.vertices:
                        continue
                    neighbor = zx.vertices[neighbor_id]
                    if neighbor.type == 'Z':
                        if neighbor_id in boundary_ids:
                            continue
                        v.phase = (v.phase + neighbor.phase) % 2.0
                        for nn_id in neighbor.neighbors:
                            if nn_id != v.id:
                                zx.add_edge(v.id, nn_id)
                        zx.remove_vertex(neighbor_id)
                        changed = True
                        break
                if changed:
                    break
                    
            if changed:
                continue

            # 2. Identity spider removal
            for v_id in list(zx.vertices.keys()):
                if v_id not in zx.vertices:
                    continue
                v = zx.vertices[v_id]
                if v.type == 'Z' and v.phase == 0.0 and len(v.neighbors) == 2:
                    n1, n2 = list(v.neighbors)
                    zx.add_edge(n1, n2)
                    zx.remove_vertex(v_id)
                    changed = True
                    break
                    
            if changed:
                continue

            # 3. H-box cancellation
            for v_id in list(zx.vertices.keys()):
                if v_id not in zx.vertices:
                    continue
                v = zx.vertices[v_id]
                if v.type == 'H' and len(v.neighbors) == 2:
                    n1, n2 = list(v.neighbors)
                    if n1 in zx.vertices and zx.vertices[n1].type == 'H':
                        h2_neighbors = list(zx.vertices[n1].neighbors - {v_id})
                        h1_neighbors = list(v.neighbors - {n1})
                        if h2_neighbors and h1_neighbors:
                            zx.add_edge(h1_neighbors[0], h2_neighbors[0])
                        zx.remove_vertex(v_id)
                        zx.remove_vertex(n1)
                        changed = True
                        break
                    elif n2 in zx.vertices and zx.vertices[n2].type == 'H':
                        h2_neighbors = list(zx.vertices[n2].neighbors - {v_id})
                        h1_neighbors = list(v.neighbors - {n2})
                        if h2_neighbors and h1_neighbors:
                            zx.add_edge(h1_neighbors[0], h2_neighbors[0])
                        zx.remove_vertex(v_id)
                        zx.remove_vertex(n2)
                        changed = True
                        break
                        
            if changed:
                continue

            # 4. Local complementation rule
            if lc.local_complementation(zx):
                changed = True
                continue

            # 5. Pivoting rule
            if piv.pivot(zx):
                changed = True
                continue

            # 6. Hopf rule
            for u_id in list(zx.vertices.keys()):
                if u_id not in zx.vertices:
                    continue
                u = zx.vertices[u_id]
                if u.type != 'Z':
                    continue
                targets = {}
                for h_id in list(u.neighbors):
                    if h_id not in zx.vertices:
                        continue
                    h_box = zx.vertices[h_id]
                    if h_box.type == 'H' and len(h_box.neighbors) == 2:
                        other = list(h_box.neighbors - {u_id})[0]
                        if other in zx.vertices and zx.vertices[other].type == 'Z':
                            targets.setdefault(other, []).append(h_id)
                for _other, h_ids in targets.items():
                    if len(h_ids) >= 2:
                        to_remove = h_ids if len(h_ids) % 2 == 0 else h_ids[1:]
                        for r_id in to_remove:
                            zx.remove_vertex(r_id)
                        changed = True
                        break
                if changed:
                    break

            if changed:
                continue

            # 7. Phase Gadget Fusion
            gadgets_by_base = {}
            for g_id in list(zx.vertices.keys()):
                if g_id not in zx.vertices:
                    continue
                g = zx.vertices[g_id]
                if g.type == 'Z' and len(g.neighbors) == 1:
                    h_id = list(g.neighbors)[0]
                    if h_id in zx.vertices:
                        h = zx.vertices[h_id]
                        if h.type == 'H' and len(h.neighbors) == 2:
                            base_id = list(h.neighbors - {g_id})[0]
                            if base_id in zx.vertices:
                                base = zx.vertices[base_id]
                                if base.type == 'Z' and base.phase == 0.0:
                                    gadgets_by_base.setdefault(base_id, []).append((g_id, h_id))
            for _base_id, gadgets in gadgets_by_base.items():
                if len(gadgets) >= 2:
                    first_g_id, first_h_id = gadgets[0]
                    for g_id, h_id in gadgets[1:]:
                        zx.vertices[first_g_id].phase = (zx.vertices[first_g_id].phase + zx.vertices[g_id].phase) % 2.0
                        zx.remove_vertex(g_id)
                        zx.remove_vertex(h_id)
                    changed = True
                    break

            if not changed:
                break

    def are_equivalent(self, graph1: EQIRGraph, graph2: EQIRGraph) -> bool:
        # Check number of qubits
        qubits1 = set()
        for node in graph1.nodes.values():
            if node.type == 'ALLOC':
                qubits1.add(node.targets[0])
        qubits2 = set()
        for node in graph2.nodes.values():
            if node.type == 'ALLOC':
                qubits2.add(node.targets[0])
                
        all_qubits = sorted(list(qubits1 | qubits2))
        N = len(all_qubits)
        
        # 1. Clifford/tableau check
        if is_clifford_circuit(graph1) and is_clifford_circuit(graph2):
            if check_clifford_equivalence(graph1, graph2):
                return True
                
        # 2. ZX double-graph check
        zx1 = self.circuit_to_zx(graph1)
        zx2 = self.circuit_to_zx(graph2)
        
        if set(zx1.qubit_to_input.keys()) == set(zx2.qubit_to_input.keys()):
            dg = ZXGraph()
            g1_map = {}
            g2_map = {}
            
            for v_id, v in zx1.vertices.items():
                new_v = dg.add_vertex(v.type, v.phase)
                g1_map[v_id] = new_v.id
            for v_id, v in zx1.vertices.items():
                for neighbor_id in v.neighbors:
                    dg.add_edge(g1_map[v_id], g1_map[neighbor_id])
                    
            for v_id, v in zx2.vertices.items():
                phase = (-v.phase) % 2.0 if v.type in ('Z', 'X') else v.phase
                new_v = dg.add_vertex(v.type, phase)
                g2_map[v_id] = new_v.id
            for v_id, v in zx2.vertices.items():
                for neighbor_id in v.neighbors:
                    dg.add_edge(g2_map[v_id], g2_map[neighbor_id])
                    
            for q in zx1.qubit_to_input.keys():
                out1_id = g1_map[zx1.qubit_to_output[q]]
                out2_id = g2_map[zx2.qubit_to_output[q]]
                
                n1_list = list(dg.vertices[out1_id].neighbors)
                n2_list = list(dg.vertices[out2_id].neighbors)
                if n1_list and n2_list:
                    n1 = n1_list[0]
                    n2 = n2_list[0]
                    dg.add_edge(n1, n2)
                dg.remove_vertex(out1_id)
                dg.remove_vertex(out2_id)
                
            dg.inputs = [g1_map[v_id] for v_id in zx1.inputs]
            dg.outputs = [g2_map[v_id] for v_id in zx2.inputs]
            
            self.simplify(dg)
            
            # Check if double graph reduces to identity wires (potentially up to permutation)
            input_to_output = {}
            output_to_input = {}
            valid_reduction = True
            
            for in_id in dg.inputs:
                if in_id not in dg.vertices:
                    valid_reduction = False
                    break
                v = dg.vertices[in_id]
                if len(v.neighbors) != 1:
                    valid_reduction = False
                    break
                neighbor_id = list(v.neighbors)[0]
                if neighbor_id not in dg.outputs:
                    valid_reduction = False
                    break
                input_to_output[in_id] = neighbor_id
                
            if valid_reduction:
                for out_id in dg.outputs:
                    if out_id not in dg.vertices:
                        valid_reduction = False
                        break
                    v = dg.vertices[out_id]
                    if len(v.neighbors) != 1:
                        valid_reduction = False
                        break
                    neighbor_id = list(v.neighbors)[0]
                    if neighbor_id not in dg.inputs:
                        valid_reduction = False
                        break
                    output_to_input[out_id] = neighbor_id
                    
            if valid_reduction:
                if len(input_to_output) == len(dg.inputs) and len(output_to_input) == len(dg.outputs):
                    # Check other vertices are isolated
                    all_isolated = True
                    for v_id, v in dg.vertices.items():
                        if v_id not in dg.inputs and v_id not in dg.outputs:
                            if len(v.neighbors) > 0:
                                all_isolated = False
                                break
                    if all_isolated:
                        return True
                        
        # 3. Fallback checks (simulation/state-propagation)
        if N > 16:
            raise IndeterminateEquivalenceError(
                f"Circuit has {N} > 16 qubits. "
                "ZX equivalence checker cannot verify it without hanging."
            )
            
        if N <= 12:
            return self.check_via_unitary_simulation(graph1, graph2, all_qubits)
        else:
            return self.check_via_state_propagation(graph1, graph2, all_qubits)

    def check_via_unitary_simulation(self, graph1: EQIRGraph, graph2: EQIRGraph, all_qubits: list[str]) -> bool:
        # Full unitary matrix comparison (logic similar to EquivalenceChecker)
        U1 = self.generate_unitary(graph1, all_qubits)
        U2 = self.generate_unitary(graph2, all_qubits)
        
        dim = len(U1)
        max_val = 0.0
        max_idx = (0, 0)
        for r in range(dim):
            for c in range(dim):
                val = abs(U2[r][c])
                if val > max_val:
                    max_val = val
                    max_idx = (r, c)
                    
        r_max, c_max = max_idx
        v2 = U2[r_max][c_max]
        if abs(v2) < 1e-12:
            return False
            
        v1 = U1[r_max][c_max]
        global_phase = v1 / v2
        if abs(abs(global_phase) - 1.0) > 1e-5:
            return False
            
        for r in range(dim):
            for c in range(dim):
                if abs(U1[r][c] - global_phase * U2[r][c]) > 1e-5:
                    return False
        return True

    def generate_unitary(self, graph: EQIRGraph, qubit_order: list[str]) -> list[list[complex]]:
        dim = 2 ** len(qubit_order)
        U = [[0.0j for _ in range(dim)] for _ in range(dim)]
        nodes = graph.topological_sort()
        
        for col in range(dim):
            sim = QuantumSimulator()
            for qubit in qubit_order:
                sim.allocate_qubit(qubit)
            basis = [0.0j] * dim
            basis[col] = 1.0 + 0.0j
            sim.state_vector = basis
            
            for node in nodes:
                if node.type == 'GATE':
                    self.apply_gate_to_sim(sim, node)
                    
            final_state = sim.get_state_vector()
            for row in range(dim):
                U[row][col] = final_state[row]
        return U

    def apply_gate_to_sim(self, sim, node):
        g_name = node.gate_name
        targets = node.targets
        args = node.args
        if g_name == 'H': sim.H(targets[0])
        elif g_name == 'X': sim.X(targets[0])
        elif g_name == 'Y': sim.Y(targets[0])
        elif g_name == 'Z': sim.Z(targets[0])
        elif g_name == 'S': sim.S(targets[0])
        elif g_name == 'T': sim.T(targets[0])
        elif g_name == 'RX': sim.RX(targets[0], args[0])
        elif g_name == 'RY': sim.RY(targets[0], args[0])
        elif g_name == 'RZ': sim.RZ(targets[0], args[0])
        elif g_name == 'CNOT': sim.CNOT(targets[0], targets[1])
        elif g_name == 'CZ': sim.CZ(targets[0], targets[1])
        elif g_name == 'SWAP': sim.SWAP(targets[0], targets[1])

    def check_via_state_propagation(self, graph1: EQIRGraph, graph2: EQIRGraph, all_qubits: list[str]) -> bool:
        # Check equivalence by executing G1 o G2^\dagger on |0>^{\otimes N}, |+>^{\otimes N},
        # and 3 Haar-random entangled states, verifying that the overlap is near 1.
        N = len(all_qubits)
        dim = 2 ** N
        
        def prep_zero():
            vec = [0.0j] * dim
            vec[0] = 1.0 + 0.0j
            return vec
            
        def prep_plus():
            val = (1.0 / math.sqrt(2)) ** N
            return [complex(val, 0.0)] * dim
            
        def prep_haar(seed_val):
            rng = random.Random(seed_val)
            vec = []
            norm_sq = 0.0
            for _ in range(dim):
                u1 = rng.uniform(0.001, 1.0)
                u2 = rng.uniform(0.001, 1.0)
                r1 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                r2 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
                c = complex(r1, r2)
                vec.append(c)
                norm_sq += r1 * r1 + r2 * r2
            norm = math.sqrt(norm_sq)
            return [c / norm for c in vec]

        initial_states = [
            prep_zero(),
            prep_plus(),
            prep_haar(12345),
            prep_haar(67890),
            prep_haar(54321)
        ]
        
        for initial_state in initial_states:
            sim = QuantumSimulator()
            for q in all_qubits:
                sim.allocate_qubit(q)
                
            sim.state_vector = initial_state
            
            for node in graph1.topological_sort():
                if node.type == 'GATE':
                    self.apply_gate_to_sim(sim, node)
            for node in reversed(graph2.topological_sort()):
                if node.type == 'GATE':
                    self.apply_adjoint_gate_to_sim(sim, node)
                    
            final_state = sim.get_state_vector()
            
            overlap = sum(i.conjugate() * f for i, f in zip(initial_state, final_state, strict=False))
            if abs(abs(overlap) - 1.0) > 1e-9:
                return False
                
        return True

    def apply_adjoint_gate_to_sim(self, sim, node):
        g_name = node.gate_name
        targets = node.targets
        args = node.args
        if g_name == 'H': sim.H(targets[0])
        elif g_name == 'X': sim.X(targets[0])
        elif g_name == 'Y': sim.Y(targets[0])
        elif g_name == 'Z': sim.Z(targets[0])
        elif g_name == 'S':
            sim.apply_1qubit_gate(targets[0], [[1.0, 0.0], [0.0, -1j]])
        elif g_name == 'T':
            sim.apply_1qubit_gate(targets[0], [[1.0, 0.0], [0.0, 0.7071067811865475 - 0.7071067811865475j]])
        elif g_name in ('RX', 'RY', 'RZ'):
            if g_name == 'RX': sim.RX(targets[0], -args[0])
            elif g_name == 'RY': sim.RY(targets[0], -args[0])
            elif g_name == 'RZ': sim.RZ(targets[0], -args[0])
        elif g_name == 'CNOT': sim.CNOT(targets[0], targets[1])
        elif g_name == 'CZ': sim.CZ(targets[0], targets[1])
        elif g_name == 'SWAP': sim.SWAP(targets[0], targets[1])
