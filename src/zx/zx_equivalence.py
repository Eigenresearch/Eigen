# ZX-Calculus based Equivalence Checker using Z-spiders and H-boxes
import math
from src.zx.zx_graph import ZXGraph
from src.ir.ir_graph import EQIRGraph

class ZXEquivalenceChecker:
    def __init__(self):
        pass

    def circuit_to_zx(self, graph: EQIRGraph) -> ZXGraph:
        zx = ZXGraph()
        qubit_wires = {}
        
        # Sort nodes topologically
        nodes = graph.topological_sort()
        
        for node in nodes:
            if node.type == 'ALLOC':
                q = node.targets[0]
                v = zx.add_vertex('Boundary')
                zx.inputs.append(v.id)
                
                # Start each wire with a Z spider
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
                        # prev_Z -> H -> next_Z
                        h = zx.add_vertex('H')
                        next_z = zx.add_vertex('Z')
                        zx.add_edge(prev_z_id, h.id)
                        zx.add_edge(h.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'X':
                        # X is H -> Z(1.0) -> H
                        h1 = zx.add_vertex('H')
                        z = zx.add_vertex('Z', 1.0)
                        h2 = zx.add_vertex('H')
                        next_z = zx.add_vertex('Z')
                        
                        zx.add_edge(prev_z_id, h1.id)
                        zx.add_edge(h1.id, z.id)
                        zx.add_edge(z.id, h2.id)
                        zx.add_edge(h2.id, next_z.id)
                        qubit_wires[q] = next_z.id
                    elif g_name == 'Z':
                        # Z(1.0) is just a phase on Z spider
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
                        # H -> Z(phase) -> H
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
                    
                    # CNOT has Z spider on c and X spider on t connected by an edge.
                    # In Z/H representation, this is a Z spider on c and Z spider on t connected by an H-box.
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
                    
                elif g_name == 'SWAP':
                    q1, q2 = targets[0], targets[1]
                    qubit_wires[q1], qubit_wires[q2] = qubit_wires[q2], qubit_wires[q1]

        # Add output boundary vertices
        for q, last_v_id in qubit_wires.items():
            out_v = zx.add_vertex('Boundary')
            zx.add_edge(last_v_id, out_v.id)
            zx.outputs.append(out_v.id)
            
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
        
        # Repeatedly simplify to fixed-point
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
            
            # 1. Z-spider fusion (adjacent Z spiders merge)
            for v_id in list(zx.vertices.keys()):
                if v_id not in zx.vertices:
                    continue
                v = zx.vertices[v_id]
                if v.type != 'Z':
                    continue
                for neighbor_id in list(v.neighbors):
                    if neighbor_id not in zx.vertices:
                        continue
                    neighbor = zx.vertices[neighbor_id]
                    if neighbor.type == 'Z':
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

            # 2. Identity spider removal: Z-spider with phase 0 and exactly 2 neighbors
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

            # 3. H-box cancellation: two adjacent H-boxes cancel to an identity wire
            # H1 - H2  ->  wire
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

            # 6. Hopf rule: parallel H-box cancellation between two Z-spiders
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
                for other, h_ids in targets.items():
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
            for base_id, gadgets in gadgets_by_base.items():
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
        zx1 = self.circuit_to_zx(graph1)
        zx2 = self.circuit_to_zx(graph2)
        
        self.simplify(zx1)
        self.simplify(zx2)
        
        # Structural check
        if len(zx1.vertices) != len(zx2.vertices):
            return False
            
        phases1 = sorted([round(v.phase, 4) for v in zx1.vertices.values() if v.type == 'Z'])
        phases2 = sorted([round(v.phase, 4) for v in zx2.vertices.values() if v.type == 'Z'])
        if phases1 != phases2:
            return False
            
        return True
