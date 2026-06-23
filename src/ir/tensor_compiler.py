"""Quantum Tensor Compiler for Eigen 2.3 — Helios.

Compiles quantum circuits from EQIR graphs into tensor network graphs,
computes optimized contraction orderings, and simulates circuits
via tensor contraction.
"""
import math
import cmath
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TensorNode:
    """Represents a tensor in the contraction graph."""
    def __init__(self, node_id: int, name: str, data=None, indices: list[str] = None):
        self.id = node_id
        self.name = name
        self.data = data  # numpy array or None
        self.indices = indices or []  # list of index label strings
    
    def rank(self) -> int:
        return len(self.indices)
    
    def __repr__(self) -> str:
        return f"TensorNode({self.id}, {self.name}, rank={self.rank()}, indices={self.indices})"


class ContractionEdge:
    """Represents a shared index between two tensors."""
    def __init__(self, node_a_id: int, node_b_id: int, index_label: str):
        self.node_a_id = node_a_id
        self.node_b_id = node_b_id
        self.index_label = index_label
    
    def __repr__(self) -> str:
        return f"Edge({self.node_a_id}<->{self.node_b_id}, '{self.index_label}')"


class TensorGraph:
    """Directed acyclic tensor contraction graph."""
    def __init__(self):
        self.nodes: dict[int, TensorNode] = {}
        self.edges: list[ContractionEdge] = []
        self.next_id = 0
        self.next_idx = 0
    
    def new_index(self, prefix: str = "i") -> str:
        self.next_idx += 1
        return f"{prefix}{self.next_idx}"
    
    def add_node(self, name: str, data=None, indices: list[str] = None) -> TensorNode:
        node = TensorNode(self.next_id, name, data, indices)
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node
    
    def add_edge(self, node_a_id: int, node_b_id: int, index_label: str):
        self.edges.append(ContractionEdge(node_a_id, node_b_id, index_label))
    
    def summary(self) -> dict:
        return {
            'num_tensors': len(self.nodes),
            'num_contractions': len(self.edges),
            'total_rank': sum(n.rank() for n in self.nodes.values()),
        }


class GreedyContractionOptimizer:
    """Computes a contraction ordering using greedy cost minimization."""
    
    def find_contraction_order(self, graph: TensorGraph) -> list[tuple[int, int, str]]:
        """Returns a list of (node_a_id, node_b_id, shared_index) in contraction order."""
        if not graph.edges:
            return []
        
        # Greedy: always contract the edge with the smallest combined rank first
        remaining_edges = list(graph.edges)
        order = []
        merged_into = {}  # old_id -> canonical_id
        
        def canonical(nid):
            while nid in merged_into:
                nid = merged_into[nid]
            return nid
        
        while remaining_edges:
            # Score each edge by combined rank of the two nodes
            best_idx = 0
            best_cost = float('inf')
            for i, edge in enumerate(remaining_edges):
                a = canonical(edge.node_a_id)
                b = canonical(edge.node_b_id)
                if a == b:
                    continue
                na = graph.nodes.get(a)
                nb = graph.nodes.get(b)
                if na is None or nb is None:
                    continue
                cost = na.rank() + nb.rank()
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
            
            edge = remaining_edges.pop(best_idx)
            a = canonical(edge.node_a_id)
            b = canonical(edge.node_b_id)
            if a != b:
                order.append((a, b, edge.index_label))
                # Merge b into a
                merged_into[b] = a
                na = graph.nodes[a]
                nb = graph.nodes[b]
                # Remove shared index from both, union the rest
                new_indices = [idx for idx in na.indices if idx != edge.index_label] + \
                              [idx for idx in nb.indices if idx != edge.index_label]
                na.indices = new_indices
        
        return order


class TensorCircuitCompiler:
    """Compiles an EQIR graph into a TensorGraph for contraction-based simulation."""
    
    # Standard gate matrices
    GATE_MATRICES = {}
    
    @staticmethod
    def _init_gate_matrices():
        if TensorCircuitCompiler.GATE_MATRICES:
            return
        if not HAS_NUMPY:
            return
        inv = 1.0 / math.sqrt(2.0)
        TensorCircuitCompiler.GATE_MATRICES = {
            'H': np.array([[inv, inv], [inv, -inv]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            'S': np.array([[1, 0], [0, 1j]], dtype=complex),
            'T': np.array([[1, 0], [0, inv + inv*1j]], dtype=complex),
            'CNOT': np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex).reshape(2,2,2,2),
            'CZ': np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex).reshape(2,2,2,2),
            'SWAP': np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex).reshape(2,2,2,2),
        }
    
    def compile_eqir(self, eqir_graph) -> TensorGraph:
        """Compile an EQIR graph to a TensorGraph."""
        self._init_gate_matrices()
        tg = TensorGraph()
        
        # Track current output index for each qubit wire
        qubit_wire = {}  # qubit_name -> current_index_label
        
        nodes = eqir_graph.topological_sort()
        
        for node in nodes:
            if node.type == 'ALLOC':
                qname = node.targets[0]
                idx = tg.new_index(f"q{qname}_")
                # Initial state |0> as a rank-1 tensor
                if HAS_NUMPY:
                    data = np.array([1.0+0j, 0.0+0j], dtype=complex)
                else:
                    data = None
                tn = tg.add_node(f"init_{qname}", data=data, indices=[idx])
                qubit_wire[qname] = (idx, tn.id)
                
            elif node.type == 'GATE':
                gate_name = node.gate_name
                targets = node.targets
                
                if len(targets) == 1:
                    q = targets[0]
                    if q not in qubit_wire:
                        continue
                    in_idx, prev_node_id = qubit_wire[q]
                    out_idx = tg.new_index(f"q{q}_")
                    
                    if HAS_NUMPY and gate_name in self.GATE_MATRICES:
                        data = self.GATE_MATRICES[gate_name].copy()
                    elif HAS_NUMPY and gate_name in ('RX', 'RY', 'RZ') and node.args:
                        theta = float(node.args[0])
                        if gate_name == 'RX':
                            c, s = math.cos(theta/2), math.sin(theta/2)
                            data = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
                        elif gate_name == 'RY':
                            c, s = math.cos(theta/2), math.sin(theta/2)
                            data = np.array([[c, -s], [s, c]], dtype=complex)
                        else:  # RZ
                            data = np.array([[cmath.exp(-1j*theta/2), 0], [0, cmath.exp(1j*theta/2)]], dtype=complex)
                    else:
                        data = None
                    
                    tn = tg.add_node(f"gate_{gate_name}_{q}", data=data, indices=[out_idx, in_idx])
                    tg.add_edge(prev_node_id, tn.id, in_idx)
                    qubit_wire[q] = (out_idx, tn.id)
                    
                elif len(targets) == 2:
                    q0, q1 = targets[0], targets[1]
                    if q0 not in qubit_wire or q1 not in qubit_wire:
                        continue
                    in_idx0, prev0 = qubit_wire[q0]
                    in_idx1, prev1 = qubit_wire[q1]
                    out_idx0 = tg.new_index(f"q{q0}_")
                    out_idx1 = tg.new_index(f"q{q1}_")
                    
                    if HAS_NUMPY and gate_name in self.GATE_MATRICES:
                        data = self.GATE_MATRICES[gate_name].copy()
                    else:
                        data = None
                    
                    tn = tg.add_node(f"gate_{gate_name}_{q0}_{q1}", data=data,
                                     indices=[out_idx0, out_idx1, in_idx0, in_idx1])
                    tg.add_edge(prev0, tn.id, in_idx0)
                    tg.add_edge(prev1, tn.id, in_idx1)
                    qubit_wire[q0] = (out_idx0, tn.id)
                    qubit_wire[q1] = (out_idx1, tn.id)
        
        return tg
    
    def simulate(self, eqir_graph) -> list[complex]:
        """Full tensor contraction simulation of an EQIR circuit."""
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for tensor contraction simulation.")
        
        tg = self.compile_eqir(eqir_graph)
        
        if not tg.nodes:
            return [1.0 + 0.0j]
        
        optimizer = GreedyContractionOptimizer()
        order = optimizer.find_contraction_order(tg)
        
        # Build tensor data map
        tensor_data = {}
        for nid, tn in tg.nodes.items():
            if tn.data is not None:
                tensor_data[nid] = (tn.data, list(tn.indices))
        
        # If no contraction edges, do outer products
        if not order:
            result = None
            for nid, (data, indices) in tensor_data.items():
                if result is None:
                    result = data
                else:
                    result = np.tensordot(result, data, axes=0).flatten()
            return list(result) if result is not None else [1.0 + 0.0j]
        
        # Execute contractions in order
        merged = {}  # canonical_id -> (data, indices)
        for nid, (data, indices) in tensor_data.items():
            merged[nid] = (data, list(indices))
        
        canonical_map = {}
        def canonical(nid):
            while nid in canonical_map:
                nid = canonical_map[nid]
            return nid
        
        for a_id, b_id, shared_idx in order:
            ca = canonical(a_id)
            cb = canonical(b_id)
            if ca == cb:
                continue
            if ca not in merged or cb not in merged:
                continue
            
            data_a, idx_a = merged[ca]
            data_b, idx_b = merged[cb]
            
            if shared_idx in idx_a and shared_idx in idx_b:
                ax_a = idx_a.index(shared_idx)
                ax_b = idx_b.index(shared_idx)
                result = np.tensordot(data_a, data_b, axes=([ax_a], [ax_b]))
                new_idx = [i for i in idx_a if i != shared_idx] + [i for i in idx_b if i != shared_idx]
            else:
                result = np.tensordot(data_a, data_b, axes=0)
                new_idx = idx_a + idx_b
            
            merged[ca] = (result, new_idx)
            canonical_map[cb] = ca
            if cb in merged:
                del merged[cb]
        
        # The final result should be a single tensor
        if merged:
            final_id = list(merged.keys())[0]
            final_data, final_idx = merged[final_id]
            return list(final_data.flatten())
        
        return [1.0 + 0.0j]
