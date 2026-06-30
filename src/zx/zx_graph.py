# ZX-Calculus Graph Representation for Eigen
class ZXVertex:
    def __init__(self, vertex_id: int, vertex_type: str, phase: float = 0.0):
        # vertex_type: 'Z', 'X', 'H', 'Boundary'
        self.id = vertex_id
        self.type = vertex_type
        self.phase = phase % 2.0  # Phase in multiples of PI (0 to 2)
        self.neighbors = set()    # Set of vertex IDs

    def __repr__(self) -> str:
        phase_str = f"({self.phase:.2f}pi)" if self.phase != 0.0 else ""
        return f"{self.type}{self.id}{phase_str}"

class ZXGraph:
    def __init__(self):
        self.vertices = {}  # vertex_id -> ZXVertex
        self.next_vertex_id = 0
        self.inputs = []    # List of input Boundary vertex IDs
        self.outputs = []   # List of output Boundary vertex IDs
        self.hadamard_edges = set()  # Set of frozenset({v1_id, v2_id}) for Hadamard edges

    def add_vertex(self, vertex_type: str, phase: float = 0.0) -> ZXVertex:
        v_id = self.next_vertex_id
        self.next_vertex_id += 1
        v = ZXVertex(v_id, vertex_type, phase)
        self.vertices[v_id] = v
        return v

    def add_edge(self, v1_id: int, v2_id: int, hadamard: bool = False):
        if v1_id in self.vertices and v2_id in self.vertices:
            edge_key = frozenset({v1_id, v2_id})
            if hadamard:
                self.hadamard_edges.add(edge_key)
                self.vertices[v1_id].neighbors.discard(v2_id)
                self.vertices[v2_id].neighbors.discard(v1_id)
            else:
                self.hadamard_edges.discard(edge_key)
                self.vertices[v1_id].neighbors.add(v2_id)
                self.vertices[v2_id].neighbors.add(v1_id)

    def remove_edge(self, v1_id: int, v2_id: int):
        if v1_id in self.vertices and v2_id in self.vertices:
            self.vertices[v1_id].neighbors.discard(v2_id)
            self.vertices[v2_id].neighbors.discard(v1_id)
            self.hadamard_edges.discard(frozenset({v1_id, v2_id}))

    def is_hadamard_edge(self, v1_id: int, v2_id: int) -> bool:
        return frozenset({v1_id, v2_id}) in self.hadamard_edges

    def remove_vertex(self, v_id: int):
        if v_id in self.vertices:
            v = self.vertices[v_id]
            for neighbor_id in list(v.neighbors):
                self.vertices[neighbor_id].neighbors.discard(v_id)
            del self.vertices[v_id]
            if v_id in self.inputs:
                self.inputs.remove(v_id)
            if v_id in self.outputs:
                self.outputs.remove(v_id)
