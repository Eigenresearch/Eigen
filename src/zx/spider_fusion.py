# Spider fusion rule implementation for ZX-Calculus
from src.zx.zx_graph import ZXGraph

class SpiderFuser:
    def fuse_spiders(self, graph: ZXGraph) -> bool:
        changed = False
        for v_id in list(graph.vertices.keys()):
            if v_id not in graph.vertices:
                continue
            v = graph.vertices[v_id]
            if v.type not in ('Z', 'X'):
                continue
                
            for neighbor_id in list(v.neighbors):
                if neighbor_id not in graph.vertices:
                    continue
                neighbor = graph.vertices[neighbor_id]
                
                # Fuse if same color
                if neighbor.type == v.type:
                    # Update phase
                    v.phase = (v.phase + neighbor.phase) % 2.0
                    
                    # Connect neighbor's neighbors to v
                    for n_n_id in neighbor.neighbors:
                        if n_n_id != v.id:
                            graph.add_edge(v.id, n_n_id)
                            
                    # Remove neighbor
                    graph.remove_vertex(neighbor_id)
                    changed = True
                    break  # Break inner loop since neighbor was removed
        return changed
