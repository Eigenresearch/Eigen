# Local complementation rule for ZX-Calculus in Z-H representation
from src.zx.zx_graph import ZXGraph

class LocalComplementer:
    def local_complementation(self, graph: ZXGraph) -> bool:
        changed = False
        for v_id in list(graph.vertices.keys()):
            if v_id not in graph.vertices:
                continue
            v = graph.vertices[v_id]
            
            # Local complementation applies to Z-spiders with phase +/- 0.5 (pi/2)
            if v.type == 'Z' and round(abs(v.phase) % 1.0, 5) == 0.5:
                # Find all neighbor Z-spiders (connected via H-boxes)
                neighbors = []
                valid = True
                h_boxes = []
                for n_id in list(v.neighbors):
                    if n_id not in graph.vertices:
                        valid = False
                        break
                    n_node = graph.vertices[n_id]
                    if n_node.type == 'H' and len(n_node.neighbors) == 2:
                        other_ids = list(n_node.neighbors - {v_id})
                        if other_ids and other_ids[0] in graph.vertices:
                            other_node = graph.vertices[other_ids[0]]
                            if other_node.type == 'Z':
                                neighbors.append(other_node.id)
                                h_boxes.append(n_id)
                                continue
                    valid = False
                    break
                
                if not valid or not neighbors:
                    continue
                
                # Perform local complementation:
                # 1. Toggle H-connections between all pairs of neighbors
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        w1, w2 = neighbors[i], neighbors[j]
                        # Check if w1 and w2 are connected via an H-box
                        shared_h = None
                        for w1_neigh_id in graph.vertices[w1].neighbors:
                            if w1_neigh_id in graph.vertices[w2].neighbors:
                                if graph.vertices[w1_neigh_id].type == 'H':
                                    shared_h = w1_neigh_id
                                    break
                        if shared_h is not None:
                            # Remove connection
                            graph.remove_vertex(shared_h)
                        else:
                            # Add H-connection
                            h_new = graph.add_vertex('H')
                            graph.add_edge(w1, h_new.id)
                            graph.add_edge(h_new.id, w2)
                            
                # 2. Add -v.phase to neighbors
                for w_id in neighbors:
                    graph.vertices[w_id].phase = (graph.vertices[w_id].phase - v.phase) % 2.0
                    
                # 3. Remove v and its H-boxes
                for h_id in h_boxes:
                    graph.remove_vertex(h_id)
                graph.remove_vertex(v_id)
                changed = True
                break
                
        return changed
