# Pivoting rule implementation for ZX-Calculus simplification in Z-H representation
from src.zx.zx_graph import ZXGraph

class Pivoter:
    def pivot(self, graph: ZXGraph) -> bool:
        changed = False
        for v1_id in list(graph.vertices.keys()):
            if v1_id not in graph.vertices:
                continue
            u = graph.vertices[v1_id]
            if u.type != 'Z' or round(u.phase % 1.0, 5) not in (0.0, 1.0):
                continue
            
            # Find an adjacent Z-spider connected via an H-box
            for h_id in list(u.neighbors):
                if h_id not in graph.vertices:
                    continue
                h_box = graph.vertices[h_id]
                if h_box.type != 'H' or len(h_box.neighbors) != 2:
                    continue
                
                other_ids = list(h_box.neighbors - {v1_id})
                if not other_ids or other_ids[0] not in graph.vertices:
                    continue
                v2_id = other_ids[0]
                v = graph.vertices[v2_id]
                
                if v.type != 'Z' or round(v.phase % 1.0, 5) not in (0.0, 1.0):
                    continue
                
                # u and v are Z-spiders connected by h_box, both with phase 0 or pi
                # Let's extract neighbor Z-spiders of u (excluding h_box)
                N_u = {}
                valid = True
                for n_id in list(u.neighbors):
                    if n_id == h_id:
                        continue
                    n_node = graph.vertices[n_id]
                    if n_node.type == 'H' and len(n_node.neighbors) == 2:
                        other = list(n_node.neighbors - {v1_id})[0]
                        if other in graph.vertices and graph.vertices[other].type == 'Z':
                            N_u[other] = n_id
                            continue
                    valid = False
                    break
                
                if not valid:
                    continue
                
                # Extract neighbor Z-spiders of v (excluding h_box)
                N_v = {}
                for n_id in list(v.neighbors):
                    if n_id == h_id:
                        continue
                    n_node = graph.vertices[n_id]
                    if n_node.type == 'H' and len(n_node.neighbors) == 2:
                        other = list(n_node.neighbors - {v2_id})[0]
                        if other in graph.vertices and graph.vertices[other].type == 'Z':
                            N_v[other] = n_id
                            continue
                    valid = False
                    break
                
                if not valid or (not N_u and not N_v):
                    continue
                
                # Perform pivot simplification:
                # 1. Toggle H-connections between N_u and N_v
                u_keys = list(N_u.keys())
                v_keys = list(N_v.keys())
                for x in u_keys:
                    for y in v_keys:
                        if x == y:
                            continue
                        # Check if connected via H-box
                        shared_h = None
                        for x_neigh_id in graph.vertices[x].neighbors:
                            if x_neigh_id in graph.vertices[y].neighbors:
                                if graph.vertices[x_neigh_id].type == 'H':
                                    shared_h = x_neigh_id
                                    break
                        if shared_h is not None:
                            graph.remove_vertex(shared_h)
                        else:
                            h_new = graph.add_vertex('H')
                            graph.add_edge(x, h_new.id)
                            graph.add_edge(h_new.id, y)
                
                # 2. Update phases
                for x in u_keys:
                    if x in N_v:
                        # Intersection N_u and N_v
                        graph.vertices[x].phase = (graph.vertices[x].phase + u.phase + v.phase + 1.0) % 2.0
                    else:
                        graph.vertices[x].phase = (graph.vertices[x].phase + v.phase) % 2.0
                for y in v_keys:
                    if y not in N_u:
                        graph.vertices[y].phase = (graph.vertices[y].phase + u.phase) % 2.0
                
                # 3. Remove u, v, h_box, and the intermediate H-boxes connecting u and v to their neighbors
                for intermediate_h in list(N_u.values()) + list(N_v.values()):
                    graph.remove_vertex(intermediate_h)
                graph.remove_vertex(h_id)
                graph.remove_vertex(v1_id)
                graph.remove_vertex(v2_id)
                
                changed = True
                break
            if changed:
                break
        return changed
