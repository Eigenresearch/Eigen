use pyo3::prelude::*;
use std::collections::{VecDeque, HashSet, HashMap};

#[pyfunction]
pub fn fast_shortest_path(
    edges: Vec<(usize, usize)>,
    src: usize,
    dst: usize,
) -> PyResult<Vec<usize>> {
    if src == dst {
        return Ok(vec![src]);
    }
    
    // Build adjacency list
    let mut adj = HashMap::new();
    for (u, v) in edges {
        adj.entry(u).or_insert_with(HashSet::new).insert(v);
        adj.entry(v).or_insert_with(HashSet::new).insert(u);
    }
    
    let mut visited = HashSet::new();
    visited.insert(src);
    
    let mut queue = VecDeque::new();
    queue.push_back((src, vec![src]));
    
    while let Some((current, path)) = queue.pop_front() {
        if let Some(neighbors) = adj.get(&current) {
            for &neighbor in neighbors {
                if neighbor == dst {
                    let mut final_path = path.clone();
                    final_path.push(neighbor);
                    return Ok(final_path);
                }
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let mut next_path = path.clone();
                    next_path.push(neighbor);
                    queue.push_back((neighbor, next_path));
                }
            }
        }
    }
    
    Ok(vec![]) // No path
}
