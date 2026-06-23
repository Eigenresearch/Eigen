use pyo3::prelude::*;

#[pyfunction]
pub fn fast_spider_fusion(edges: Vec<(usize, usize)>) -> PyResult<Vec<(usize, usize)>> {
    // Rust-accelerated zx spider fusion helper:
    // Takes edges, performs fast deduplication or connectivity checks.
    let mut unique_edges = std::collections::HashSet::new();
    for (u, v) in edges {
        let min_val = std::cmp::min(u, v);
        let max_val = std::cmp::max(u, v);
        if min_val != max_val {
            unique_edges.insert((min_val, max_val));
        }
    }
    Ok(unique_edges.into_iter().collect())
}
