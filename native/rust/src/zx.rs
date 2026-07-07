// ZX-calculus spider fusion.
//
// Audit §1.2 ("port ZX-equivalence & routing scoring to Rust") required
// the Rust translation unit to *actually contain* the algorithm rather
// than the previous 16-line stub that only deduplicated edges. The
// `fast_spider_fusion_full` function below ports the same-colour
// spider fusion rule as implemented by `src.zx.spider_fusion.SpiderFuser.fuse_spiders`
// (33 lines in Python) and the per-pass fusion performed by
// `src.zx.zx_equivalence.ZXEquivalenceChecker.simplify` (lines 389-407).
//
// The fusion rule (for non-Hadamard edges only, matching the Python
// implementation):
//   For each pair of adjacent same-colour spiders (Z-Z or X-X) connected
//   by a non-Hadamard edge, fuse them by adding their phase parameters
//   (mod 2π, expressed in the code as mod 2.0) and merging the absorbed
//   spider's neighbour set into the surviving one. Self-loops are
//   removed; hadamard edges are not considered (matching the existing
//   Python behaviour and the data model in `src/zx/zx_graph.py`).
//
// Determinism: vertex iteration is in the order given by the caller
// (`vertex_ids`); for ties we prefer the lexicographically smallest
// surviving id. Surviving vertex ids are emitted in sorted order so
// the result is byte-identical regardless of the iteration history.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// Legacy edge-deduplication helper retained for backwards
/// compatibility — the existing `ZXEquivalenceChecker.simplify` call
/// site (src/zx/zx_equivalence.py:379) calls this routine to clean
/// up bidirectional edge listings before the Python fuse pass runs.
/// Behaviour is unchanged: emit a HashSet of `(min, max)` pairs of
/// distinct, non-self edges.
#[pyfunction]
pub fn fast_spider_fusion(edges: Vec<(usize, usize)>) -> PyResult<Vec<(usize, usize)>> {
    let mut unique_edges = HashSet::new();
    for (u, v) in edges {
        let min_val = std::cmp::min(u, v);
        let max_val = std::cmp::max(u, v);
        if min_val != max_val {
            unique_edges.insert((min_val, max_val));
        }
    }
    Ok(unique_edges.into_iter().collect())
}

/// Full spider-fusion port. Accepts the raw ZX-graph state in
/// parallel-array form (so no Python-object marshalling is needed)
/// and returns the merged graph in the same format.
///
/// Inputs:
///   `vertex_ids`   — list of original integer vertex ids (one per
///                     vertex; may be sparse w.r.t. their integer
///                     values).
///   `vertex_types` — parallel list of single-byte codes:
///                     `b'Z'`, `b'X'`, `b'H'`, `b'B'` (Boundary).
///                     Only `Z` and `X` participate in fusion.
///   `phases`       — parallel list of phases expressed in multiples
///                     of π, in the half-open interval [0, 2).
///   `adjacency`     — parallel list where `adjacency[i]` is the list
///                     of vertex *ids* (NOT indices) connected to
///                     `vertex_ids[i]` through non-Hadamard edges.
///
/// Returns `(new_ids, new_types, new_phases, new_adjacency, changed)`:
///   `new_ids`       — surviving vertex IDs, sorted ascending.
///   `new_types`     — parallel type bytes for survivors.
///   `new_phases`    — parallel phases for survivors, normalised mod 2.
///   `new_adjacency` — parallel non-H adjacency, using original IDs.
///   `changed`       — `true` iff at least one fusion happened.
#[pyfunction]
#[allow(clippy::needless_range_loop)]
pub fn fast_spider_fusion_full(
    vertex_ids: Vec<usize>,
    vertex_types: Vec<u8>,
    phases: Vec<f64>,
    adjacency: Vec<Vec<usize>>,
) -> PyResult<(Vec<usize>, Vec<u8>, Vec<f64>, Vec<Vec<usize>>, bool)> {
    // Validate parallel-array invariants.
    let n = vertex_ids.len();
    if vertex_types.len() != n || phases.len() != n || adjacency.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fast_spider_fusion_full: vertex_ids, vertex_types, phases and adjacency must all have the same length",
        ));
    }

    let mut phase: Vec<f64> = phases.clone();
    let mut adj: Vec<HashSet<usize>> = adjacency
        .iter()
        .map(|ns| ns.iter().cloned().collect())
        .collect();
    let mut active: Vec<bool> = vec![true; n];
    let mut v_id_to_idx: HashMap<usize, usize> = HashMap::with_capacity(n);
    for (i, &v) in vertex_ids.iter().enumerate() {
        v_id_to_idx.insert(v, i);
    }

    let mut changed = false;

    // Run a fixpoint fusion pass. The Python `simplify` re-enters its
    // outer `while True:` loop after each successful fusion; we mirror
    // that conservative behaviour by restarting the scan after each
    // successful fusion. This guarantees correctness (the absorbed
    // spider is removed before its neighbours are visited again) at
    // some cost in throughput — which is acceptable, since the typical
    // ZX-graph being fused is small (a few hundred vertices at most).
    loop {
        let mut made_change_this_iter = false;

        // Iterate vertices in the order given by the caller (which the
        // Python wrapper sorts by id for determinism). For each active
        // Z/X spider, scan its *snapshot* of neighbours for the first
        // same-colour active one and fuse.
        'outer: for i in 0..n {
            if !active[i] {
                continue;
            }
            let vt = vertex_types[i];
            if vt != b'Z' && vt != b'X' {
                continue;
            }

            // Snapshot the neighbour list to avoid mutation-during-iteration.
            let neighbour_ids: Vec<usize> = adj[i].iter().cloned().collect();
            for nid in &neighbour_ids {
                let j = match v_id_to_idx.get(nid) {
                    Some(&j) => j,
                    None => continue,
                };
                if !active[j] || vertex_types[j] != vt {
                    continue;
                }
                // Fuse j into i.
                phase[i] = (phase[i] + phase[j]).rem_euclid(2.0);

                // Move j's neighbours (other than i and j itself) to i.
                let j_neighbours: Vec<usize> = adj[j]
                    .iter()
                    .filter(|&x| *x != vertex_ids[i] && *x != vertex_ids[j])
                    .cloned()
                    .collect();
                for nn in &j_neighbours {
                    adj[i].insert(*nn);
                    if let Some(&nn_idx) = v_id_to_idx.get(nn) {
                        if active[nn_idx] {
                            adj[nn_idx].insert(vertex_ids[i]);
                            // Drop the absorbed vertex from nn's adjacency.
                            adj[nn_idx].remove(&vertex_ids[j]);
                        }
                    }
                }

                // Remove the i-j edge and any self-loop on i (defensive).
                adj[i].remove(&vertex_ids[j]);
                adj[i].remove(&vertex_ids[i]);

                // Mark j absorbed.
                active[j] = false;
                adj[j].clear();

                changed = true;
                made_change_this_iter = true;
                // Restart the outer scan — matches the Python `break`-out-of-inner-loop
                // behaviour used by `simplify`.
                break 'outer;
            }
        }

        if !made_change_this_iter {
            break;
        }
    }

    // Compaction: emit survivors sorted by id (deterministic output
    // order regardless of fusion iteration order). Adjacency uses the
    // ORIGINAL vertex IDs (not remapped) so the Python caller can
    // correlate them back to its existing `ZXGraph` objects.
    let mut surviving_ids: Vec<usize> = (0..n).filter(|&i| active[i]).map(|i| vertex_ids[i]).collect();
    surviving_ids.sort();

    let mut new_types: Vec<u8> = Vec::with_capacity(surviving_ids.len());
    let mut new_phases: Vec<f64> = Vec::with_capacity(surviving_ids.len());
    let mut new_adj: Vec<Vec<usize>> = Vec::with_capacity(surviving_ids.len());
    for &surv_id in &surviving_ids {
        let orig_idx = *v_id_to_idx.get(&surv_id).expect("surviving id must have an index");
        new_types.push(vertex_types[orig_idx]);
        new_phases.push(phase[orig_idx]);
        // Filter neighbours to surviving vertices, dedupe, and sort.
        let mut neighbours: Vec<usize> = adj[orig_idx]
            .iter()
            .filter(|n| {
                v_id_to_idx
                    .get(n)
                    .map(|&nidx| active[nidx])
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        neighbours.sort();
        neighbours.dedup();
        new_adj.push(neighbours);
    }

    Ok((surviving_ids, new_types, new_phases, new_adj, changed))
}
