// Quantum routing helpers.
//
// Audit §1.2 ("port ZX-equivalence & routing scoring to Rust")
// demanded the Rust translation unit contain the *actual* routing
// algorithm, not just the 46-line `fast_shortest_path` stub. The new
// `fast_sabre_swap_score` routine ports the SABRE swap-scoring inner
// loop from `src/routing/router.py` (lines 518-554 in `SabreRouter.route`)
// — the hot kernel that evaluates every candidate SWAP against the
// current front + extended layers and returns the lowest-cost one.
//
// The Python scoring is:
//     for each candidate swap (q1, q2) on the coupling map:
//         # apply swap on a trial mapping
//         front_score = avg over front_layer of dist(p0, p1)
//         ext_score   = avg over extended_layer[:5] of dist(p0, p1)
//         score = front_score + lookahead_weight * ext_score
//     pick (q1, q2) with smallest score
//
// The Rust port preserves that arithmetic exactly. It also adds
// deterministic tie-breaking: when two candidate SWAPs have identical
// scores, the lexicographically smallest (q1, q2) pair is chosen, so the
// routing output is byte-stable regardless of input edge order. Input
// edges are scanned in their given order; ties broken by (q1, q2).

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// Legacy BFS shortest-path helper retained for backwards
/// compatibility with `CouplingMap.shortest_path` (`src/routing/router.py`).
/// Behaviour unchanged.
#[pyfunction]
pub fn fast_shortest_path(
    edges: Vec<(usize, usize)>,
    src: usize,
    dst: usize,
) -> PyResult<Vec<usize>> {
    if src == dst {
        return Ok(vec![src]);
    }

    // Build adjacency list.
    let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (u, v) in edges {
        adj.entry(u).or_insert_with(HashSet::new).insert(v);
        adj.entry(v).or_insert_with(HashSet::new).insert(u);
    }

    let mut visited: HashSet<usize> = HashSet::new();
    visited.insert(src);

    let mut queue: VecDeque<(usize, Vec<usize>)> = VecDeque::new();
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

    Ok(vec![]) // No path.
}

/// SABRE swap-scoring inner loop, ported from
/// `src/routing/router.py::SabreRouter.route` lines 518-554.
///
/// Inputs (all parallel arrays of plain Python ints/floats):
///   `edges`            — coupling-map edges as `List[(q1, q2)]` of
///                        physical qubit indices.
///   `distances`        — precomputed `num_physical × num_physical`
///                        integer distance matrix.
///   `mapping`          — `List[physical]` indexed by logical qubit,
///                        i.e. `mapping[logical] = physical`. Length =
///                        number of logical qubits in the circuit.
///   `front_layer`      — `List[(l0, l1)]` of 2-qubit op pairs still
///                        waiting to be routed (logical indices).
///   `extended_layer`   — `List[(l0, l1)]` of 2-qubit ops used for
///                        lookahead; only the first 5 are used.
///   `lookahead_weight` — scalar blending front and extended scores.
///
/// Returns `None` if `edges` is empty (no candidate SWAPs exist).
/// Otherwise returns `Some((q1, q2, score))` — the best SWAP and its
/// score, with ties broken lexicographically by `(q1, q2)` so the
/// caller's routing output is deterministic w.r.t. input ordering.
#[pyfunction]
pub fn fast_sabre_swap_score(
    edges: Vec<(usize, usize)>,
    distances: Vec<Vec<usize>>,
    mapping: Vec<usize>,
    front_layer: Vec<(usize, usize)>,
    extended_layer: Vec<(usize, usize)>,
    lookahead_weight: f64,
) -> PyResult<Option<(usize, usize, f64)>> {
    if edges.is_empty() {
        return Ok(None);
    }

    // Build reverse mapping (physical -> logical) once per call. If a
    // physical qubit is not currently in the mapping, it has no logical
    // assignment and the SWAP simply moves it without affecting any
    // logical→physical binding.
    let mut reverse_mapping: HashMap<usize, usize> = HashMap::new();
    for (logical, &physical) in mapping.iter().enumerate() {
        reverse_mapping.insert(physical, logical);
    }

    let mut best: Option<(usize, usize, f64)> = None;

    // Iterate candidate SWAPs in input order; for each, build a trial
    // mapping, compute the score, and keep the lowest one (with
    // lexicographic tie-break on (q1, q2)).
    for &(q1, q2) in &edges {
        if q1 == q2 {
            continue;
        }
        let lq_a_opt = reverse_mapping.get(&q1).cloned();
        let lq_b_opt = reverse_mapping.get(&q2).cloned();

        // Build the trial mapping from the current one. We clone
        // eagerly — `mapping.len()` is the number of logical qubits,
        // typically a few dozen. The clone is amortized by the work
        // in the inner loop below.
        let mut trial = mapping.clone();
        if let Some(lq_a) = lq_a_opt {
            if lq_a < trial.len() {
                trial[lq_a] = q2;
            }
        }
        if let Some(lq_b) = lq_b_opt {
            if lq_b < trial.len() {
                trial[lq_b] = q1;
            }
        }

        // Front-layer score: average pairwise distance over the front
        // layer (only counts valid pairs — pairs whose logical qubits
        // are within the mapping's range, and whose physical indices
        // fall within the distance matrix).
        let mut front_score_sum = 0.0_f64;
        let mut front_count = 0_usize;
        for &(l0, l1) in &front_layer {
            if l0 < trial.len() && l1 < trial.len() {
                let p0 = trial[l0];
                let p1 = trial[l1];
                if p0 < distances.len() && p1 < distances.len() {
                    front_score_sum += distances[p0][p1] as f64;
                    front_count += 1;
                }
            }
        }
        let front_avg = if front_count > 0 {
            front_score_sum / (front_count as f64)
        } else {
            0.0
        };

        // Extended-layer score: average over the first 5 entries of
        // `extended_layer`. Matches the Python slice
        // `extended_layer[:5]`.
        let mut ext_score_sum = 0.0_f64;
        let mut ext_count = 0_usize;
        for &(l0, l1) in extended_layer.iter().take(5) {
            if l0 < trial.len() && l1 < trial.len() {
                let p0 = trial[l0];
                let p1 = trial[l1];
                if p0 < distances.len() && p1 < distances.len() {
                    ext_score_sum += distances[p0][p1] as f64;
                    ext_count += 1;
                }
            }
        }
        let ext_avg = if ext_count > 0 {
            ext_score_sum / (ext_count as f64)
        } else {
            0.0
        };

        let score = front_avg + lookahead_weight * ext_avg;

        match best {
            None => {
                best = Some((q1, q2, score));
            }
            Some((bq1, bq2, bs)) => {
                let strictly_better = score < bs;
                let tie_and_smaller_pair = (score == bs) && ((q1, q2) < (bq1, bq2));
                if strictly_better || tie_and_smaller_pair {
                    best = Some((q1, q2, score));
                }
            }
        }
    }

    Ok(best)
}
