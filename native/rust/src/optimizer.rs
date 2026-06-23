use pyo3::prelude::*;
use std::collections::HashSet;

#[pyfunction]
pub fn fast_unused_vars(
    defined: Vec<String>,
    used: Vec<String>,
) -> PyResult<Vec<String>> {
    let used_set: HashSet<String> = used.into_iter().collect();
    let unused: Vec<String> = defined
        .into_iter()
        .filter(|v| !used_set.contains(v))
        .collect();
    Ok(unused)
}
