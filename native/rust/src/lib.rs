use pyo3::prelude::*;

mod simulator;
mod zx;
mod routing;
mod optimizer;
mod runtime;

#[pymodule]
fn eigen_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulator::apply_h, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::apply_x, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::apply_y, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::apply_z, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::apply_s, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::apply_t, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::apply_cnot, m)?)?;
    m.add_function(wrap_pyfunction!(zx::fast_spider_fusion, m)?)?;
    m.add_function(wrap_pyfunction!(routing::fast_shortest_path, m)?)?;
    m.add_function(wrap_pyfunction!(optimizer::fast_unused_vars, m)?)?;
    m.add_function(wrap_pyfunction!(runtime::execute_bytecode_native, m)?)?;
    Ok(())
}
