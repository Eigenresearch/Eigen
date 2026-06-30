#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

mod simulator;
#[cfg(feature = "pyo3")]
mod zx;
#[cfg(feature = "pyo3")]
mod routing;
#[cfg(feature = "pyo3")]
mod optimizer;
#[cfg(feature = "pyo3")]
mod runtime;
pub mod qrt;
pub mod frontend;

#[cfg(feature = "pyo3")]
#[pyfunction]
fn parse_native(py: Python, source: &str) -> PyResult<PyObject> {
    crate::frontend::py_ast::parse_native(py, source)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn type_check_source(source: &str, workspace_root: &str) -> PyResult<()> {
    let mut lexer = crate::frontend::lexer::Lexer::new(source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => return Err(pyo3::exceptions::PySyntaxError::new_err(e)),
    };
    let mut parser = crate::frontend::parser::Parser::new(tokens);
    let root_id = match parser.parse() {
        Ok(id) => id,
        Err(e) => return Err(pyo3::exceptions::PySyntaxError::new_err(e)),
    };
    // 1. Resolve and merge imports
    crate::frontend::type_checker::resolve_imports(root_id, workspace_root, &mut parser.ast)
        .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e))?;
    // 2. Perform type check
    let mut tc = crate::frontend::type_checker::RustTypeChecker::new(&parser.ast);
    tc.check(root_id).map_err(|e| pyo3::exceptions::PyTypeError::new_err(e))?;
    Ok(())
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn eigen_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<simulator::RustStatevector>()?;
    m.add_class::<simulator::RustSparseSimulator>()?;
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
    m.add_function(wrap_pyfunction!(optimizer::optimize_eqir_native, m)?)?;
    m.add_function(wrap_pyfunction!(runtime::execute_bytecode_native, m)?)?;
    m.add_function(wrap_pyfunction!(parse_native, m)?)?;
    m.add_function(wrap_pyfunction!(type_check_source, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::compute_svd_native, m)?)?;
    Ok(())
}
