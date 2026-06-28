use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
pub fn execute_bytecode_native(
    instructions: Vec<(String, Option<PyObject>)>,
    mut variables: HashMap<String, PyObject>
) -> PyResult<(HashMap<String, PyObject>, Vec<PyObject>)> {
    let mut stack: Vec<PyObject> = Vec::new();
    let mut ip = 0;
    
    Python::with_gil(|py| {
        while ip < instructions.len() {
            let (ref opcode, ref arg) = instructions[ip];
            match opcode.as_str() {
                "LOAD_CONST" => {
                    if let Some(val) = arg {
                        stack.push(val.clone_ref(py));
                    }
                }
                "STORE_VAR" => {
                    if let Some(var_name_obj) = arg {
                        let var_name: String = var_name_obj.extract(py)?;
                        if let Some(val) = stack.pop() {
                            variables.insert(var_name, val);
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"));
                        }
                    }
                }
                "LOAD_VAR" => {
                    if let Some(var_name_obj) = arg {
                        let var_name: String = var_name_obj.extract(py)?;
                        if let Some(val) = variables.get(&var_name) {
                            stack.push(val.clone_ref(py));
                        } else {
                            stack.push(var_name_obj.clone_ref(py));
                        }
                    }
                }
                "ADD" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).call_method1("__add__", (v2,))?;
                    stack.push(res.into());
                }
                "SUB" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).call_method1("__sub__", (v2,))?;
                    stack.push(res.into());
                }
                "MUL" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).call_method1("__mul__", (v2,))?;
                    stack.push(res.into());
                }
                "DIV" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let is_zero: bool = v2.bind(py).rich_compare(0, pyo3::class::basic::CompareOp::Eq)?.extract()?;
                    if is_zero {
                        return Err(pyo3::exceptions::PyZeroDivisionError::new_err("DivisionByZeroError: Division by zero."));
                    }
                    let res = v1.bind(py).call_method1("__truediv__", (v2,))?;
                    stack.push(res.into());
                }
                "EQ" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).rich_compare(v2, pyo3::class::basic::CompareOp::Eq)?;
                    stack.push(res.into());
                }
                "NEQ" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).rich_compare(v2, pyo3::class::basic::CompareOp::Ne)?;
                    stack.push(res.into());
                }
                "JMP" => {
                    if let Some(target_obj) = arg {
                        let target: usize = target_obj.extract(py)?;
                        ip = target;
                        continue;
                    }
                }
                "JMP_IF_FALSE" => {
                    if let Some(target_obj) = arg {
                        let target: usize = target_obj.extract(py)?;
                        let cond_obj = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                        let cond: bool = cond_obj.extract(py)?;
                        if !cond {
                            ip = target;
                            continue;
                        }
                    }
                }
                "PRINT" => {
                    let val = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let val_str: String = val.bind(py).str()?.extract()?;
                    println!("[PRINT DIRECTIVE] {}", val_str);
                }
                "HALT" => {
                    break;
                }
                _ => {}
            }
            ip += 1;
        }
        Ok(())
    })?;
    Ok((variables, stack))
}
