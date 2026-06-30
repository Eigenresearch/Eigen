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
                            let is_literal = {
                                let bytes = var_name.as_bytes();
                                if bytes.is_empty() {
                                    false
                                } else {
                                    let first = bytes[0];
                                    if first == b'q' || first == b'c' {
                                        bytes[1..].iter().all(|&b| b.is_ascii_digit() || b == b'_')
                                    } else {
                                        false
                                    }
                                }
                            };
                            if is_literal {
                                stack.push(var_name_obj.clone_ref(py));
                            } else {
                                let vm_mod = py.import_bound("src.backend.vm")?;
                                let err_cls = vm_mod.getattr("UndefinedVariableError")?;
                                return Err(PyErr::from_value_bound(err_cls.call1((format!("Variable '{}' is not defined.", var_name),))?));
                            }
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
                "LT" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).rich_compare(v2, pyo3::class::basic::CompareOp::Lt)?;
                    stack.push(res.into());
                }
                "GT" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).rich_compare(v2, pyo3::class::basic::CompareOp::Gt)?;
                    stack.push(res.into());
                }
                "LTE" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).rich_compare(v2, pyo3::class::basic::CompareOp::Le)?;
                    stack.push(res.into());
                }
                "GTE" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).rich_compare(v2, pyo3::class::basic::CompareOp::Ge)?;
                    stack.push(res.into());
                }
                "AND" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let b1: bool = v1.bind(py).is_truthy()?;
                    let b2: bool = v2.bind(py).is_truthy()?;
                    stack.push((b1 && b2).into_py(py));
                }
                "OR" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let b1: bool = v1.bind(py).is_truthy()?;
                    let b2: bool = v2.bind(py).is_truthy()?;
                    stack.push((b1 || b2).into_py(py));
                }
                "NOT" => {
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let b1: bool = v1.bind(py).is_truthy()?;
                    stack.push((!b1).into_py(py));
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
                "JMP_IF_TRUE" => {
                    if let Some(target_obj) = arg {
                        let target: usize = target_obj.extract(py)?;
                        let cond_obj = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                        let cond: bool = cond_obj.extract(py)?;
                        if cond {
                            ip = target;
                            continue;
                        }
                    }
                }
                "MOD" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let is_zero: bool = v2.bind(py).rich_compare(0, pyo3::class::basic::CompareOp::Eq)?.extract()?;
                    if is_zero {
                        return Err(pyo3::exceptions::PyZeroDivisionError::new_err("DivisionByZeroError: Division by zero."));
                    }
                    let res = v1.bind(py).call_method1("__mod__", (v2,))?;
                    stack.push(res.into());
                }
                "BIT_AND" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).call_method1("__and__", (v2,))?;
                    stack.push(res.into());
                }
                "BIT_OR" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).call_method1("__or__", (v2,))?;
                    stack.push(res.into());
                }
                "BIT_XOR" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).call_method1("__xor__", (v2,))?;
                    stack.push(res.into());
                }
                "BIT_NOT" => {
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).call_method0("__invert__")?;
                    stack.push(res.into());
                }
                "SHL" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).call_method1("__lshift__", (v2,))?;
                    stack.push(res.into());
                }
                "SHR" => {
                    let v2 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let v1 = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let res = v1.bind(py).call_method1("__rshift__", (v2,))?;
                    stack.push(res.into());
                }
                "LOAD_CONST_STORE" => {
                    if let Some(tuple_obj) = arg {
                        let tuple: (PyObject, String) = tuple_obj.extract(py)?;
                        variables.insert(tuple.1, tuple.0);
                    }
                }
                "LOAD_VAR_LOAD_CONST_ADD" => {
                    if let Some(tuple_obj) = arg {
                        let tuple: (String, PyObject) = tuple_obj.extract(py)?;
                        if let Some(val) = variables.get(&tuple.0) {
                            let res = val.bind(py).call_method1("__add__", (tuple.1,))?;
                            stack.push(res.into());
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err("UndefinedVariable"));
                        }
                    }
                }
                "LOAD_VAR_LOAD_CONST_SUB" => {
                    if let Some(tuple_obj) = arg {
                        let tuple: (String, PyObject) = tuple_obj.extract(py)?;
                        if let Some(val) = variables.get(&tuple.0) {
                            let res = val.bind(py).call_method1("__sub__", (tuple.1,))?;
                            stack.push(res.into());
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err("UndefinedVariable"));
                        }
                    }
                }
                "LOAD_VAR_LOAD_CONST_LT" => {
                    if let Some(tuple_obj) = arg {
                        let tuple: (String, PyObject) = tuple_obj.extract(py)?;
                        if let Some(val) = variables.get(&tuple.0) {
                            let res = val.bind(py).rich_compare(tuple.1, pyo3::class::basic::CompareOp::Lt)?;
                            stack.push(res.into());
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err("UndefinedVariable"));
                        }
                    }
                }
                "LOAD_VAR_LOAD_CONST_GT" => {
                    if let Some(tuple_obj) = arg {
                        let tuple: (String, PyObject) = tuple_obj.extract(py)?;
                        if let Some(val) = variables.get(&tuple.0) {
                            let res = val.bind(py).rich_compare(tuple.1, pyo3::class::basic::CompareOp::Gt)?;
                            stack.push(res.into());
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err("UndefinedVariable"));
                        }
                    }
                }
                "LOAD_VAR_LOAD_CONST_LTE" => {
                    if let Some(tuple_obj) = arg {
                        let tuple: (String, PyObject) = tuple_obj.extract(py)?;
                        if let Some(val) = variables.get(&tuple.0) {
                            let res = val.bind(py).rich_compare(tuple.1, pyo3::class::basic::CompareOp::Le)?;
                            stack.push(res.into());
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err("UndefinedVariable"));
                        }
                    }
                }
                "LOAD_VAR_LOAD_CONST_GTE" => {
                    if let Some(tuple_obj) = arg {
                        let tuple: (String, PyObject) = tuple_obj.extract(py)?;
                        if let Some(val) = variables.get(&tuple.0) {
                            let res = val.bind(py).rich_compare(tuple.1, pyo3::class::basic::CompareOp::Ge)?;
                            stack.push(res.into());
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err("UndefinedVariable"));
                        }
                    }
                }
                "PRINT" => {
                    let val = stack.pop().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("StackUnderflow"))?;
                    let val_str: String = val.bind(py).str()?.extract()?;
                    println!("{}", val_str);
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
