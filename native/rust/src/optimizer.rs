use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashSet, HashMap};

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

pub struct RustNode {
    pub id: usize,
    pub node_type: String,
    pub gate_name: Option<String>,
    pub targets: Vec<String>,
    pub args: Vec<f64>,
    pub cbit_name: Option<String>,
    pub condition: Option<PyObject>,
    pub print_expr: Option<PyObject>,
    pub assert_cond: Option<PyObject>,
    pub children_ids: Vec<usize>,
    pub parents_ids: Vec<usize>,
}

impl RustNode {
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        RustNode {
            id: self.id,
            node_type: self.node_type.clone(),
            gate_name: self.gate_name.clone(),
            targets: self.targets.clone(),
            args: self.args.clone(),
            cbit_name: self.cbit_name.clone(),
            condition: self.condition.as_ref().map(|x| x.clone_ref(py)),
            print_expr: self.print_expr.as_ref().map(|x| x.clone_ref(py)),
            assert_cond: self.assert_cond.as_ref().map(|x| x.clone_ref(py)),
            children_ids: self.children_ids.clone(),
            parents_ids: self.parents_ids.clone(),
        }
    }
}

fn dict_to_node(dict: &Bound<'_, PyDict>) -> PyResult<RustNode> {
    let id: usize = dict.get_item("id")?.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing id"))?.extract()?;
    let node_type: String = dict.get_item("type")?.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing type"))?.extract()?;
    
    let gate_name: Option<String> = match dict.get_item("gate_name")? {
        Some(val) if !val.is_none() => Some(val.extract()?),
        _ => None,
    };
    
    let targets: Vec<String> = dict.get_item("targets")?.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing targets"))?.extract()?;
    let args: Vec<f64> = dict.get_item("args")?.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing args"))?.extract()?;
    
    let cbit_name: Option<String> = match dict.get_item("cbit_name")? {
        Some(val) if !val.is_none() => Some(val.extract()?),
        _ => None,
    };
    
    let condition: Option<PyObject> = match dict.get_item("condition")? {
        Some(val) if !val.is_none() => Some(val.unbind()),
        _ => None,
    };
    
    let print_expr: Option<PyObject> = match dict.get_item("print_expr")? {
        Some(val) if !val.is_none() => Some(val.unbind()),
        _ => None,
    };
    
    let assert_cond: Option<PyObject> = match dict.get_item("assert_cond")? {
        Some(val) if !val.is_none() => Some(val.unbind()),
        _ => None,
    };
    
    let children_ids: Vec<usize> = dict.get_item("children_ids")?.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing children_ids"))?.extract()?;
    
    Ok(RustNode {
        id,
        node_type,
        gate_name,
        targets,
        args,
        cbit_name,
        condition,
        print_expr,
        assert_cond,
        children_ids,
        parents_ids: Vec::new(),
    })
}

fn node_to_dict<'py>(node: &RustNode, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", node.id)?;
    dict.set_item("type", &node.node_type)?;
    dict.set_item("gate_name", &node.gate_name)?;
    dict.set_item("targets", &node.targets)?;
    dict.set_item("args", &node.args)?;
    dict.set_item("cbit_name", &node.cbit_name)?;
    dict.set_item("condition", &node.condition)?;
    dict.set_item("print_expr", &node.print_expr)?;
    dict.set_item("assert_cond", &node.assert_cond)?;
    dict.set_item("children_ids", &node.children_ids)?;
    Ok(dict)
}

fn conditions_equal(py: Python<'_>, cond1: &Option<PyObject>, cond2: &Option<PyObject>) -> bool {
    match (cond1, cond2) {
        (None, None) => true,
        (Some(c1), Some(c2)) => {
            if let Ok(res) = c1.bind(py).rich_compare(c2, pyo3::class::basic::CompareOp::Eq) {
                res.extract::<bool>().unwrap_or(false)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn remove_child(nodes: &mut HashMap<usize, RustNode>, parent_id: usize, child_id: usize) {
    if let Some(parent) = nodes.get_mut(&parent_id) {
        parent.children_ids.retain(|&x| x != child_id);
    }
    if let Some(child) = nodes.get_mut(&child_id) {
        child.parents_ids.retain(|&x| x != parent_id);
    }
}

fn add_child(nodes: &mut HashMap<usize, RustNode>, parent_id: usize, child_id: usize) {
    if let Some(parent) = nodes.get_mut(&parent_id) {
        if !parent.children_ids.contains(&child_id) {
            parent.children_ids.push(child_id);
        }
    }
    if let Some(child) = nodes.get_mut(&child_id) {
        if !child.parents_ids.contains(&parent_id) {
            child.parents_ids.push(parent_id);
        }
    }
}

fn cancel_nodes(nodes: &mut HashMap<usize, RustNode>, node1_id: usize, node2_id: usize) {
    let parents = if let Some(n1) = nodes.get(&node1_id) {
        n1.parents_ids.clone()
    } else {
        return;
    };
    let children = if let Some(n2) = nodes.get(&node2_id) {
        n2.children_ids.clone()
    } else {
        return;
    };
    
    // Remove links to node1 and node2
    for &p in &parents {
        if let Some(parent) = nodes.get_mut(&p) {
            parent.children_ids.retain(|&x| x != node1_id);
        }
    }
    for &c in &children {
        if let Some(child) = nodes.get_mut(&c) {
            child.parents_ids.retain(|&x| x != node2_id);
        }
    }
    
    // Connect parents directly to children
    for &p in &parents {
        for &c in &children {
            add_child(nodes, p, c);
        }
    }
    
    nodes.remove(&node1_id);
    nodes.remove(&node2_id);
}

fn bypass_node(nodes: &mut HashMap<usize, RustNode>, node_id: usize) {
    let parents = if let Some(n) = nodes.get(&node_id) {
        n.parents_ids.clone()
    } else {
        return;
    };
    let children = if let Some(n) = nodes.get(&node_id) {
        n.children_ids.clone()
    } else {
        return;
    };
    
    for &p in &parents {
        if let Some(parent) = nodes.get_mut(&p) {
            parent.children_ids.retain(|&x| x != node_id);
        }
    }
    for &c in &children {
        if let Some(child) = nodes.get_mut(&c) {
            child.parents_ids.retain(|&x| x != node_id);
        }
    }
    
    for &p in &parents {
        for &c in &children {
            add_child(nodes, p, c);
        }
    }
    
    nodes.remove(&node_id);
}

#[pyfunction]
pub fn optimize_eqir_native<'py>(py: Python<'py>, graph_dict: Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let next_node_id: usize = graph_dict.get_item("next_node_id")?.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing next_node_id"))?.extract()?;
    let nodes_list: Bound<'_, PyList> = graph_dict.get_item("nodes")?.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing nodes"))?.downcast::<PyList>()?.clone();
    
    let mut nodes: Vec<RustNode> = Vec::new();
    for item in nodes_list.iter() {
        let dict = item.downcast::<PyDict>()?;
        nodes.push(dict_to_node(&dict)?);
    }
    
    // Reconstruct nodes map and parent ids
    let mut nodes_map: HashMap<usize, RustNode> = HashMap::new();
    for node in nodes {
        nodes_map.insert(node.id, node);
    }
    let mut parents_updates: HashMap<usize, Vec<usize>> = HashMap::new();
    for (&id, node) in &nodes_map {
        for &child_id in &node.children_ids {
            parents_updates.entry(child_id).or_default().push(id);
        }
    }
    for (child_id, parents) in parents_updates {
        if let Some(child_node) = nodes_map.get_mut(&child_id) {
            child_node.parents_ids = parents;
        }
    }
    
    let mut worklist: HashSet<usize> = nodes_map.keys().cloned().collect();
    let max_iterations = nodes_map.len() * 5 + 1000;
    let mut iterations = 0;
    let mut optimizations_count = 0;
    
    let self_inverse_gates: HashSet<&str> = ["H", "X", "Y", "Z"].iter().cloned().collect();
    let rotation_gates: HashSet<&str> = ["RX", "RY", "RZ"].iter().cloned().collect();
    
    while !worklist.is_empty() && iterations < max_iterations {
        // Audit §2.3 (determinism): the prior code used
        // `*worklist.iter().next().unwrap()`, which is non-deterministic
        // because HashSet iteration order depends on the RandomState seed.
        // Different pop orderings can yield different sets of surviving
        // nodes (early rewrites change the graph), making the canonical hash
        // of identical inputs vary across runs. Use the smallest id first,
        // matching the Python optimizer's deterministic pop.
        let node_id = {
            let id = *worklist.iter().min().unwrap();
            worklist.remove(&id);
            id
        };
        
        if !nodes_map.contains_key(&node_id) {
            continue;
        }
        
        let node = nodes_map.get(&node_id).unwrap().clone_ref(py);
        if node.node_type != "GATE" {
            continue;
        }
        
        iterations += 1;
        
        let gate_name = match &node.gate_name {
            Some(g) => g.as_str(),
            None => continue,
        };
        
        // Rule 1: Self-inverse cancellation
        if self_inverse_gates.contains(gate_name) {
            let target_qubit = &node.targets[0];
            let mut next_node_id = None;
            for &child_id in &node.children_ids {
                if let Some(child) = nodes_map.get(&child_id) {
                    if child.targets.contains(target_qubit) {
                        next_node_id = Some(child_id);
                        break;
                    }
                }
            }
            if let Some(next_id) = next_node_id {
                let next_node = nodes_map.get(&next_id).unwrap().clone_ref(py);
                if next_node.node_type == "GATE" 
                   && next_node.gate_name.as_deref() == Some(gate_name)
                   && next_node.targets[0] == *target_qubit
                   && conditions_equal(py, &node.condition, &next_node.condition) {
                       
                       let mut affected: HashSet<usize> = node.parents_ids.iter().cloned().collect();
                       affected.extend(next_node.children_ids.iter().cloned());
                       
                       cancel_nodes(&mut nodes_map, node_id, next_id);
                       worklist.extend(affected);
                       optimizations_count += 1;
                       continue;
                }
            }
        }
        
        // Rule 2: Rotation merging
        if rotation_gates.contains(gate_name) {
            let target_qubit = &node.targets[0];
            let mut next_node_id = None;
            for &child_id in &node.children_ids {
                if let Some(child) = nodes_map.get(&child_id) {
                    if child.targets.contains(target_qubit) {
                        next_node_id = Some(child_id);
                        break;
                    }
                }
            }
            if let Some(next_id) = next_node_id {
                let next_node = nodes_map.get(&next_id).unwrap().clone_ref(py);
                if next_node.node_type == "GATE"
                   && next_node.gate_name.as_deref() == Some(gate_name)
                   && next_node.targets[0] == *target_qubit
                   && conditions_equal(py, &node.condition, &next_node.condition) {
                       
                       let angle1 = node.args[0];
                       let angle2 = next_node.args[0];
                       let new_angle = (angle1 + angle2) % (2.0 * std::f64::consts::PI);
                       
                       // Update node's angle
                       if let Some(n) = nodes_map.get_mut(&node_id) {
                           n.args[0] = new_angle;
                       }
                       
                       let mut affected: HashSet<usize> = next_node.parents_ids.iter().cloned().collect();
                       affected.extend(next_node.children_ids.iter().cloned());
                       affected.insert(node_id);
                       
                       bypass_node(&mut nodes_map, next_id);
                       worklist.extend(affected);
                       optimizations_count += 1;
                       continue;
                }
            }
        }
        
        // Rule 3: Dead gate elimination
        if rotation_gates.contains(gate_name) && !node.args.is_empty() && node.args[0].abs() < 1e-9 {
            let mut affected: HashSet<usize> = node.parents_ids.iter().cloned().collect();
            affected.extend(node.children_ids.iter().cloned());
            
            bypass_node(&mut nodes_map, node_id);
            worklist.extend(affected);
            optimizations_count += 1;
            continue;
        }
        
        // Rule 4: Peephole optimizations (H -> X/Z -> H)
        if gate_name == "H" {
            let q = &node.targets[0];
            let mut n2_id = None;
            for &child_id in &node.children_ids {
                if let Some(child) = nodes_map.get(&child_id) {
                    if !child.targets.is_empty() && child.targets[0] == *q {
                        n2_id = Some(child_id);
                        break;
                    }
                }
            }
            if let Some(id2) = n2_id {
                let n2 = nodes_map.get(&id2).unwrap().clone_ref(py);
                if n2.node_type == "GATE" && (n2.gate_name.as_deref() == Some("X") || n2.gate_name.as_deref() == Some("Z")) {
                    let mut n3_id = None;
                    for &child_id in &n2.children_ids {
                        if let Some(child) = nodes_map.get(&child_id) {
                            if !child.targets.is_empty() && child.targets[0] == *q {
                                n3_id = Some(child_id);
                                break;
                            }
                        }
                    }
                    if let Some(id3) = n3_id {
                        let n3 = nodes_map.get(&id3).unwrap().clone_ref(py);
                        if n3.node_type == "GATE" && n3.gate_name.as_deref() == Some("H") {
                            let target_gate = if n2.gate_name.as_deref() == Some("X") { "Z" } else { "X" };
                            if let Some(n) = nodes_map.get_mut(&id2) {
                                n.gate_name = Some(target_gate.to_string());
                            }
                            
                            let mut affected: HashSet<usize> = node.parents_ids.iter().cloned().collect();
                            affected.extend(n3.children_ids.iter().cloned());
                            affected.insert(id2);
                            
                            bypass_node(&mut nodes_map, node_id);
                            bypass_node(&mut nodes_map, id3);
                            worklist.extend(affected);
                            optimizations_count += 1;
                            continue;
                        }
                    }
                }
            }
        }
        
        // Rule 5: Peephole optimizations (S -> S -> Z and T -> T -> S)
        if gate_name == "S" || gate_name == "T" {
            let q = &node.targets[0];
            let mut n2_id = None;
            for &child_id in &node.children_ids {
                if let Some(child) = nodes_map.get(&child_id) {
                    if !child.targets.is_empty() && child.targets[0] == *q {
                        n2_id = Some(child_id);
                        break;
                    }
                }
            }
            if let Some(id2) = n2_id {
                let n2 = nodes_map.get(&id2).unwrap().clone_ref(py);
                if n2.node_type == "GATE" && n2.gate_name.as_deref() == Some(gate_name) {
                    let target_gate = if gate_name == "S" { "Z" } else { "S" };
                    if let Some(n) = nodes_map.get_mut(&id2) {
                        n.gate_name = Some(target_gate.to_string());
                    }
                    
                    let mut affected: HashSet<usize> = node.parents_ids.iter().cloned().collect();
                    affected.extend(n2.children_ids.iter().cloned());
                    affected.insert(id2);
                    
                    bypass_node(&mut nodes_map, node_id);
                    worklist.extend(affected);
                    optimizations_count += 1;
                    continue;
                }
            }
        }
        
        // Rule 6: Commutation cancellation (Case 1: Z q0 -> CNOT q0, q1 -> Z q0)
        if gate_name == "Z" {
            let q0 = &node.targets[0];
            let mut n2_id = None;
            for &child_id in &node.children_ids {
                if let Some(child) = nodes_map.get(&child_id) {
                    if !child.targets.is_empty() && child.targets[0] == *q0 {
                        n2_id = Some(child_id);
                        break;
                    }
                }
            }
            if let Some(id2) = n2_id {
                let n2 = nodes_map.get(&id2).unwrap().clone_ref(py);
                if n2.node_type == "GATE" && n2.gate_name.as_deref() == Some("CNOT") && n2.targets[0] == *q0 {
                    let mut n3_id = None;
                    for &child_id in &n2.children_ids {
                        if let Some(child) = nodes_map.get(&child_id) {
                            if !child.targets.is_empty() && child.targets[0] == *q0 {
                                n3_id = Some(child_id);
                                break;
                            }
                        }
                    }
                    if let Some(id3) = n3_id {
                        let n3 = nodes_map.get(&id3).unwrap().clone_ref(py);
                        if n3.node_type == "GATE" && n3.gate_name.as_deref() == Some("Z") && n3.targets[0] == *q0 {
                            let mut affected: HashSet<usize> = node.parents_ids.iter().cloned().collect();
                            affected.extend(n3.children_ids.iter().cloned());
                            affected.insert(id2);
                            
                            bypass_node(&mut nodes_map, node_id);
                            bypass_node(&mut nodes_map, id3);
                            worklist.extend(affected);
                            optimizations_count += 1;
                            continue;
                        }
                    }
                }
            }
        }
        
        // Rule 7: Commutation cancellation (Case 2: X q1 -> CNOT q0, q1 -> X q1)
        if gate_name == "X" {
            let q1 = &node.targets[0];
            let mut n2_id = None;
            for &child_id in &node.children_ids {
                if let Some(child) = nodes_map.get(&child_id) {
                    if child.targets.len() > 1 && child.targets[1] == *q1 {
                        n2_id = Some(child_id);
                        break;
                    }
                }
            }
            if let Some(id2) = n2_id {
                let n2 = nodes_map.get(&id2).unwrap().clone_ref(py);
                if n2.node_type == "GATE" && n2.gate_name.as_deref() == Some("CNOT") && n2.targets[1] == *q1 {
                    let mut n3_id = None;
                    for &child_id in &n2.children_ids {
                        if let Some(child) = nodes_map.get(&child_id) {
                            if !child.targets.is_empty() && child.targets[0] == *q1 {
                                n3_id = Some(child_id);
                                break;
                            }
                        }
                    }
                    if let Some(id3) = n3_id {
                        let n3 = nodes_map.get(&id3).unwrap().clone_ref(py);
                        if n3.node_type == "GATE" && n3.gate_name.as_deref() == Some("X") && n3.targets[0] == *q1 {
                            let mut affected: HashSet<usize> = node.parents_ids.iter().cloned().collect();
                            affected.extend(n3.children_ids.iter().cloned());
                            affected.insert(id2);
                            
                            bypass_node(&mut nodes_map, node_id);
                            bypass_node(&mut nodes_map, id3);
                            worklist.extend(affected);
                            optimizations_count += 1;
                            continue;
                        }
                    }
                }
            }
        }
        
        // Rule 8: CNOT cancellation
        if gate_name == "CNOT" {
            let ctrl = &node.targets[0];
            let target = &node.targets[1];
            let mut next_node_id = None;
            for &child_id in &node.children_ids {
                if let Some(child) = nodes_map.get(&child_id) {
                    if child.targets.contains(ctrl) || child.targets.contains(target) {
                        next_node_id = Some(child_id);
                        break;
                    }
                }
            }
            if let Some(next_id) = next_node_id {
                let next_node = nodes_map.get(&next_id).unwrap().clone_ref(py);
                if next_node.node_type == "GATE"
                   && next_node.gate_name.as_deref() == Some("CNOT")
                   && next_node.targets.len() == 2
                   && next_node.targets[0] == *ctrl
                   && next_node.targets[1] == *target
                   && conditions_equal(py, &node.condition, &next_node.condition) {
                       
                       let mut affected: HashSet<usize> = node.parents_ids.iter().cloned().collect();
                       affected.extend(next_node.children_ids.iter().cloned());
                       
                       cancel_nodes(&mut nodes_map, node_id, next_id);
                       worklist.extend(affected);
                       optimizations_count += 1;
                       continue;
                }
            }
        }
    }
    
    // Construct return dictionary
    let ret_dict = PyDict::new_bound(py);
    ret_dict.set_item("next_node_id", next_node_id)?;
    ret_dict.set_item("iterations_count", iterations)?;
    ret_dict.set_item("optimizations_count", optimizations_count)?;
    
    let ret_nodes = PyList::empty_bound(py);
    // Audit §2.3 (determinism): sort nodes by id before serializing so the
    // returned list order is byte-identical across runs regardless of
    // HashMap/HashSet hash-seed sensitivity.
    let mut sorted_ids: Vec<usize> = nodes_map.keys().cloned().collect();
    sorted_ids.sort();
    for id in sorted_ids {
        let node = nodes_map.remove(&id).unwrap();
        ret_nodes.append(node_to_dict(&node, py)?)?;
    }
    ret_dict.set_item("nodes", ret_nodes)?;
    
    Ok(ret_dict)
}
