use crate::frontend::ast::*;
use std::collections::{HashMap, HashSet};

pub struct RustTypeChecker<'a> {
    pub ast: &'a AST,
    pub global_qfuncs: HashMap<String, NodeId>,
    pub global_funcs: HashMap<String, NodeId>,
    pub global_structs: HashMap<String, NodeId>,
    pub global_enums: HashMap<String, NodeId>,
    pub scopes: Vec<HashMap<String, String>>,
    pub current_function: Option<NodeId>,
    pub loop_depth: usize,
    pub type_cache: HashMap<NodeId, String>,
    pub errors: Vec<String>,
}

impl<'a> RustTypeChecker<'a> {
    pub fn new(ast: &'a AST) -> Self {
        Self {
            ast,
            global_qfuncs: HashMap::new(),
            global_funcs: HashMap::new(),
            global_structs: HashMap::new(),
            global_enums: HashMap::new(),
            scopes: vec![HashMap::new()],
            current_function: None,
            loop_depth: 0,
            type_cache: HashMap::new(),
            errors: Vec::new(),
        }
    }

    pub fn check(&mut self, root_id: NodeId) -> Result<(), String> {
        // Register globals
        let body = match &self.ast.nodes[root_id] {
            ASTNode::Program(p) => &p.body,
            _ => return Err("Root must be a ProgramNode".to_string()),
        };

        for &node_id in body {
            match &self.ast.nodes[node_id] {
                ASTNode::QFuncDecl(qf) => {
                    if self.global_qfuncs.contains_key(&qf.name) || self.global_funcs.contains_key(&qf.name) {
                        self.errors.push(format!("Duplicate declaration of function '{}'", qf.name));
                    }
                    self.global_qfuncs.insert(qf.name.clone(), node_id);
                }
                ASTNode::FuncDecl(f) => {
                    if self.global_funcs.contains_key(&f.name) || self.global_qfuncs.contains_key(&f.name) {
                        self.errors.push(format!("Duplicate declaration of function '{}'", f.name));
                    }
                    self.global_funcs.insert(f.name.clone(), node_id);
                }
                ASTNode::StructDecl(s) => {
                    if self.global_structs.contains_key(&s.name) {
                        self.errors.push(format!("Duplicate declaration of struct '{}'", s.name));
                    }
                    self.global_structs.insert(s.name.clone(), node_id);
                }
                ASTNode::EnumDecl(e) => {
                    if self.global_enums.contains_key(&e.name) {
                        self.errors.push(format!("Duplicate declaration of enum '{}'", e.name));
                    }
                    self.global_enums.insert(e.name.clone(), node_id);
                }
                _ => {}
            }
        }

        // Type check statements
        for &node_id in body {
            self.check_node(node_id);
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.join("\n"))
        }
    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    fn declare_var(&mut self, name: String, type_name: String) {
        if let Some(current_scope) = self.scopes.last_mut() {
            if current_scope.contains_key(&name) {
                self.errors.push(format!("Redeclaration of variable '{}' in the same scope", name));
            }
            current_scope.insert(name, type_name);
        }
    }

    fn lookup_var(&mut self, name: &str) -> String {
        for scope in self.scopes.iter().rev() {
            if let Some(t) = scope.get(name) {
                return t.clone();
            }
        }
        for (enum_name, enum_id) in &self.global_enums {
            if let ASTNode::EnumDecl(e) = &self.ast.nodes[*enum_id] {
                if e.variants.contains(&name.to_string()) {
                    return enum_name.clone();
                }
            }
        }
        self.errors.push(format!("Undeclared variable '{}'", name));
        "unknown".to_string()
    }

    fn types_compatible(&self, expected: &str, actual: &str) -> bool {
        if expected == actual {
            return true;
        }
        if expected == "any" || actual == "any" || expected == "unknown" || actual == "unknown" {
            return true;
        }
        if expected == "float" && actual == "int" {
            return true;
        }
        if (expected == "cbit" && actual == "int") || (expected == "int" && actual == "cbit") {
            return true;
        }
        if expected == "null" || actual == "null" {
            return true;
        }
        if expected.starts_with("array<") && actual.starts_with("array<") {
            let t_exp = &expected[6..expected.len() - 1];
            let t_act = &actual[6..actual.len() - 1];
            return self.types_compatible(t_exp, t_act);
        }
        if expected.starts_with("map<") && actual.starts_with("map<") {
            let parts_exp: Vec<&str> = expected[4..expected.len() - 1].split(',').collect();
            let parts_act: Vec<&str> = actual[4..actual.len() - 1].split(',').collect();
            if parts_exp.len() == 2 && parts_act.len() == 2 {
                return self.types_compatible(parts_exp[0].trim(), parts_act[0].trim())
                    && self.types_compatible(parts_exp[1].trim(), parts_act[1].trim());
            }
        }
        false
    }

    fn check_node(&mut self, node_id: NodeId) -> String {
        if let Some(t) = self.type_cache.get(&node_id) {
            return t.clone();
        }
        let t = self.check_node_uncached(node_id);
        if t != "void" {
            self.type_cache.insert(node_id, t.clone());
        }
        t
    }

    fn check_node_uncached(&mut self, node_id: NodeId) -> String {
        match &self.ast.nodes[node_id] {
            ASTNode::Program(p) => {
                for &stmt in &p.body {
                    self.check_node(stmt);
                }
                "void".to_string()
            }
            ASTNode::QFuncDecl(qf) => {
                self.enter_scope();
                for (p_name, p_type) in &qf.params {
                    self.declare_var(p_name.clone(), p_type.clone());
                }
                for &stmt in &qf.body {
                    self.check_node(stmt);
                }
                self.exit_scope();
                "void".to_string()
            }
            ASTNode::FuncDecl(f) => {
                self.enter_scope();
                let prev_func = self.current_function;
                self.current_function = Some(node_id);
                for (p_name, p_type) in &f.params {
                    self.declare_var(p_name.clone(), p_type.clone());
                }
                for &stmt in &f.body {
                    self.check_node(stmt);
                }
                self.current_function = prev_func;
                self.exit_scope();
                "void".to_string()
            }
            ASTNode::Let(l) => {
                let val_type = self.check_node(l.value);
                if !self.types_compatible(&l.type_name, &val_type) {
                    self.errors.push(format!(
                        "Cannot assign expression of type '{}' to variable '{}' of type '{}'",
                        val_type, l.name, l.type_name
                    ));
                }
                self.declare_var(l.name.clone(), l.type_name.clone());
                "void".to_string()
            }
            ASTNode::VarDecl(v) => {
                self.declare_var(v.name.clone(), v.type_name.clone());
                "void".to_string()
            }
            ASTNode::Assignment(a) => {
                let target_type = self.check_node(a.target);
                let val_type = self.check_node(a.value);
                if a.op != "=" {
                    let numeric_types = ["int", "float"];
                    if !numeric_types.contains(&target_type.as_str()) || !numeric_types.contains(&val_type.as_str()) {
                        self.errors.push(format!("Operator {} not supported between types '{}' and '{}'", a.op, target_type, val_type));
                    }
                } else if !self.types_compatible(&target_type, &val_type) {
                    self.errors.push(format!("Cannot assign type '{}' to target of type '{}'", val_type, target_type));
                }
                "void".to_string()
            }
            ASTNode::Literal(l) => l.type_name.clone(),
            ASTNode::VarRef(v) => self.lookup_var(&v.name),
            ASTNode::BinaryOp(b) => {
                let left_type = self.check_node(b.left);
                let right_type = self.check_node(b.right);
                if ["==", "!=", "<", ">", "<=", ">="].contains(&b.op.as_str()) {
                    if !self.types_compatible(&left_type, &right_type) && !self.types_compatible(&right_type, &left_type) {
                        self.errors.push(format!("Comparison '{}' not supported between '{}' and '{}'", b.op, left_type, right_type));
                    }
                    "bool".to_string()
                } else if ["and", "or", "not"].contains(&b.op.as_str()) {
                    if left_type != "bool" || right_type != "bool" {
                        self.errors.push(format!("Logical operator '{}' expects boolean arguments", b.op));
                    }
                    "bool".to_string()
                } else if ["%", "&", "|", "^", "~", "<<", ">>"].contains(&b.op.as_str()) {
                    let integer_types = ["int", "cbit"];
                    if !integer_types.contains(&left_type.as_str()) || !integer_types.contains(&right_type.as_str()) {
                        self.errors.push(format!("Operator '{}' is only supported on integer types, got '{}' and '{}'", b.op, left_type, right_type));
                    }
                    "int".to_string()
                } else {
                    let numeric_types = ["int", "float", "cbit"];
                    if !numeric_types.contains(&left_type.as_str()) || !numeric_types.contains(&right_type.as_str()) {
                        self.errors.push(format!("Binary operation '{}' not supported between '{}' and '{}'", b.op, left_type, right_type));
                    }
                    if left_type == "float" || right_type == "float" {
                        "float".to_string()
                    } else {
                        "int".to_string()
                    }
                }
            }
            ASTNode::For(f) => {
                let iter_type = self.check_node(f.iterable);
                let elem_type = if iter_type.starts_with("array<") {
                    iter_type[6..iter_type.len() - 1].to_string()
                } else {
                    self.errors.push(format!("For loop expected array type, got '{}'", iter_type));
                    "unknown".to_string()
                };
                self.enter_scope();
                self.declare_var(f.variable.clone(), elem_type);
                self.loop_depth += 1;
                for &stmt in &f.body {
                    self.check_node(stmt);
                }
                self.loop_depth -= 1;
                self.exit_scope();
                "void".to_string()
            }
            ASTNode::While(w) => {
                let cond_type = self.check_node(w.condition);
                if cond_type != "bool" {
                    self.errors.push(format!("While condition must be 'bool', got '{}'", cond_type));
                }
                self.enter_scope();
                self.loop_depth += 1;
                for &stmt in &w.body {
                    self.check_node(stmt);
                }
                self.loop_depth -= 1;
                self.exit_scope();
                "void".to_string()
            }
            ASTNode::Break(_) | ASTNode::Continue(_) => {
                if self.loop_depth == 0 {
                    self.errors.push("Loop control statement outside loop".to_string());
                }
                "void".to_string()
            }
            ASTNode::Return(r) => {
                if let Some(curr) = self.current_function {
                    if let ASTNode::FuncDecl(f) = &self.ast.nodes[curr] {
                        let expected = &f.return_type;
                        let actual = match r.expr {
                            Some(expr_id) => self.check_node(expr_id),
                            None => "void".to_string(),
                        };
                        if !self.types_compatible(expected, &actual) {
                            self.errors.push(format!("Function return type mismatch: expected '{}', got '{}'", expected, actual));
                        }
                    }
                }
                "void".to_string()
            }
            ASTNode::StructLiteral(sl) => {
                if let Some(struct_node_id) = self.global_structs.get(&sl.struct_name) {
                    if let ASTNode::StructDecl(s) = &self.ast.nodes[*struct_node_id] {
                        let mut bindings_map = HashMap::new();
                        for (k, v) in &sl.field_bindings {
                            bindings_map.insert(k.clone(), *v);
                        }
                        for (f_name, f_type) in &s.fields {
                            if let Some(&val_node_id) = bindings_map.get(f_name) {
                                let val_type = self.check_node(val_node_id);
                                if !self.types_compatible(f_type, &val_type) {
                                    self.errors.push(format!(
                                        "Field '{}' type mismatch: expected '{}', got '{}'",
                                        f_name, f_type, val_type
                                    ));
                                }
                            } else {
                                self.errors.push(format!(
                                    "Missing field '{}' in struct literal of '{}'",
                                    f_name, sl.struct_name
                                ));
                            }
                        }
                    }
                } else {
                    self.errors.push(format!("Undefined struct '{}'", sl.struct_name));
                }
                sl.struct_name.clone()
            }
            ASTNode::DotAccess(d) => {
                let obj_type = self.check_node(d.obj);
                if let Some(struct_node_id) = self.global_structs.get(&obj_type) {
                    if let ASTNode::StructDecl(s) = &self.ast.nodes[*struct_node_id] {
                        for (f_name, f_type) in &s.fields {
                            if f_name == &d.member {
                                return f_type.clone();
                            }
                        }
                        self.errors.push(format!("Struct '{}' has no field '{}'", obj_type, d.member));
                    }
                } else {
                    self.errors.push(format!("Cannot access member of non-struct object of type '{}'", obj_type));
                }
                "unknown".to_string()
            }
            ASTNode::ArrayLiteral(al) => {
                let mut elem_type = "unknown".to_string();
                if !al.elements.is_empty() {
                    elem_type = self.check_node(al.elements[0]);
                    for &el in &al.elements[1..] {
                        let t = self.check_node(el);
                        if t != elem_type {
                            elem_type = "any".to_string();
                        }
                    }
                }
                format!("array<{}>", elem_type)
            }
            ASTNode::TupleLiteral(tl) => {
                let mut types = Vec::new();
                for &el in &tl.elements {
                    types.push(self.check_node(el));
                }
                format!("tuple<{}>", types.join(", "))
            }
            ASTNode::TryCatch(tc) => {
                self.enter_scope();
                for &stmt in &tc.try_body {
                    self.check_node(stmt);
                }
                self.exit_scope();

                self.enter_scope();
                if let Some(ref var) = tc.catch_var {
                    self.declare_var(var.clone(), "any".to_string());
                }
                for &stmt in &tc.catch_body {
                    self.check_node(stmt);
                }
                self.exit_scope();
                "void".to_string()
            }
            ASTNode::Throw(t) => {
                self.check_node(t.expr);
                "void".to_string()
            }
            ASTNode::Noise(n) => {
                self.check_node(n.expr);
                for target in &n.targets {
                    let t = self.lookup_var(target);
                    if t != "qubit" {
                        self.errors.push(format!("Noise target must be qubit, got '{}'", t));
                    }
                }
                "void".to_string()
            }
            ASTNode::Call(c) => {
                let callee_name = match &c.callee {
                    CallCallee::Name(name) => name.clone(),
                    CallCallee::Node(node_id) => {
                        if let ASTNode::VarRef(v) = &self.ast.nodes[*node_id] {
                            v.name.clone()
                        } else {
                            "unknown".to_string()
                        }
                    }
                };

                if let Some(func_node_id) = self.global_funcs.get(&callee_name) {
                    if let ASTNode::FuncDecl(f) = &self.ast.nodes[*func_node_id] {
                        if c.args.len() != f.params.len() {
                            self.errors.push(format!(
                                "Argument count mismatch for call to '{}': expected {}, got {}",
                                callee_name, f.params.len(), c.args.len()
                            ));
                        }
                        let mut bindings: HashMap<String, String> = HashMap::new();
                        for (arg_node_id, (p_name, p_type)) in c.args.iter().zip(&f.params) {
                            let arg_type = self.check_node(*arg_node_id);
                            if f.generic_params.contains(p_type) {
                                if let Some(bound_type) = bindings.get(p_type) {
                                    if !self.types_compatible(bound_type, &arg_type) {
                                        self.errors.push(format!(
                                            "Generic parameter '{}' bound to conflicting types '{}' and '{}'",
                                            p_type, bound_type, arg_type
                                        ));
                                    }
                                } else {
                                    bindings.insert(p_type.clone(), arg_type);
                                }
                            } else if !self.types_compatible(p_type, &arg_type) {
                                self.errors.push(format!(
                                    "Type mismatch for parameter '{}': expected '{}', got '{}'",
                                    p_name, p_type, arg_type
                                ));
                            }
                        }
                        if let Some(bound_ret_type) = bindings.get(&f.return_type) {
                            return bound_ret_type.clone();
                        }
                        return f.return_type.clone();
                    }
                } else {
                    self.errors.push(format!("Call to undefined classic function '{}'", callee_name));
                }
                "unknown".to_string()
            }
            ASTNode::IndexAccess(ia) => {
                let obj_type = self.check_node(ia.obj);
                let idx_type = self.check_node(ia.index);
                if obj_type.starts_with("array<") {
                    if idx_type != "int" {
                        self.errors.push(format!("Array index must be 'int', got '{}'", idx_type));
                    }
                    return obj_type[6..obj_type.len() - 1].to_string();
                } else if obj_type.starts_with("map<") {
                    let parts: Vec<&str> = obj_type[4..obj_type.len() - 1].split(',').collect();
                    if parts.len() == 2 {
                        let k_type = parts[0].trim();
                        let v_type = parts[1].trim();
                        if !self.types_compatible(k_type, &idx_type) {
                            self.errors.push(format!("Map key type mismatch: expected '{}', got '{}'", k_type, idx_type));
                        }
                        return v_type.to_string();
                    }
                }
                self.errors.push(format!("Index access not supported on type '{}'", obj_type));
                "unknown".to_string()
            }
            ASTNode::MapAlloc(ma) => {
                let mut key_type = "unknown".to_string();
                let mut val_type = "unknown".to_string();
                if !ma.keys.is_empty() {
                    key_type = self.check_node(ma.keys[0]);
                    val_type = self.check_node(ma.values[0]);
                }
                format!("map<{}, {}>", key_type, val_type)
            }
            ASTNode::QFuncCall(qfc) => {
                if let Some(qfunc_node_id) = self.global_qfuncs.get(&qfc.name) {
                    if let ASTNode::QFuncDecl(qf) = &self.ast.nodes[*qfunc_node_id] {
                        if qfc.args.len() != qf.params.len() {
                            self.errors.push(format!(
                                "Argument count mismatch for qfunc '{}': expected {}, got {}",
                                qfc.name, qf.params.len(), qfc.args.len()
                            ));
                        }
                        for (arg_name, (_, param_type)) in qfc.args.iter().zip(&qf.params) {
                            let arg_type = self.lookup_var(arg_name);
                            if &arg_type != param_type {
                                self.errors.push(format!(
                                    "Type mismatch for argument '{}': expected '{}', got '{}'",
                                    arg_name, param_type, arg_type
                                ));
                            }
                        }
                    }
                } else {
                    self.errors.push(format!("Call to undefined qfunc '{}'", qfc.name));
                }
                "void".to_string()
            }
            ASTNode::Gate(g) => {
                for target in &g.targets {
                    let t_type = self.lookup_var(target);
                    if t_type != "qubit" {
                        self.errors.push(format!(
                            "Gate '{}' target '{}' must be of type 'qubit', got '{}'",
                            g.gate_name, target, t_type
                        ));
                    }
                }
                for &arg_node_id in &g.args {
                    let arg_type = self.check_node(arg_node_id);
                    if arg_type != "int" && arg_type != "float" {
                        self.errors.push(format!(
                            "Rotation gate '{}' angle must evaluate to a number, got '{}'",
                            g.gate_name, arg_type
                        ));
                    }
                }
                "void".to_string()
            }
            ASTNode::Measure(m) => {
                let q_type = self.lookup_var(&m.qubit_name);
                let c_type = self.lookup_var(&m.cbit_name);
                if q_type != "qubit" {
                    self.errors.push(format!(
                        "Measurement target '{}' must be of type 'qubit', got '{}'",
                        m.qubit_name, q_type
                    ));
                }
                if c_type != "cbit" {
                    self.errors.push(format!(
                        "Measurement destination '{}' must be of type 'cbit', got '{}'",
                        m.cbit_name, c_type
                    ));
                }
                "void".to_string()
            }
            ASTNode::If(i) => {
                let left_type = self.check_node(i.condition_left);
                let right_type = self.check_node(i.condition_right);
                let comparable = ["int", "float", "cbit", "bool"];
                if !comparable.contains(&left_type.as_str()) || !comparable.contains(&right_type.as_str()) {
                    self.errors.push(format!(
                        "Condition comparison '{}' not supported between '{}' and '{}'",
                        i.op, left_type, right_type
                    ));
                }
                self.enter_scope();
                for &stmt in &i.body {
                    self.check_node(stmt);
                }
                self.exit_scope();
                self.enter_scope();
                for &stmt in &i.else_body {
                    self.check_node(stmt);
                }
                self.exit_scope();
                "void".to_string()
            }
            ASTNode::Trace(_) => "void".to_string(),
            ASTNode::Print(p) => {
                self.check_node(p.expr);
                "void".to_string()
            }
            ASTNode::Assert(a) => {
                let left_type = self.check_node(a.condition_left);
                let right_type = self.check_node(a.condition_right);
                let comparable = ["int", "float", "cbit", "bool"];
                if !comparable.contains(&left_type.as_str()) || !comparable.contains(&right_type.as_str()) {
                    self.errors.push(format!(
                        "Assert comparison '{}' not supported between '{}' and '{}'",
                        a.op, left_type, right_type
                    ));
                }
                "void".to_string()
            }
            ASTNode::ParallelBlock(pb) => {
                for &task in &pb.tasks {
                    self.check_node(task);
                }
                "void".to_string()
            }
            ASTNode::TaskStatement(ts) => {
                self.check_node(ts.call);
                "void".to_string()
            }
            _ => "void".to_string(),
        }
    }
}

fn resolve_imports_recursive(
    node_id: NodeId,
    workspace_root: &str,
    dest_ast: &mut AST,
    visited: &mut HashSet<String>,
    merged_declarations: &mut Vec<NodeId>,
) -> Result<(), String> {
    let imports = match &dest_ast.nodes[node_id] {
        ASTNode::Program(p) => p.imports.clone(),
        _ => return Ok(()),
    };

    for imp_id in imports {
        let module_path = match &dest_ast.nodes[imp_id] {
            ASTNode::Import(imp) => imp.module_path.clone(),
            _ => continue,
        };

        if visited.contains(&module_path) {
            continue;
        }
        visited.insert(module_path.clone());

        // Resolve file path using stdlib first, then local workspace
        let relative_path = module_path.replace('.', "/") + ".eig";
        let std_modules = ["math", "std", "collections", "random", "io", "time", "string", "linalg", "quantum"];
        let first_part = module_path.split('.').next().unwrap_or("");
        
        let mut resolved_path = None;
        let stdlib_root = format!("{}/stdlib", workspace_root);

        if std_modules.contains(&first_part) {
            let path = format!("{}/{}", stdlib_root, relative_path);
            if std::path::Path::new(&path).is_file() {
                resolved_path = Some(path);
            }
        }

        if resolved_path.is_none() {
            let path = format!("{}/{}", workspace_root, relative_path);
            if std::path::Path::new(&path).is_file() {
                resolved_path = Some(path);
            }
        }

        if resolved_path.is_none() {
            let path = format!("{}/{}", stdlib_root, relative_path);
            if std::path::Path::new(&path).is_file() {
                resolved_path = Some(path);
            }
        }

        let path = match resolved_path {
            Some(p) => p,
            None => return Err(format!("Import Error: Module '{}' not found", module_path)),
        };

        // Read, tokenize, and parse the imported file
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read imported file '{}': {}", path, e))?;
        
        let mut lexer = crate::frontend::lexer::Lexer::new(&content);
        let tokens = lexer.tokenize().map_err(|e| format!("Lexer error in '{}': {}", path, e))?;
        let mut parser = crate::frontend::parser::Parser::new(tokens);
        let sub_root_id = parser.parse().map_err(|e| format!("Parser error in '{}': {}", path, e))?;

        // Resolve imports of the sub-module recursively
        resolve_imports_recursive(sub_root_id, workspace_root, &mut parser.ast, visited, merged_declarations)?;

        // Copy declarations into the main AST
        let mut id_map = HashMap::new();
        let sub_body = match &parser.ast.nodes[sub_root_id] {
            ASTNode::Program(p) => p.body.clone(),
            _ => Vec::new(),
        };

        for decl_id in sub_body {
            match &parser.ast.nodes[decl_id] {
                ASTNode::QFuncDecl(_) | ASTNode::FuncDecl(_) | ASTNode::StructDecl(_) | ASTNode::EnumDecl(_) => {
                    let new_id = copy_node(&parser.ast, dest_ast, decl_id, &mut id_map);
                    merged_declarations.push(new_id);
                }
                _ => {}
            }
        }
    }

    Ok(())
}

pub fn resolve_imports(
    root_id: NodeId,
    workspace_root: &str,
    dest_ast: &mut AST,
) -> Result<(), String> {
    let mut visited = HashSet::new();
    let mut merged_declarations = Vec::new();

    if let ASTNode::Program(p) = &dest_ast.nodes[root_id] {
        if let Some(ref m_name) = p.module_name {
            visited.insert(m_name.clone());
        }
    }

    resolve_imports_recursive(root_id, workspace_root, dest_ast, &mut visited, &mut merged_declarations)?;

    if let ASTNode::Program(ref mut p) = &mut dest_ast.nodes[root_id] {
        let mut new_body = merged_declarations;
        new_body.extend(p.body.clone());
        p.body = new_body;
    }

    Ok(())
}

fn copy_node(src: &AST, dest: &mut AST, id: NodeId, map: &mut HashMap<NodeId, NodeId>) -> NodeId {
    if let Some(&new_id) = map.get(&id) {
        return new_id;
    }
    
    let new_node = match &src.nodes[id] {
        ASTNode::Program(p) => {
            let imports: Vec<NodeId> = p.imports.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            let body: Vec<NodeId> = p.body.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::Program(ProgramNode {
                version: p.version,
                module_name: p.module_name.clone(),
                imports,
                body,
            })
        }
        ASTNode::Import(imp) => ASTNode::Import(ImportNode { module_path: imp.module_path.clone() }),
        ASTNode::QFuncDecl(qf) => {
            let body: Vec<NodeId> = qf.body.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::QFuncDecl(QFuncDeclNode {
                name: qf.name.clone(),
                params: qf.params.clone(),
                body,
            })
        }
        ASTNode::Let(l) => {
            let val = copy_node(src, dest, l.value, map);
            ASTNode::Let(LetNode {
                name: l.name.clone(),
                type_name: l.type_name.clone(),
                value: val,
            })
        }
        ASTNode::VarDecl(v) => ASTNode::VarDecl(VarDeclNode { name: v.name.clone(), type_name: v.type_name.clone() }),
        ASTNode::BinaryOp(b) => {
            let left = copy_node(src, dest, b.left, map);
            let right = copy_node(src, dest, b.right, map);
            ASTNode::BinaryOp(BinaryOpNode {
                op: b.op.clone(),
                left,
                right,
            })
        }
        ASTNode::Literal(lit) => ASTNode::Literal(LiteralNode {
            value: lit.value.clone(),
            type_name: lit.type_name.clone(),
        }),
        ASTNode::VarRef(vr) => ASTNode::VarRef(VarRefNode { name: vr.name.clone() }),
        ASTNode::QFuncCall(qfc) => ASTNode::QFuncCall(QFuncCallNode {
            name: qfc.name.clone(),
            args: qfc.args.clone(),
        }),
        ASTNode::Gate(g) => {
            let args: Vec<NodeId> = g.args.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::Gate(GateNode {
                gate_name: g.gate_name.clone(),
                targets: g.targets.clone(),
                args,
            })
        }
        ASTNode::Measure(m) => ASTNode::Measure(MeasureNode {
            qubit_name: m.qubit_name.clone(),
            cbit_name: m.cbit_name.clone(),
        }),
        ASTNode::If(if_node) => {
            let left = copy_node(src, dest, if_node.condition_left, map);
            let right = copy_node(src, dest, if_node.condition_right, map);
            let body: Vec<NodeId> = if_node.body.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            let else_body: Vec<NodeId> = if_node.else_body.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::If(IfNode {
                condition_left: left,
                op: if_node.op.clone(),
                condition_right: right,
                body,
                else_body,
            })
        }
        ASTNode::Return(r) => {
            let expr = r.expr.map(|i| copy_node(src, dest, i, map));
            ASTNode::Return(ReturnNode { expr })
        }
        ASTNode::Trace(_) => ASTNode::Trace(TraceNode {}),
        ASTNode::Print(p) => {
            let expr = copy_node(src, dest, p.expr, map);
            ASTNode::Print(PrintNode { expr })
        }
        ASTNode::Assert(a) => {
            let left = copy_node(src, dest, a.condition_left, map);
            let right = copy_node(src, dest, a.condition_right, map);
            ASTNode::Assert(AssertNode {
                condition_left: left,
                op: a.op.clone(),
                condition_right: right,
            })
        }
        ASTNode::FuncDecl(fd) => {
            let body: Vec<NodeId> = fd.body.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::FuncDecl(FuncDeclNode {
                name: fd.name.clone(),
                generic_params: fd.generic_params.clone(),
                params: fd.params.clone(),
                return_type: fd.return_type.clone(),
                body,
            })
        }
        ASTNode::For(for_node) => {
            let iter = copy_node(src, dest, for_node.iterable, map);
            let body: Vec<NodeId> = for_node.body.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::For(ForNode {
                variable: for_node.variable.clone(),
                iterable: iter,
                body,
            })
        }
        ASTNode::While(w) => {
            let cond = copy_node(src, dest, w.condition, map);
            let body: Vec<NodeId> = w.body.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::While(WhileNode {
                condition: cond,
                body,
            })
        }
        ASTNode::Break(_) => ASTNode::Break(BreakNode {}),
        ASTNode::Continue(_) => ASTNode::Continue(ContinueNode {}),
        ASTNode::StructDecl(sd) => ASTNode::StructDecl(StructDeclNode {
            name: sd.name.clone(),
            generic_params: sd.generic_params.clone(),
            fields: sd.fields.clone(),
        }),
        ASTNode::StructLiteral(sl) => {
            let bindings = sl.field_bindings.iter().map(|(k, v)| (k.clone(), copy_node(src, dest, *v, map))).collect();
            ASTNode::StructLiteral(StructLiteralNode {
                struct_name: sl.struct_name.clone(),
                field_bindings: bindings,
            })
        }
        ASTNode::DotAccess(da) => {
            let obj = copy_node(src, dest, da.obj, map);
            ASTNode::DotAccess(DotAccessNode {
                obj,
                member: da.member.clone(),
            })
        }
        ASTNode::ArrayLiteral(al) => {
            let elements = al.elements.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::ArrayLiteral(ArrayLiteralNode { elements })
        }
        ASTNode::TupleLiteral(tl) => {
            let elements = tl.elements.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::TupleLiteral(TupleLiteralNode { elements })
        }
        ASTNode::TryCatch(tc) => {
            let try_body = tc.try_body.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            let catch_body = tc.catch_body.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::TryCatch(TryCatchNode {
                try_body,
                catch_var: tc.catch_var.clone(),
                catch_body,
            })
        }
        ASTNode::Throw(t) => {
            let expr = copy_node(src, dest, t.expr, map);
            ASTNode::Throw(ThrowNode { expr })
        }
        ASTNode::EnumDecl(ed) => ASTNode::EnumDecl(EnumDeclNode {
            name: ed.name.clone(),
            variants: ed.variants.clone(),
        }),
        ASTNode::Noise(n) => {
            let expr = copy_node(src, dest, n.expr, map);
            ASTNode::Noise(NoiseNode {
                noise_type: n.noise_type.clone(),
                expr,
                targets: n.targets.clone(),
            })
        }
        ASTNode::Assignment(a) => {
            let target = copy_node(src, dest, a.target, map);
            let value = copy_node(src, dest, a.value, map);
            ASTNode::Assignment(AssignmentNode {
                target,
                op: a.op.clone(),
                value,
            })
        }
        ASTNode::Call(c) => {
            let callee = match &c.callee {
                CallCallee::Name(name) => CallCallee::Name(name.clone()),
                CallCallee::Node(node_id) => CallCallee::Node(copy_node(src, dest, *node_id, map)),
            };
            let args = c.args.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::Call(CallNode { callee, args })
        }
        ASTNode::IndexAccess(ia) => {
            let obj = copy_node(src, dest, ia.obj, map);
            let index = copy_node(src, dest, ia.index, map);
            ASTNode::IndexAccess(IndexAccessNode { obj, index })
        }
        ASTNode::MapAlloc(ma) => {
            let keys = ma.keys.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            let values = ma.values.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::MapAlloc(MapAllocNode { keys, values })
        }
        ASTNode::StructAlloc(sa) => {
            let values = sa.values.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::StructAlloc(StructAllocNode {
                field_names: sa.field_names.clone(),
                values,
            })
        }
        ASTNode::StructGet(sg) => {
            let struct_expr = copy_node(src, dest, sg.struct_expr, map);
            ASTNode::StructGet(StructGetNode {
                struct_expr,
                field_name: sg.field_name.clone(),
            })
        }
        ASTNode::StructSet(ss) => {
            let struct_expr = copy_node(src, dest, ss.struct_expr, map);
            let value_expr = copy_node(src, dest, ss.value_expr, map);
            ASTNode::StructSet(StructSetNode {
                struct_expr,
                field_name: ss.field_name.clone(),
                value_expr,
            })
        }
        ASTNode::MapGet(mg) => {
            let map_expr = copy_node(src, dest, mg.map_expr, map);
            let key_expr = copy_node(src, dest, mg.key_expr, map);
            ASTNode::MapGet(MapGetNode { map_expr, key_expr })
        }
        ASTNode::MapSet(ms) => {
            let map_expr = copy_node(src, dest, ms.map_expr, map);
            let key_expr = copy_node(src, dest, ms.key_expr, map);
            let value_expr = copy_node(src, dest, ms.value_expr, map);
            ASTNode::MapSet(MapSetNode { map_expr, key_expr, value_expr })
        }
        ASTNode::ArrayAlloc(aa) => {
            let elements = aa.elements.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::ArrayAlloc(ArrayAllocNode { elements })
        }
        ASTNode::ArrayGet(ag) => {
            let array_expr = copy_node(src, dest, ag.array_expr, map);
            let index_expr = copy_node(src, dest, ag.index_expr, map);
            ASTNode::ArrayGet(ArrayGetNode { array_expr, index_expr })
        }
        ASTNode::ArraySet(as_node) => {
            let array_expr = copy_node(src, dest, as_node.array_expr, map);
            let index_expr = copy_node(src, dest, as_node.index_expr, map);
            let value_expr = copy_node(src, dest, as_node.value_expr, map);
            ASTNode::ArraySet(ArraySetNode { array_expr, index_expr, value_expr })
        }
        ASTNode::ParallelBlock(pb) => {
            let tasks = pb.tasks.iter().map(|&i| copy_node(src, dest, i, map)).collect();
            ASTNode::ParallelBlock(ParallelBlockNode { tasks })
        }
        ASTNode::TaskStatement(ts) => {
            let call = copy_node(src, dest, ts.call, map);
            ASTNode::TaskStatement(TaskStatementNode { call })
        }
    };
    
    let new_id = dest.add(new_node);
    map.insert(id, new_id);
    new_id
}
