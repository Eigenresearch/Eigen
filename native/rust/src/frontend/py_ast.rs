use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::frontend::ast::{ASTNode, AST, NodeId, LiteralValue, CallCallee};

pub fn wrap_node_to_py(py: Python, ast: &AST, id: NodeId) -> PyResult<PyObject> {
    let ast_module = py.import_bound("src.frontend.ast")?;
    
    match &ast.nodes[id] {
        ASTNode::Program(node) => {
            let imports: Vec<PyObject> = node.imports.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let body: Vec<PyObject> = node.body.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("ProgramNode")?.call1((node.version, node.module_name.clone().into_py(py), imports, body))?;
            Ok(obj.into())
        }
        ASTNode::Import(node) => {
            let obj = ast_module.getattr("ImportNode")?.call1((&node.module_path,))?;
            Ok(obj.into())
        }
        ASTNode::QFuncDecl(node) => {
            let body: Vec<PyObject> = node.body.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("QFuncDeclNode")?.call1((&node.name, node.params.clone().into_py(py), body))?;
            Ok(obj.into())
        }
        ASTNode::Let(node) => {
            let val = wrap_node_to_py(py, ast, node.value)?;
            let obj = ast_module.getattr("LetNode")?.call1((&node.name, &node.type_name, val))?;
            Ok(obj.into())
        }
        ASTNode::VarDecl(node) => {
            let obj = ast_module.getattr("VarDeclNode")?.call1((&node.name, &node.type_name))?;
            Ok(obj.into())
        }
        ASTNode::BinaryOp(node) => {
            let left = wrap_node_to_py(py, ast, node.left)?;
            let right = wrap_node_to_py(py, ast, node.right)?;
            let obj = ast_module.getattr("BinaryOpNode")?.call1((&node.op, left, right))?;
            Ok(obj.into())
        }
        ASTNode::Literal(node) => {
            let val = match &node.value {
                LiteralValue::Int(i) => i.into_py(py),
                LiteralValue::Float(f) => f.into_py(py),
                LiteralValue::String(s) => s.into_py(py),
                LiteralValue::Bool(b) => b.into_py(py),
                LiteralValue::Null => py.None(),
            };
            let obj = ast_module.getattr("LiteralNode")?.call1((val, &node.type_name))?;
            Ok(obj.into())
        }
        ASTNode::VarRef(node) => {
            let obj = ast_module.getattr("VarRefNode")?.call1((&node.name,))?;
            Ok(obj.into())
        }
        ASTNode::QFuncCall(node) => {
            let obj = ast_module.getattr("QFuncCallNode")?.call1((&node.name, node.args.clone().into_py(py)))?;
            Ok(obj.into())
        }
        ASTNode::Gate(node) => {
            let args: Vec<PyObject> = node.args.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("GateNode")?.call1((&node.gate_name, node.targets.clone().into_py(py), args))?;
            Ok(obj.into())
        }
        ASTNode::Measure(node) => {
            let obj = ast_module.getattr("MeasureNode")?.call1((&node.qubit_name, &node.cbit_name))?;
            Ok(obj.into())
        }
        ASTNode::If(node) => {
            let cond_left = wrap_node_to_py(py, ast, node.condition_left)?;
            let cond_right = wrap_node_to_py(py, ast, node.condition_right)?;
            let body: Vec<PyObject> = node.body.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("IfNode")?.call1((cond_left, &node.op, cond_right, body))?;
            Ok(obj.into())
        }
        ASTNode::Return(node) => {
            let expr = match node.expr {
                Some(i) => wrap_node_to_py(py, ast, i)?,
                None => py.None(),
            };
            let obj = ast_module.getattr("ReturnNode")?.call1((expr,))?;
            Ok(obj.into())
        }
        ASTNode::Trace(_) => {
            let obj = ast_module.getattr("TraceNode")?.call1(())?;
            Ok(obj.into())
        }
        ASTNode::Print(node) => {
            let expr = wrap_node_to_py(py, ast, node.expr)?;
            let obj = ast_module.getattr("PrintNode")?.call1((expr,))?;
            Ok(obj.into())
        }
        ASTNode::Assert(node) => {
            let cond_left = wrap_node_to_py(py, ast, node.condition_left)?;
            let cond_right = wrap_node_to_py(py, ast, node.condition_right)?;
            let obj = ast_module.getattr("AssertNode")?.call1((cond_left, &node.op, cond_right))?;
            Ok(obj.into())
        }
        ASTNode::FuncDecl(node) => {
            let body: Vec<PyObject> = node.body.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("FuncDeclNode")?.call1((&node.name, node.generic_params.clone().into_py(py), node.params.clone().into_py(py), &node.return_type, body))?;
            Ok(obj.into())
        }
        ASTNode::For(node) => {
            let iterable = wrap_node_to_py(py, ast, node.iterable)?;
            let body: Vec<PyObject> = node.body.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("ForNode")?.call1((&node.variable, iterable, body))?;
            Ok(obj.into())
        }
        ASTNode::While(node) => {
            let cond = wrap_node_to_py(py, ast, node.condition)?;
            let body: Vec<PyObject> = node.body.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("WhileNode")?.call1((cond, body))?;
            Ok(obj.into())
        }
        ASTNode::Break(_) => {
            let obj = ast_module.getattr("BreakNode")?.call1(())?;
            Ok(obj.into())
        }
        ASTNode::Continue(_) => {
            let obj = ast_module.getattr("ContinueNode")?.call1(())?;
            Ok(obj.into())
        }
        ASTNode::StructDecl(node) => {
            let obj = ast_module.getattr("StructDeclNode")?.call1((&node.name, node.generic_params.clone().into_py(py), node.fields.clone().into_py(py)))?;
            Ok(obj.into())
        }
        ASTNode::StructLiteral(node) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in &node.field_bindings {
                dict.set_item(k, wrap_node_to_py(py, ast, *v)?)?;
            }
            let obj = ast_module.getattr("StructLiteralNode")?.call1((&node.struct_name, dict))?;
            Ok(obj.into())
        }
        ASTNode::DotAccess(node) => {
            let obj_node = wrap_node_to_py(py, ast, node.obj)?;
            let obj = ast_module.getattr("DotAccessNode")?.call1((obj_node, &node.member))?;
            Ok(obj.into())
        }
        ASTNode::ArrayLiteral(node) => {
            let elements: Vec<PyObject> = node.elements.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("ArrayLiteralNode")?.call1((elements,))?;
            Ok(obj.into())
        }
        ASTNode::TupleLiteral(node) => {
            let elements: Vec<PyObject> = node.elements.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("TupleLiteralNode")?.call1((elements,))?;
            Ok(obj.into())
        }
        ASTNode::TryCatch(node) => {
            let try_body: Vec<PyObject> = node.try_body.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let catch_body: Vec<PyObject> = node.catch_body.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("TryCatchNode")?.call1((try_body, node.catch_var.clone().into_py(py), catch_body))?;
            Ok(obj.into())
        }
        ASTNode::Throw(node) => {
            let expr = wrap_node_to_py(py, ast, node.expr)?;
            let obj = ast_module.getattr("ThrowNode")?.call1((expr,))?;
            Ok(obj.into())
        }
        ASTNode::EnumDecl(node) => {
            let obj = ast_module.getattr("EnumDeclNode")?.call1((&node.name, node.variants.clone().into_py(py)))?;
            Ok(obj.into())
        }
        ASTNode::Noise(node) => {
            let expr = wrap_node_to_py(py, ast, node.expr)?;
            let obj = ast_module.getattr("NoiseNode")?.call1((&node.noise_type, expr, node.targets.clone().into_py(py)))?;
            Ok(obj.into())
        }
        ASTNode::Assignment(node) => {
            let target = wrap_node_to_py(py, ast, node.target)?;
            let value = wrap_node_to_py(py, ast, node.value)?;
            let obj = ast_module.getattr("AssignmentNode")?.call1((target, &node.op, value))?;
            Ok(obj.into())
        }
        ASTNode::Call(node) => {
            let callee = match &node.callee {
                CallCallee::Name(name) => name.into_py(py),
                CallCallee::Node(node_id) => wrap_node_to_py(py, ast, *node_id)?,
            };
            let args: Vec<PyObject> = node.args.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("CallNode")?.call1((callee, args))?;
            Ok(obj.into())
        }
        ASTNode::IndexAccess(node) => {
            let obj_node = wrap_node_to_py(py, ast, node.obj)?;
            let index = wrap_node_to_py(py, ast, node.index)?;
            let obj = ast_module.getattr("IndexAccessNode")?.call1((obj_node, index))?;
            Ok(obj.into())
        }
        ASTNode::MapAlloc(node) => {
            let keys: Vec<PyObject> = node.keys.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let values: Vec<PyObject> = node.values.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("MapAllocNode")?.call1((keys, values))?;
            Ok(obj.into())
        }
        ASTNode::StructAlloc(node) => {
            let values: Vec<PyObject> = node.values.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("StructAllocNode")?.call1((node.field_names.clone().into_py(py), values))?;
            Ok(obj.into())
        }
        ASTNode::StructGet(node) => {
            let struct_expr = wrap_node_to_py(py, ast, node.struct_expr)?;
            let obj = ast_module.getattr("StructGetNode")?.call1((struct_expr, &node.field_name))?;
            Ok(obj.into())
        }
        ASTNode::StructSet(node) => {
            let struct_expr = wrap_node_to_py(py, ast, node.struct_expr)?;
            let value = wrap_node_to_py(py, ast, node.value_expr)?;
            let obj = ast_module.getattr("StructSetNode")?.call1((struct_expr, &node.field_name, value))?;
            Ok(obj.into())
        }
        ASTNode::MapGet(node) => {
            let map_expr = wrap_node_to_py(py, ast, node.map_expr)?;
            let key = wrap_node_to_py(py, ast, node.key_expr)?;
            let obj = ast_module.getattr("MapGetNode")?.call1((map_expr, key))?;
            Ok(obj.into())
        }
        ASTNode::MapSet(node) => {
            let map_expr = wrap_node_to_py(py, ast, node.map_expr)?;
            let key = wrap_node_to_py(py, ast, node.key_expr)?;
            let value = wrap_node_to_py(py, ast, node.value_expr)?;
            let obj = ast_module.getattr("MapSetNode")?.call1((map_expr, key, value))?;
            Ok(obj.into())
        }
        ASTNode::ArrayAlloc(node) => {
            let elements: Vec<PyObject> = node.elements.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("ArrayAllocNode")?.call1((elements,))?;
            Ok(obj.into())
        }
        ASTNode::ArrayGet(node) => {
            let array_expr = wrap_node_to_py(py, ast, node.array_expr)?;
            let index = wrap_node_to_py(py, ast, node.index_expr)?;
            let obj = ast_module.getattr("ArrayGetNode")?.call1((array_expr, index))?;
            Ok(obj.into())
        }
        ASTNode::ArraySet(node) => {
            let array_expr = wrap_node_to_py(py, ast, node.array_expr)?;
            let index = wrap_node_to_py(py, ast, node.index_expr)?;
            let value = wrap_node_to_py(py, ast, node.value_expr)?;
            let obj = ast_module.getattr("ArraySetNode")?.call1((array_expr, index, value))?;
            Ok(obj.into())
        }
        ASTNode::ParallelBlock(node) => {
            let tasks: Vec<PyObject> = node.tasks.iter().map(|&i| wrap_node_to_py(py, ast, i)).collect::<PyResult<_>>()?;
            let obj = ast_module.getattr("ParallelBlockNode")?.call1((tasks,))?;
            Ok(obj.into())
        }
        ASTNode::TaskStatement(node) => {
            let call = wrap_node_to_py(py, ast, node.call)?;
            let obj = ast_module.getattr("TaskStatementNode")?.call1((call,))?;
            Ok(obj.into())
        }
    }
}

#[pyfunction]
pub fn parse_native(py: Python, source: &str) -> PyResult<PyObject> {
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
    
    let py_ast = wrap_node_to_py(py, &parser.ast, root_id)?;
    py_ast.setattr(py, "source", source)?;
    Ok(py_ast)
}