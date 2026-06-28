use serde::Serialize;

pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum LiteralValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum CallCallee {
    Name(String),
    Node(NodeId),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProgramNode {
    pub version: f64,
    pub module_name: Option<String>,
    pub imports: Vec<NodeId>,
    pub body: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImportNode {
    pub module_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QFuncDeclNode {
    pub name: String,
    pub params: Vec<(String, String)>,
    pub body: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LetNode {
    pub name: String,
    pub type_name: String,
    pub value: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VarDeclNode {
    pub name: String,
    pub type_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BinaryOpNode {
    pub op: String,
    pub left: NodeId,
    pub right: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiteralNode {
    pub value: LiteralValue,
    pub type_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VarRefNode {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QFuncCallNode {
    pub name: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateNode {
    pub gate_name: String,
    pub targets: Vec<String>,
    pub args: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MeasureNode {
    pub qubit_name: String,
    pub cbit_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IfNode {
    pub condition_left: NodeId,
    pub op: String,
    pub condition_right: NodeId,
    pub body: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReturnNode {
    pub expr: Option<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TraceNode {}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PrintNode {
    pub expr: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AssertNode {
    pub condition_left: NodeId,
    pub op: String,
    pub condition_right: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FuncDeclNode {
    pub name: String,
    pub generic_params: Vec<String>,
    pub params: Vec<(String, String)>,
    pub return_type: String,
    pub body: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ForNode {
    pub variable: String,
    pub iterable: NodeId,
    pub body: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WhileNode {
    pub condition: NodeId,
    pub body: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BreakNode {}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContinueNode {}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StructDeclNode {
    pub name: String,
    pub generic_params: Vec<String>,
    pub fields: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StructLiteralNode {
    pub struct_name: String,
    pub field_bindings: Vec<(String, NodeId)>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DotAccessNode {
    pub obj: NodeId,
    pub member: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArrayLiteralNode {
    pub elements: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TupleLiteralNode {
    pub elements: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TryCatchNode {
    pub try_body: Vec<NodeId>,
    pub catch_var: Option<String>,
    pub catch_body: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThrowNode {
    pub expr: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnumDeclNode {
    pub name: String,
    pub variants: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NoiseNode {
    pub noise_type: String,
    pub expr: NodeId,
    pub targets: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AssignmentNode {
    pub target: NodeId,
    pub op: String,
    pub value: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CallNode {
    pub callee: CallCallee,
    pub args: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IndexAccessNode {
    pub obj: NodeId,
    pub index: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MapAllocNode {
    pub keys: Vec<NodeId>,
    pub values: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StructAllocNode {
    pub field_names: Vec<String>,
    pub values: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StructGetNode {
    pub struct_expr: NodeId,
    pub field_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StructSetNode {
    pub struct_expr: NodeId,
    pub field_name: String,
    pub value_expr: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MapGetNode {
    pub map_expr: NodeId,
    pub key_expr: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MapSetNode {
    pub map_expr: NodeId,
    pub key_expr: NodeId,
    pub value_expr: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArrayAllocNode {
    pub elements: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArrayGetNode {
    pub array_expr: NodeId,
    pub index_expr: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArraySetNode {
    pub array_expr: NodeId,
    pub index_expr: NodeId,
    pub value_expr: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ParallelBlockNode {
    pub tasks: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TaskStatementNode {
    pub call: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ASTNode {
    Program(ProgramNode),
    Import(ImportNode),
    QFuncDecl(QFuncDeclNode),
    Let(LetNode),
    VarDecl(VarDeclNode),
    BinaryOp(BinaryOpNode),
    Literal(LiteralNode),
    VarRef(VarRefNode),
    QFuncCall(QFuncCallNode),
    Gate(GateNode),
    Measure(MeasureNode),
    If(IfNode),
    Return(ReturnNode),
    Trace(TraceNode),
    Print(PrintNode),
    Assert(AssertNode),
    FuncDecl(FuncDeclNode),
    For(ForNode),
    While(WhileNode),
    Break(BreakNode),
    Continue(ContinueNode),
    StructDecl(StructDeclNode),
    StructLiteral(StructLiteralNode),
    DotAccess(DotAccessNode),
    ArrayLiteral(ArrayLiteralNode),
    TupleLiteral(TupleLiteralNode),
    TryCatch(TryCatchNode),
    Throw(ThrowNode),
    EnumDecl(EnumDeclNode),
    Noise(NoiseNode),
    Assignment(AssignmentNode),
    Call(CallNode),
    IndexAccess(IndexAccessNode),
    MapAlloc(MapAllocNode),
    StructAlloc(StructAllocNode),
    StructGet(StructGetNode),
    StructSet(StructSetNode),
    MapGet(MapGetNode),
    MapSet(MapSetNode),
    ArrayAlloc(ArrayAllocNode),
    ArrayGet(ArrayGetNode),
    ArraySet(ArraySetNode),
    ParallelBlock(ParallelBlockNode),
    TaskStatement(TaskStatementNode),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AST {
    pub nodes: Vec<ASTNode>,
}

impl AST {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add(&mut self, node: ASTNode) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }
}
