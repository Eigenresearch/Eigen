use crate::frontend::lexer::{Token, TokenType};
use crate::frontend::ast::{
    ASTNode, AST, NodeId, ProgramNode, ImportNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, LiteralNode, LiteralValue, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, FuncDeclNode,
    ForNode, WhileNode, BreakNode, ContinueNode, StructDeclNode, StructLiteralNode,
    DotAccessNode, ArrayLiteralNode, TupleLiteralNode, TryCatchNode, ThrowNode,
    EnumDeclNode, NoiseNode, AssignmentNode, CallNode, CallCallee, IndexAccessNode,
    MapAllocNode, ParallelBlockNode, TaskStatementNode
};

pub struct Parser<'a> {
    tokens: Vec<Token<'a>>,
    pos: usize,
    pub ast: AST,
}

fn unescape_string(s: &str) -> String {
    let mut chars = s.chars().peekable();
    let mut result = String::new();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next_ch) = chars.next() {
                result.push(next_ch);
            }
        } else {
            result.push(ch);
        }
    }
    result
}

impl<'a> Parser<'a> {
    pub fn new(tokens: Vec<Token<'a>>) -> Self {
        Self {
            tokens,
            pos: 0,
            ast: AST::new(),
        }
    }

    fn error(&self, msg: &str) -> String {
        let tok = self.current();
        format!("Parser Error at line {}, col {}: {} (found {:?})", tok.line, tok.column, msg, tok)
    }

    fn current(&self) -> &Token<'a> {
        if self.pos >= self.tokens.len() {
            &self.tokens[self.tokens.len() - 1]
        } else {
            &self.tokens[self.pos]
        }
    }

    fn peek(&self, offset: usize) -> &Token<'a> {
        if self.pos + offset >= self.tokens.len() {
            &self.tokens[self.tokens.len() - 1]
        } else {
            &self.tokens[self.pos + offset]
        }
    }

    fn match_type(&mut self, types: &[TokenType]) -> Option<Token<'a>> {
        let current_type = self.current().token_type;
        if types.contains(&current_type) {
            let tok = self.current().clone();
            self.pos += 1;
            Some(tok)
        } else {
            None
        }
    }

    fn consume(&mut self, expected: TokenType, msg: &str) -> Result<Token<'a>, String> {
        let tok = self.match_type(&[expected]);
        if let Some(t) = tok {
            Ok(t)
        } else {
            Err(self.error(msg))
        }
    }

    pub fn parse(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::Eigen, "Expected 'eigen' header directive at start of file")?;
        let version_tok = if let Some(t) = self.match_type(&[TokenType::FloatLit, TokenType::IntLit]) {
            t
        } else {
            return Err(self.error("Expected version number (e.g. 1.0) after 'eigen'"));
        };
        let version = version_tok.value.parse::<f64>().map_err(|e| e.to_string())?;

        let mut module_name = None;
        if self.match_type(&[TokenType::Module]).is_some() {
            module_name = Some(self.parse_dotted_path()?);
        }

        let mut imports = Vec::new();
        while self.match_type(&[TokenType::Import]).is_some() {
            let import_path = self.parse_dotted_path()?;
            let node = self.ast.add(ASTNode::Import(ImportNode { module_path: import_path }));
            imports.push(node);
        }

        let mut body = Vec::new();
        while self.current().token_type != TokenType::Eof {
            if let Some(stmt) = self.parse_statement()? {
                body.push(stmt);
            }
        }

        let prog_node = ProgramNode {
            version,
            module_name,
            imports,
            body,
        };
        Ok(self.ast.add(ASTNode::Program(prog_node)))
    }

    fn parse_dotted_path(&mut self) -> Result<String, String> {
        let mut parts = vec![self.consume(TokenType::Identifier, "Expected identifier")?.value.to_string()];
        while self.match_type(&[TokenType::Dot]).is_some() {
            parts.push(self.consume(TokenType::Identifier, "Expected identifier after '.'")?.value.to_string());
        }
        Ok(parts.join("."))
    }

    fn parse_type(&mut self, allow_gates: bool) -> Result<String, String> {
        let mut types = vec![
            TokenType::Int, TokenType::Float, TokenType::StringType, TokenType::Bool,
            TokenType::Qubit, TokenType::Cbit, TokenType::Array, TokenType::Map,
            TokenType::Identifier
        ];
        if allow_gates {
            types.extend_from_slice(&[
                TokenType::GateH, TokenType::GateX, TokenType::GateY, TokenType::GateZ, TokenType::GateS, TokenType::GateT
            ]);
        }
        let tok = if let Some(t) = self.match_type(&types) {
            t
        } else {
            return Err(self.error("Expected type name"));
        };

        let mut type_str = match tok.token_type {
            TokenType::StringType => "string".to_string(),
            _ => tok.value.to_string(),
        };

        if self.match_type(&[TokenType::Lt]).is_some() {
            let mut generic_types = vec![self.parse_type(true)?];
            while self.match_type(&[TokenType::Comma]).is_some() {
                generic_types.push(self.parse_type(true)?);
            }
            self.consume(TokenType::Gt, "Expected '>' after generic type parameters")?;
            type_str = format!("{}<{}>", type_str, generic_types.join(", "));
        }

        Ok(type_str)
    }

    fn parse_statement(&mut self) -> Result<Option<NodeId>, String> {
        let tok = self.current().clone();

        if tok.token_type == TokenType::Qfunc {
            return Ok(Some(self.parse_qfunc_decl()?));
        }
        if tok.token_type == TokenType::Func {
            return Ok(Some(self.parse_func_decl()?));
        }
        if tok.token_type == TokenType::Struct {
            return Ok(Some(self.parse_struct_decl()?));
        }
        if tok.token_type == TokenType::Enum {
            return Ok(Some(self.parse_enum_decl()?));
        }
        if tok.token_type == TokenType::Let {
            return Ok(Some(self.parse_let()?));
        }
        if tok.token_type == TokenType::For {
            return Ok(Some(self.parse_for()?));
        }
        if tok.token_type == TokenType::While {
            return Ok(Some(self.parse_while()?));
        }
        if tok.token_type == TokenType::Break {
            self.pos += 1;
            return Ok(Some(self.ast.add(ASTNode::Break(BreakNode {}))));
        }
        if tok.token_type == TokenType::Continue {
            self.pos += 1;
            return Ok(Some(self.ast.add(ASTNode::Continue(ContinueNode {}))));
        }
        if tok.token_type == TokenType::Try {
            return Ok(Some(self.parse_try_catch()?));
        }
        if tok.token_type == TokenType::Throw {
            self.pos += 1;
            let expr = self.parse_expr()?;
            return Ok(Some(self.ast.add(ASTNode::Throw(ThrowNode { expr }))));
        }
        if tok.token_type == TokenType::Noise {
            return Ok(Some(self.parse_noise()?));
        }
        if tok.token_type == TokenType::If {
            return Ok(Some(self.parse_if()?));
        }
        if tok.token_type == TokenType::Measure {
            self.pos += 1;
            let q_tok = self.consume(TokenType::Identifier, "Expected qubit identifier to measure")?;
            self.consume(TokenType::Arrow, "Expected '->' after qubit in measure statement")?;
            let c_tok = self.consume(TokenType::Identifier, "Expected classical bit identifier to store measurement")?;
            return Ok(Some(self.ast.add(ASTNode::Measure(MeasureNode {
                qubit_name: q_tok.value.to_string(),
                cbit_name: c_tok.value.to_string(),
            }))));
        }
        if tok.token_type == TokenType::Return {
            self.pos += 1;
            let mut expr = None;
            let curr = self.current().token_type;
            if curr != TokenType::Rbrace && curr != TokenType::Eof && curr != TokenType::Comma {
                expr = Some(self.parse_expr()?);
            }
            return Ok(Some(self.ast.add(ASTNode::Return(ReturnNode { expr }))));
        }
        if tok.token_type == TokenType::Trace {
            self.pos += 1;
            return Ok(Some(self.ast.add(ASTNode::Trace(TraceNode {}))));
        }
        if tok.token_type == TokenType::Print {
            self.pos += 1;
            let expr = self.parse_expr()?;
            return Ok(Some(self.ast.add(ASTNode::Print(PrintNode { expr }))));
        }
        if tok.token_type == TokenType::Assert {
            self.pos += 1;
            let left = self.parse_expr()?;
            let cmp_ops = [
                TokenType::Eq, TokenType::Ne, TokenType::Lt, TokenType::Gt, TokenType::Le, TokenType::Ge
            ];
            if let Some(op_tok) = self.match_type(&cmp_ops) {
                let right = self.parse_expr()?;
                return Ok(Some(self.ast.add(ASTNode::Assert(AssertNode {
                    condition_left: left,
                    op: op_tok.value.to_string(),
                    condition_right: right,
                }))));
            } else {
                let right = self.ast.add(ASTNode::Literal(LiteralNode {
                    value: LiteralValue::Bool(true),
                    type_name: "bool".to_string(),
                }));
                return Ok(Some(self.ast.add(ASTNode::Assert(AssertNode {
                    condition_left: left,
                    op: "==".to_string(),
                    condition_right: right,
                }))));
            }
        }
        if tok.token_type == TokenType::Parallel {
            return Ok(Some(self.parse_parallel_block()?));
        }

        // Variable declarations: e.g. qubit q0, cbit c0, map<string, int> m, MyStruct s
        let saved_pos = self.pos;
        if let Ok(type_name) = self.parse_type(false) {
            if self.current().token_type == TokenType::Identifier {
                if let Ok(name_tok) = self.consume(TokenType::Identifier, "Expected identifier") {
                    return Ok(Some(self.ast.add(ASTNode::VarDecl(VarDeclNode {
                        name: name_tok.value.to_string(),
                        type_name,
                    }))));
                }
            }
        }
        self.pos = saved_pos;

        // Custom qfunc/func call at statement level: e.g. bell(q0, q1)
        if tok.token_type == TokenType::Identifier && self.peek(1).token_type == TokenType::Lparen {
            let name_tok = self.consume(TokenType::Identifier, "Expected identifier")?;
            self.consume(TokenType::Lparen, "Expected '('")?;
            let mut args = Vec::new();
            if self.current().token_type != TokenType::Rparen {
                args.push(self.parse_expr()?);
                while self.match_type(&[TokenType::Comma]).is_some() {
                    args.push(self.parse_expr()?);
                }
            }
            self.consume(TokenType::Rparen, "Expected ')'")?;

            // check if all are VarRef
            let mut all_var_refs = true;
            let mut var_names = Vec::new();
            for &arg in &args {
                match &self.ast.nodes[arg] {
                    ASTNode::VarRef(v) => {
                        var_names.push(v.name.clone());
                    }
                    _ => {
                        all_var_refs = false;
                        break;
                    }
                }
            }

            if all_var_refs {
                return Ok(Some(self.ast.add(ASTNode::QFuncCall(QFuncCallNode {
                    name: name_tok.value.to_string(),
                    args: var_names,
                }))));
            } else {
                let callee_node = self.ast.add(ASTNode::VarRef(VarRefNode {
                    name: name_tok.value.to_string(),
                }));
                return Ok(Some(self.ast.add(ASTNode::Call(CallNode {
                    callee: CallCallee::Node(callee_node),
                    args,
                }))));
            }
        }

        // Built-in gates
        let gates_with_1_qubit = [
            TokenType::GateH, TokenType::GateX, TokenType::GateY, TokenType::GateZ,
            TokenType::GateS, TokenType::GateT
        ];
        if gates_with_1_qubit.contains(&tok.token_type) {
            let gate_tok = self.match_type(&gates_with_1_qubit).unwrap();
            let q_tok = self.consume(TokenType::Identifier, "Expected qubit identifier")?;
            return Ok(Some(self.ast.add(ASTNode::Gate(GateNode {
                gate_name: gate_tok.value.to_string(),
                targets: vec![q_tok.value.to_string()],
                args: Vec::new(),
            }))));
        }

        let gates_with_2_qubit = [
            TokenType::GateCnot, TokenType::GateCz, TokenType::GateSwap
        ];
        if gates_with_2_qubit.contains(&tok.token_type) {
            let gate_tok = self.match_type(&gates_with_2_qubit).unwrap();
            let q1_tok = self.consume(TokenType::Identifier, "Expected first qubit identifier")?;
            self.consume(TokenType::Comma, "Expected ','")?;
            let q2_tok = self.consume(TokenType::Identifier, "Expected second qubit identifier")?;
            return Ok(Some(self.ast.add(ASTNode::Gate(GateNode {
                gate_name: gate_tok.value.to_string(),
                targets: vec![q1_tok.value.to_string(), q2_tok.value.to_string()],
                args: Vec::new(),
            }))));
        }

        let rotation_gates = [
            TokenType::GateRx, TokenType::GateRy, TokenType::GateRz
        ];
        if rotation_gates.contains(&tok.token_type) {
            let gate_tok = self.match_type(&rotation_gates).unwrap();
            let q_tok = self.consume(TokenType::Identifier, "Expected qubit identifier")?;
            self.consume(TokenType::Comma, "Expected ',' before rotation angle")?;
            let angle_expr = self.parse_expr()?;
            return Ok(Some(self.ast.add(ASTNode::Gate(GateNode {
                gate_name: gate_tok.value.to_string(),
                targets: vec![q_tok.value.to_string()],
                args: vec![angle_expr],
            }))));
        }

        let gates_with_3_qubit = [
            TokenType::GateCcx, TokenType::GateCswap
        ];
        if gates_with_3_qubit.contains(&tok.token_type) {
            let gate_tok = self.match_type(&gates_with_3_qubit).unwrap();
            let q1_tok = self.consume(TokenType::Identifier, "Expected first qubit identifier")?;
            self.consume(TokenType::Comma, "Expected ','")?;
            let q2_tok = self.consume(TokenType::Identifier, "Expected second qubit identifier")?;
            self.consume(TokenType::Comma, "Expected ','")?;
            let q3_tok = self.consume(TokenType::Identifier, "Expected third qubit identifier")?;
            
            let mut g_name = gate_tok.value.to_uppercase();
            if g_name == "TOFFOLI" || g_name == "CCX" {
                g_name = "CCX".to_string();
            } else if g_name == "FREDKIN" || g_name == "CSWAP" {
                g_name = "CSWAP".to_string();
            }
            
            return Ok(Some(self.ast.add(ASTNode::Gate(GateNode {
                gate_name: g_name,
                targets: vec![q1_tok.value.to_string(), q2_tok.value.to_string(), q3_tok.value.to_string()],
                args: Vec::new(),
            }))));
        }

        let controlled_rotations = [
            TokenType::GateCp, TokenType::GateCrx, TokenType::GateCry, TokenType::GateCrz
        ];
        if controlled_rotations.contains(&tok.token_type) {
            let gate_tok = self.match_type(&controlled_rotations).unwrap();
            let q1_tok = self.consume(TokenType::Identifier, "Expected control qubit identifier")?;
            self.consume(TokenType::Comma, "Expected ','")?;
            let q2_tok = self.consume(TokenType::Identifier, "Expected target qubit identifier")?;
            self.consume(TokenType::Comma, "Expected ',' before rotation angle")?;
            let angle_expr = self.parse_expr()?;
            return Ok(Some(self.ast.add(ASTNode::Gate(GateNode {
                gate_name: gate_tok.value.to_uppercase(),
                targets: vec![q1_tok.value.to_string(), q2_tok.value.to_string()],
                args: vec![angle_expr],
            }))));
        }

        // Try parsing as an assignment or general expression statement
        let expression_starters = [
            TokenType::Identifier, TokenType::IntLit, TokenType::FloatLit, TokenType::StringLit,
            TokenType::True, TokenType::False, TokenType::Null, TokenType::Lparen, TokenType::Lbrack,
            TokenType::Minus, TokenType::Plus, TokenType::Not, TokenType::Pi, TokenType::Tau, TokenType::E
        ];
        if expression_starters.contains(&tok.token_type) {
            let saved_pos = self.pos;
            if let Ok(expr) = self.parse_expr() {
                let assign_ops = [
                    TokenType::Equals, TokenType::AddAssign, TokenType::SubAssign, TokenType::MulAssign, TokenType::DivAssign
                ];
                if let Some(op_tok) = self.match_type(&assign_ops) {
                    if let Ok(val_expr) = self.parse_expr() {
                        return Ok(Some(self.ast.add(ASTNode::Assignment(AssignmentNode {
                            target: expr,
                            op: op_tok.value.to_string(),
                            value: val_expr,
                        }))));
                    }
                } else {
                    return Ok(Some(expr));
                }
            }
            self.pos = saved_pos;
        }

        Err(self.error(&format!("Unexpected token in statement: {:?}", tok)))
    }

    fn parse_qfunc_decl(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::Qfunc, "Expected 'qfunc'")?;
        let name_tok = self.consume(TokenType::Identifier, "Expected identifier for qfunc name")?;
        self.consume(TokenType::Lparen, "Expected '('")?;

        let mut params = Vec::new();
        if self.current().token_type != TokenType::Rparen {
            let param_types = [TokenType::Qubit, TokenType::Cbit, TokenType::Int, TokenType::Float];
            let p_type_tok = if let Some(t) = self.match_type(&param_types) {
                t
            } else {
                return Err(self.error("Expected type for qfunc parameter"));
            };
            let p_name_tok = self.consume(TokenType::Identifier, "Expected parameter name")?;
            params.push((p_name_tok.value.to_string(), p_type_tok.value.to_string()));

            while self.match_type(&[TokenType::Comma]).is_some() {
                let p_type_tok = if let Some(t) = self.match_type(&param_types) {
                    t
                } else {
                    return Err(self.error("Expected type for qfunc parameter"));
                };
                let p_name_tok = self.consume(TokenType::Identifier, "Expected parameter name")?;
                params.push((p_name_tok.value.to_string(), p_type_tok.value.to_string()));
            }
        }

        self.consume(TokenType::Rparen, "Expected ')'")?;
        self.consume(TokenType::Lbrace, "Expected '{'")?;

        let mut body = Vec::new();
        while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
            if let Some(stmt) = self.parse_statement()? {
                body.push(stmt);
            }
        }
        self.consume(TokenType::Rbrace, "Expected '}'")?;

        Ok(self.ast.add(ASTNode::QFuncDecl(QFuncDeclNode {
            name: name_tok.value.to_string(),
            params,
            body,
        })))
    }

    fn parse_func_decl(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::Func, "Expected 'func'")?;
        let name_tok = self.consume(TokenType::Identifier, "Expected function name")?;

        let mut generic_params = Vec::new();
        if self.match_type(&[TokenType::Lt]).is_some() {
            let p_tok = self.current().clone();
            if p_tok.token_type == TokenType::Identifier || p_tok.token_type == TokenType::GateT {
                self.pos += 1;
                generic_params.push(p_tok.value.to_string());
            } else {
                return Err(format!("Expected generic parameter name, found {:?}", p_tok.token_type));
            }
            while self.match_type(&[TokenType::Comma]).is_some() {
                let p_tok = self.current().clone();
                if p_tok.token_type == TokenType::Identifier || p_tok.token_type == TokenType::GateT {
                    self.pos += 1;
                    generic_params.push(p_tok.value.to_string());
                } else {
                    return Err(format!("Expected generic parameter name, found {:?}", p_tok.token_type));
                }
            }
            self.consume(TokenType::Gt, "Expected '>'")?;
        }

        self.consume(TokenType::Lparen, "Expected '('")?;
        let mut params = Vec::new();
        if self.current().token_type != TokenType::Rparen {
            let p_name_tok = self.consume(TokenType::Identifier, "Expected parameter name")?;
            self.consume(TokenType::Colon, "Expected ':'")?;
            let p_type = self.parse_type(true)?;
            params.push((p_name_tok.value.to_string(), p_type));

            while self.match_type(&[TokenType::Comma]).is_some() {
                let p_name_tok = self.consume(TokenType::Identifier, "Expected parameter name")?;
                self.consume(TokenType::Colon, "Expected ':'")?;
                let p_type = self.parse_type(true)?;
                params.push((p_name_tok.value.to_string(), p_type));
            }
        }
        self.consume(TokenType::Rparen, "Expected ')'")?;
        self.consume(TokenType::Arrow, "Expected '->'")?;
        let return_type = self.parse_type(true)?;

        self.consume(TokenType::Lbrace, "Expected '{'")?;
        let mut body = Vec::new();
        while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
            if let Some(stmt) = self.parse_statement()? {
                body.push(stmt);
            }
        }
        self.consume(TokenType::Rbrace, "Expected '}'")?;

        Ok(self.ast.add(ASTNode::FuncDecl(FuncDeclNode {
            name: name_tok.value.to_string(),
            generic_params,
            params,
            return_type,
            body,
        })))
    }

    fn parse_struct_decl(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::Struct, "Expected 'struct'")?;
        let name_tok = self.consume(TokenType::Identifier, "Expected struct name")?;

        let mut generic_params = Vec::new();
        if self.match_type(&[TokenType::Lt]).is_some() {
            let p_tok = self.current().clone();
            if p_tok.token_type == TokenType::Identifier || p_tok.token_type == TokenType::GateT {
                self.pos += 1;
                generic_params.push(p_tok.value.to_string());
            } else {
                return Err(format!("Expected generic parameter name, found {:?}", p_tok.token_type));
            }
            while self.match_type(&[TokenType::Comma]).is_some() {
                let p_tok = self.current().clone();
                if p_tok.token_type == TokenType::Identifier || p_tok.token_type == TokenType::GateT {
                    self.pos += 1;
                    generic_params.push(p_tok.value.to_string());
                } else {
                    return Err(format!("Expected generic parameter name, found {:?}", p_tok.token_type));
                }
            }
            self.consume(TokenType::Gt, "Expected '>'")?;
        }

        self.consume(TokenType::Lbrace, "Expected '{'")?;
        let mut fields = Vec::new();
        while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
            let f_name_tok = self.consume(TokenType::Identifier, "Expected field name")?;
            self.consume(TokenType::Colon, "Expected ':'")?;
            let f_type = self.parse_type(true)?;
            fields.push((f_name_tok.value.to_string(), f_type));
            self.match_type(&[TokenType::Comma]);
        }
        self.consume(TokenType::Rbrace, "Expected '}'")?;

        Ok(self.ast.add(ASTNode::StructDecl(StructDeclNode {
            name: name_tok.value.to_string(),
            generic_params,
            fields,
        })))
    }

    fn parse_enum_decl(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::Enum, "Expected 'enum'")?;
        let name_tok = self.consume(TokenType::Identifier, "Expected enum name")?;
        self.consume(TokenType::Lbrace, "Expected '{'")?;

        let mut variants = Vec::new();
        if self.current().token_type != TokenType::Rbrace {
            let v_tok = self.consume(TokenType::Identifier, "Expected variant name")?;
            variants.push(v_tok.value.to_string());
            while self.match_type(&[TokenType::Comma]).is_some() {
                if self.current().token_type == TokenType::Rbrace {
                    break;
                }
                let v_tok = self.consume(TokenType::Identifier, "Expected variant name")?;
                variants.push(v_tok.value.to_string());
            }
        }
        self.consume(TokenType::Rbrace, "Expected '}'")?;

        Ok(self.ast.add(ASTNode::EnumDecl(EnumDeclNode {
            name: name_tok.value.to_string(),
            variants,
        })))
    }

    fn parse_let(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::Let, "Expected 'let'")?;
        let name_tok = self.consume(TokenType::Identifier, "Expected identifier for variable")?;
        self.consume(TokenType::Colon, "Expected ':' after variable name")?;
        let type_name = self.parse_type(true)?;
        self.consume(TokenType::Equals, "Expected '=' in let statement")?;
        let value = self.parse_expr()?;
        Ok(self.ast.add(ASTNode::Let(LetNode {
            name: name_tok.value.to_string(),
            type_name,
            value,
        })))
    }

    fn parse_if_tail(&mut self) -> Result<Vec<NodeId>, String> {
        let mut else_body = Vec::new();
        if self.match_type(&[TokenType::Else]).is_some() {
            if self.match_type(&[TokenType::If]).is_some() {
                let left = self.parse_expr()?;
                let cmp_ops = [
                    TokenType::Eq, TokenType::Ne, TokenType::Lt, TokenType::Gt, TokenType::Le, TokenType::Ge
                ];
                let op;
                let right;
                if let Some(op_tok) = self.match_type(&cmp_ops) {
                    op = op_tok.value.to_string();
                    right = self.parse_expr()?;
                } else {
                    op = "==".to_string();
                    right = self.ast.add(ASTNode::Literal(LiteralNode {
                        value: LiteralValue::Bool(true),
                        type_name: "bool".to_string(),
                    }));
                }

                self.consume(TokenType::Lbrace, "Expected '{'")?;
                let mut body = Vec::new();
                while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
                    if let Some(stmt) = self.parse_statement()? {
                        body.push(stmt);
                    }
                }
                self.consume(TokenType::Rbrace, "Expected '}'")?;
                
                let nested_else = self.parse_if_tail()?;
                let elif_node = self.ast.add(ASTNode::If(IfNode {
                    condition_left: left,
                    op,
                    condition_right: right,
                    body,
                    else_body: nested_else,
                }));
                else_body.push(elif_node);
            } else {
                self.consume(TokenType::Lbrace, "Expected '{' after 'else'")?;
                while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
                    if let Some(stmt) = self.parse_statement()? {
                        else_body.push(stmt);
                    }
                }
                self.consume(TokenType::Rbrace, "Expected '}'")?;
            }
        } else if self.match_type(&[TokenType::Elif]).is_some() {
            let left = self.parse_expr()?;
            let cmp_ops = [
                TokenType::Eq, TokenType::Ne, TokenType::Lt, TokenType::Gt, TokenType::Le, TokenType::Ge
            ];
            let op;
            let right;
            if let Some(op_tok) = self.match_type(&cmp_ops) {
                op = op_tok.value.to_string();
                right = self.parse_expr()?;
            } else {
                op = "==".to_string();
                right = self.ast.add(ASTNode::Literal(LiteralNode {
                    value: LiteralValue::Bool(true),
                    type_name: "bool".to_string(),
                }));
            }

            self.consume(TokenType::Lbrace, "Expected '{'")?;
            let mut body = Vec::new();
            while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
                if let Some(stmt) = self.parse_statement()? {
                    body.push(stmt);
                }
            }
            self.consume(TokenType::Rbrace, "Expected '}'")?;
            
            let nested_else = self.parse_if_tail()?;
            let elif_node = self.ast.add(ASTNode::If(IfNode {
                condition_left: left,
                op,
                condition_right: right,
                body,
                else_body: nested_else,
            }));
            else_body.push(elif_node);
        }
        Ok(else_body)
    }

    fn parse_if(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::If, "Expected 'if'")?;
        let left = self.parse_expr()?;
        let cmp_ops = [
            TokenType::Eq, TokenType::Ne, TokenType::Lt, TokenType::Gt, TokenType::Le, TokenType::Ge
        ];
        let op;
        let right;
        if let Some(op_tok) = self.match_type(&cmp_ops) {
            op = op_tok.value.to_string();
            right = self.parse_expr()?;
        } else {
            op = "==".to_string();
            right = self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Bool(true),
                type_name: "bool".to_string(),
            }));
        }

        self.consume(TokenType::Lbrace, "Expected '{'")?;
        let mut body = Vec::new();
        while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
            if let Some(stmt) = self.parse_statement()? {
                body.push(stmt);
            }
        }
        self.consume(TokenType::Rbrace, "Expected '}'")?;

        let else_body = self.parse_if_tail()?;

        Ok(self.ast.add(ASTNode::If(IfNode {
            condition_left: left,
            op,
            condition_right: right,
            body,
            else_body,
        })))
    }

    fn parse_for(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::For, "Expected 'for'")?;
        let var_tok = self.consume(TokenType::Identifier, "Expected loop variable")?;
        self.consume(TokenType::In, "Expected 'in'")?;
        let iterable = self.parse_expr()?;

        self.consume(TokenType::Lbrace, "Expected '{'")?;
        let mut body = Vec::new();
        while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
            if let Some(stmt) = self.parse_statement()? {
                body.push(stmt);
            }
        }
        self.consume(TokenType::Rbrace, "Expected '}'")?;

        Ok(self.ast.add(ASTNode::For(ForNode {
            variable: var_tok.value.to_string(),
            iterable,
            body,
        })))
    }

    fn parse_while(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::While, "Expected 'while'")?;
        let condition = self.parse_expr()?;

        self.consume(TokenType::Lbrace, "Expected '{'")?;
        let mut body = Vec::new();
        while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
            if let Some(stmt) = self.parse_statement()? {
                body.push(stmt);
            }
        }
        self.consume(TokenType::Rbrace, "Expected '}'")?;

        Ok(self.ast.add(ASTNode::While(WhileNode { condition, body })))
    }

    fn parse_try_catch(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::Try, "Expected 'try'")?;
        self.consume(TokenType::Lbrace, "Expected '{'")?;
        let mut try_body = Vec::new();
        while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
            if let Some(stmt) = self.parse_statement()? {
                try_body.push(stmt);
            }
        }
        self.consume(TokenType::Rbrace, "Expected '}'")?;

        self.consume(TokenType::Catch, "Expected 'catch'")?;

        let mut catch_var = None;
        if self.match_type(&[TokenType::Lparen]).is_some() {
            let catch_var_tok = self.consume(TokenType::Identifier, "Expected catch variable name")?;
            catch_var = Some(catch_var_tok.value.to_string());
            self.consume(TokenType::Rparen, "Expected ')'")?;
        } else if self.current().token_type == TokenType::Identifier {
            let catch_var_tok = self.consume(TokenType::Identifier, "Expected catch variable name")?;
            catch_var = Some(catch_var_tok.value.to_string());
        }

        self.consume(TokenType::Lbrace, "Expected '{'")?;
        let mut catch_body = Vec::new();
        while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
            if let Some(stmt) = self.parse_statement()? {
                catch_body.push(stmt);
            }
        }
        self.consume(TokenType::Rbrace, "Expected '}'")?;

        Ok(self.ast.add(ASTNode::TryCatch(TryCatchNode {
            try_body,
            catch_var,
            catch_body,
        })))
    }

    fn parse_noise(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::Noise, "Expected 'noise'")?;
        let noise_type_tok = if let Some(t) = self.match_type(&[TokenType::Depolarizing, TokenType::Bitflip]) {
            t
        } else {
            return Err(self.error("Expected depolarizing or bitflip after noise"));
        };
        self.consume(TokenType::Lparen, "Expected '('")?;
        let expr = self.parse_expr()?;
        self.consume(TokenType::Rparen, "Expected ')'")?;

        let mut targets = Vec::new();
        if self.current().token_type == TokenType::Identifier {
            targets.push(self.consume(TokenType::Identifier, "Expected qubit identifier")?.value.to_string());
            while self.match_type(&[TokenType::Comma]).is_some() {
                targets.push(self.consume(TokenType::Identifier, "Expected qubit identifier")?.value.to_string());
            }
        }

        Ok(self.ast.add(ASTNode::Noise(NoiseNode {
            noise_type: noise_type_tok.value.to_string(),
            expr,
            targets,
        })))
    }

    fn parse_parallel_block(&mut self) -> Result<NodeId, String> {
        self.consume(TokenType::Parallel, "Expected 'parallel'")?;
        self.consume(TokenType::Lbrace, "Expected '{' after 'parallel'")?;
        let mut tasks = Vec::new();
        while self.current().token_type != TokenType::Rbrace && self.current().token_type != TokenType::Eof {
            if self.match_type(&[TokenType::Task]).is_some() {
                let call_expr = self.parse_expr()?;
                tasks.push(self.ast.add(ASTNode::TaskStatement(TaskStatementNode { call: call_expr })));
            } else {
                if let Some(stmt) = self.parse_statement()? {
                    tasks.push(stmt);
                }
            }
        }
        self.consume(TokenType::Rbrace, "Expected '}' to close parallel block")?;
        Ok(self.ast.add(ASTNode::ParallelBlock(ParallelBlockNode { tasks })))
    }

    // Expressions
    fn parse_expr(&mut self) -> Result<NodeId, String> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_logical_and()?;
        while self.match_type(&[TokenType::Or]).is_some() {
            let right = self.parse_logical_and()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: "or".to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_logical_and(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_equality()?;
        while self.match_type(&[TokenType::And]).is_some() {
            let right = self.parse_equality()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: "and".to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_equality(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_comparison()?;
        while let Some(op_tok) = self.match_type(&[TokenType::Eq, TokenType::Ne]) {
            let right = self.parse_comparison()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: op_tok.value.to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_comparison(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_bitwise_or()?;
        while let Some(op_tok) = self.match_type(&[
            TokenType::Lt, TokenType::Gt, TokenType::Le, TokenType::Ge
        ]) {
            let right = self.parse_bitwise_or()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: op_tok.value.to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_bitwise_or(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_bitwise_xor()?;
        while let Some(op_tok) = self.match_type(&[TokenType::BitOr]) {
            let right = self.parse_bitwise_xor()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: op_tok.value.to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_bitwise_xor(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_bitwise_and()?;
        while let Some(op_tok) = self.match_type(&[TokenType::BitXor]) {
            let right = self.parse_bitwise_and()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: op_tok.value.to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_bitwise_and(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_bitwise_shift()?;
        while let Some(op_tok) = self.match_type(&[TokenType::BitAnd]) {
            let right = self.parse_bitwise_shift()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: op_tok.value.to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_bitwise_shift(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_additive()?;
        while let Some(op_tok) = self.match_type(&[TokenType::Lshift, TokenType::Rshift]) {
            let right = self.parse_additive()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: op_tok.value.to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_additive(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_multiplicative()?;
        while let Some(op_tok) = self.match_type(&[TokenType::Plus, TokenType::Minus]) {
            let right = self.parse_multiplicative()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: op_tok.value.to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_multiplicative(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_unary()?;
        while let Some(op_tok) = self.match_type(&[TokenType::Mul, TokenType::Div, TokenType::Mod]) {
            let right = self.parse_unary()?;
            node = self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: op_tok.value.to_string(),
                left: node,
                right,
            }));
        }
        Ok(node)
    }

    fn parse_unary(&mut self) -> Result<NodeId, String> {
        if self.match_type(&[TokenType::Not]).is_some() {
            let right = self.parse_unary()?;
            let true_lit = self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Bool(true),
                type_name: "bool".to_string(),
            }));
            return Ok(self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: "not".to_string(),
                left: right,
                right: true_lit,
            })));
        }
        if self.match_type(&[TokenType::BitNot]).is_some() {
            let right = self.parse_unary()?;
            let zero_lit = self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Int(0),
                type_name: "int".to_string(),
            }));
            return Ok(self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: "~".to_string(),
                left: right,
                right: zero_lit,
            })));
        }
        if self.match_type(&[TokenType::Minus]).is_some() {
            if self.current().token_type == TokenType::IntLit {
                let lit_tok = self.consume(TokenType::IntLit, "")?;
                let val_str = format!("-{}", lit_tok.value);
                let val = val_str.parse::<i64>().map_err(|e| e.to_string())?;
                return Ok(self.ast.add(ASTNode::Literal(LiteralNode {
                    value: LiteralValue::Int(val),
                    type_name: "int".to_string(),
                })));
            } else if self.current().token_type == TokenType::FloatLit {
                let lit_tok = self.consume(TokenType::FloatLit, "")?;
                let val_str = format!("-{}", lit_tok.value);
                let val = val_str.parse::<f64>().map_err(|e| e.to_string())?;
                return Ok(self.ast.add(ASTNode::Literal(LiteralNode {
                    value: LiteralValue::Float(val),
                    type_name: "float".to_string(),
                })));
            }

            let right = self.parse_unary()?;
            let zero_lit = self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Int(0),
                type_name: "int".to_string(),
            }));
            return Ok(self.ast.add(ASTNode::BinaryOp(BinaryOpNode {
                op: "-".to_string(),
                left: zero_lit,
                right,
            })));
        }
        if self.match_type(&[TokenType::Plus]).is_some() {
            return self.parse_unary();
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<NodeId, String> {
        let mut node = self.parse_primary()?;
        loop {
            if self.match_type(&[TokenType::Dot]).is_some() {
                let member_tok = self.consume(TokenType::Identifier, "Expected member name after '.'")?;
                node = self.ast.add(ASTNode::DotAccess(DotAccessNode {
                    obj: node,
                    member: member_tok.value.to_string(),
                }));
            } else if self.match_type(&[TokenType::Lparen]).is_some() {
                let mut args = Vec::new();
                if self.current().token_type != TokenType::Rparen {
                    args.push(self.parse_expr()?);
                    while self.match_type(&[TokenType::Comma]).is_some() {
                        args.push(self.parse_expr()?);
                    }
                }
                self.consume(TokenType::Rparen, "Expected ')'")?;
                node = self.ast.add(ASTNode::Call(CallNode {
                    callee: CallCallee::Node(node),
                    args,
                }));
            } else if self.match_type(&[TokenType::Lbrack]).is_some() {
                let index_expr = self.parse_expr()?;
                self.consume(TokenType::Rbrack, "Expected ']'")?;
                node = self.ast.add(ASTNode::IndexAccess(IndexAccessNode {
                    obj: node,
                    index: index_expr,
                }));
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_primary(&mut self) -> Result<NodeId, String> {
        // Map literal
        if self.match_type(&[TokenType::Lbrace]).is_some() {
            let mut keys = Vec::new();
            let mut values = Vec::new();
            if self.current().token_type != TokenType::Rbrace {
                let key_expr = self.parse_expr()?;
                self.consume(TokenType::Colon, "Expected ':' after key in map literal")?;
                let val_expr = self.parse_expr()?;
                keys.push(key_expr);
                values.push(val_expr);
                while self.match_type(&[TokenType::Comma]).is_some() {
                    if self.current().token_type == TokenType::Rbrace {
                        break;
                    }
                    let key_expr = self.parse_expr()?;
                    self.consume(TokenType::Colon, "Expected ':' after key in map literal")?;
                    let val_expr = self.parse_expr()?;
                    keys.push(key_expr);
                    values.push(val_expr);
                }
            }
            self.consume(TokenType::Rbrace, "Expected '}' to close map literal")?;
            return Ok(self.ast.add(ASTNode::MapAlloc(MapAllocNode { keys, values })));
        }

        // Parenthesized expression or Tuple literal
        if self.match_type(&[TokenType::Lparen]).is_some() {
            if self.match_type(&[TokenType::Rparen]).is_some() {
                return Ok(self.ast.add(ASTNode::TupleLiteral(TupleLiteralNode { elements: Vec::new() })));
            }
            let mut exprs = vec![self.parse_expr()?];
            let mut is_tuple = false;
            while self.match_type(&[TokenType::Comma]).is_some() {
                is_tuple = true;
                if self.current().token_type == TokenType::Rparen {
                    break;
                }
                exprs.push(self.parse_expr()?);
            }
            self.consume(TokenType::Rparen, "Expected ')'")?;
            if is_tuple {
                return Ok(self.ast.add(ASTNode::TupleLiteral(TupleLiteralNode { elements: exprs })));
            } else {
                return Ok(exprs[0]);
            }
        }

        // Array literal
        if self.match_type(&[TokenType::Lbrack]).is_some() {
            let mut elements = Vec::new();
            if self.current().token_type != TokenType::Rbrack {
                elements.push(self.parse_expr()?);
                while self.match_type(&[TokenType::Comma]).is_some() {
                    if self.current().token_type == TokenType::Rbrack {
                        break;
                    }
                    elements.push(self.parse_expr()?);
                }
            }
            self.consume(TokenType::Rbrack, "Expected ']'")?;
            return Ok(self.ast.add(ASTNode::ArrayLiteral(ArrayLiteralNode { elements })));
        }

        // Constants
        if let Some(const_tok) = self.match_type(&[TokenType::Pi, TokenType::Tau, TokenType::E]) {
            let val = match const_tok.token_type {
                TokenType::Pi => 3.141592653589793,
                TokenType::Tau => 6.283185307179586,
                TokenType::E => 2.718281828459045,
                _ => unreachable!(),
            };
            return Ok(self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Float(val),
                type_name: "float".to_string(),
            })));
        }

        // Literals
        if let Some(lit_tok) = self.match_type(&[TokenType::IntLit]) {
            let val = lit_tok.value.parse::<i64>().map_err(|e| e.to_string())?;
            return Ok(self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Int(val),
                type_name: "int".to_string(),
            })));
        }

        if let Some(lit_tok) = self.match_type(&[TokenType::FloatLit]) {
            let val = lit_tok.value.parse::<f64>().map_err(|e| e.to_string())?;
            return Ok(self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Float(val),
                type_name: "float".to_string(),
            })));
        }

        if let Some(lit_tok) = self.match_type(&[TokenType::StringLit]) {
            let unescaped = unescape_string(lit_tok.value);
            return Ok(self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::String(unescaped),
                type_name: "string".to_string(),
            })));
        }

        if self.match_type(&[TokenType::True]).is_some() {
            return Ok(self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Bool(true),
                type_name: "bool".to_string(),
            })));
        }

        if self.match_type(&[TokenType::False]).is_some() {
            return Ok(self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Bool(false),
                type_name: "bool".to_string(),
            })));
        }

        if self.match_type(&[TokenType::Null]).is_some() {
            return Ok(self.ast.add(ASTNode::Literal(LiteralNode {
                value: LiteralValue::Null,
                type_name: "null".to_string(),
            })));
        }

        // Struct literal or variable reference
        if self.current().token_type == TokenType::Identifier {
            if self.peek(1).token_type == TokenType::Lbrace {
                // Disambiguate
                let mut is_struct_literal = false;
                if self.peek(2).token_type == TokenType::Identifier && self.peek(3).token_type == TokenType::Colon {
                    is_struct_literal = true;
                } else if self.peek(2).token_type == TokenType::Rbrace {
                    is_struct_literal = true;
                }

                if is_struct_literal {
                    let struct_name_tok = self.consume(TokenType::Identifier, "Expected struct name")?;
                    self.consume(TokenType::Lbrace, "Expected '{'")?;
                    let mut bindings = Vec::new();
                    if self.current().token_type != TokenType::Rbrace {
                        let field_name = self.consume(TokenType::Identifier, "Expected field name")?.value.to_string();
                        self.consume(TokenType::Colon, "Expected ':'")?;
                        let field_val = self.parse_expr()?;
                        bindings.push((field_name, field_val));
                        while self.match_type(&[TokenType::Comma]).is_some() {
                            if self.current().token_type == TokenType::Rbrace {
                                break;
                            }
                            let field_name = self.consume(TokenType::Identifier, "Expected field name")?.value.to_string();
                            self.consume(TokenType::Colon, "Expected ':'")?;
                            let field_val = self.parse_expr()?;
                            bindings.push((field_name, field_val));
                        }
                    }
                    self.consume(TokenType::Rbrace, "Expected '}'")?;
                    return Ok(self.ast.add(ASTNode::StructLiteral(StructLiteralNode {
                        struct_name: struct_name_tok.value.to_string(),
                        field_bindings: bindings,
                    })));
                }
            }

            let ref_tok = self.consume(TokenType::Identifier, "Expected identifier")?;
            return Ok(self.ast.add(ASTNode::VarRef(VarRefNode {
                name: ref_tok.value.to_string(),
            })));
        }

        Err(self.error("Expected number, constant, literal, variable reference, or '('"))
    }
}
