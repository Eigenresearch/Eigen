from src.backend.bytecode import Opcode, Instruction
from src.frontend.ast import (
    ProgramNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, LiteralNode, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode, ASTNode,
    FuncDeclNode, ForNode, WhileNode, BreakNode, ContinueNode, StructDeclNode,
    StructLiteralNode, DotAccessNode, ArrayLiteralNode, TupleLiteralNode,
    TryCatchNode, ThrowNode, EnumDeclNode, NoiseNode, AssignmentNode, CallNode,
    IndexAccessNode, StructAllocNode, StructGetNode, StructSetNode,
    MapAllocNode, MapGetNode, MapSetNode, ArrayAllocNode, ArrayGetNode, ArraySetNode,
    ParallelBlockNode, TaskStatementNode, MatchNode, StringInterpolationNode
)
from src.ir.ir_graph import EQIRGraph, EQIRNode
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser

class Label:
    def __init__(self, name: str = "label"):
        self.name = name
        self.target_idx = None

    def __repr__(self) -> str:
        return f"Label({self.name}, target={self.target_idx})"


class EBCCompiler:
    def __init__(self, peephole: bool = True):
        self.raw_code = []  # list of Instruction and Label objects
        self.qfuncs = {}    # func_name -> Label
        self.global_structs = {}
        self.current_line = None
        self.loop_stack = []  # stack of (start_label, end_label)
        self.temp_var_counter = 0
        self.peephole = peephole

    def get_temp_var(self) -> str:
        self.temp_var_counter += 1
        return f"_compiler_temp_{self.temp_var_counter}"

    _cond_op_map = {
        "==": Opcode.EQ,
        "!=": Opcode.NEQ,
        "<": Opcode.LT,
        ">": Opcode.GT,
        "<=": Opcode.LTE,
        ">=": Opcode.GTE,
    }

    def _compile_condition(self, left, op, right):
        self.visit_ast(left)
        self.visit_ast(right)
        self.emit(self._cond_op_map.get(op, Opcode.EQ))

    def emit(self, opcode: str, arg=None):
        self.raw_code.append(Instruction(opcode, arg, line=self.current_line))

    def emit_label(self, label: Label):
        self.raw_code.append(label)

    def compile_ast(self, program: ProgramNode) -> list[Instruction]:
        self.raw_code = []
        self.qfuncs = {}
        self.global_structs = {}
        self.loop_stack = []
        self.temp_var_counter = 0

        # Register structs
        for node in program.body:
            if isinstance(node, StructDeclNode):
                self.global_structs[node.name] = node

        # 1. Identify all function declarations (qfuncs and classic funcs) and assign their start labels.
        # We place them at the top, but jump over them when executing the main body.
        main_start = Label("main_start")
        self.emit(Opcode.JMP, main_start)

        for node in program.body:
            if isinstance(node, (QFuncDeclNode, FuncDeclNode)):
                func_label = Label(f"func_{node.name}")
                self.qfuncs[node.name] = func_label
                
                self.emit_label(func_label)
                self.emit(Opcode.ENTER_FRAME)
                
                # Pop arguments in reverse order and store in params
                for param_name, param_type in reversed(node.params):
                    self.emit(Opcode.STORE_VAR, param_name)
                
                # Compile function body
                for stmt in node.body:
                    self.visit_ast(stmt)
                
                # Fallback return
                self.emit(Opcode.LOAD_CONST, None)
                self.emit(Opcode.RET)

        self.emit_label(main_start)

        # 2. Compile main body statements (skip declarations)
        for node in program.body:
            if isinstance(node, (QFuncDeclNode, FuncDeclNode, StructDeclNode, EnumDeclNode)):
                continue
            self.visit_ast(node)

        # Halt main execution
        self.emit(Opcode.HALT)

        return self.resolve_labels()

    def visit_ast(self, node: ASTNode):
        if hasattr(node, "line"):
            self.current_line = getattr(node, "line")

        if isinstance(node, VarDeclNode):
            if node.type_name == "qubit":
                self.emit(Opcode.Q_ALLOC, node.name)
            elif node.type_name == "cbit":
                self.emit(Opcode.LOAD_CONST, 0)
                self.emit(Opcode.STORE_VAR, node.name)
            elif node.type_name.startswith("map<"):
                self.emit(Opcode.ALLOC_MAP, 0)
                self.emit(Opcode.STORE_VAR, node.name)
            elif node.type_name.startswith("array<"):
                self.emit(Opcode.ALLOC_ARRAY, 0)
                self.emit(Opcode.STORE_VAR, node.name)
            else:
                self.emit(Opcode.LOAD_CONST, None)
                self.emit(Opcode.STORE_VAR, node.name)

        elif isinstance(node, LetNode):
            self.visit_ast(node.value)
            self.emit(Opcode.STORE_VAR, node.name)

        elif isinstance(node, LiteralNode):
            self.emit(Opcode.LOAD_CONST, node.value)

        elif isinstance(node, VarRefNode):
            self.emit(Opcode.LOAD_VAR, node.name)

        elif isinstance(node, BinaryOpNode):
            if node.op == "not":
                self.visit_ast(node.left)
                self.emit(Opcode.NOT)
            elif node.op == "~":
                self.visit_ast(node.left)
                self.emit(Opcode.BIT_NOT)
            else:
                self.visit_ast(node.left)
                self.visit_ast(node.right)
                op_map = {
                    "+": Opcode.ADD,
                    "-": Opcode.SUB,
                    "*": Opcode.MUL,
                    "/": Opcode.DIV,
                    "**": Opcode.POW,
                    "==": Opcode.EQ,
                    "!=": Opcode.NEQ,
                    "<": Opcode.LT,
                    ">": Opcode.GT,
                    "<=": Opcode.LTE,
                    ">=": Opcode.GTE,
                    "and": Opcode.AND,
                    "or": Opcode.OR,
                    "%": Opcode.MOD,
                    "&": Opcode.BIT_AND,
                    "|": Opcode.BIT_OR,
                    "^": Opcode.BIT_XOR,
                    "<<": Opcode.SHL,
                    ">>": Opcode.SHR,
                }
                if node.op in op_map:
                    self.emit(op_map[node.op])
                else:
                    raise ValueError(f"Unsupported binary operator: {node.op}")

        elif isinstance(node, GateNode):
            for arg in node.args:
                self.visit_ast(arg)
            self.emit(Opcode.Q_GATE, (node.gate_name, node.targets))

        elif isinstance(node, MeasureNode):
            self.emit(Opcode.Q_MEASURE, (node.qubit_name, node.cbit_name))

        elif isinstance(node, QFuncCallNode):
            for arg_name in node.args:
                self.emit(Opcode.LOAD_VAR, arg_name)
            func_label = self.qfuncs.get(node.name, node.name)
            self.emit(Opcode.CALL, (func_label, node.name, len(node.args)))

        elif isinstance(node, CallNode):
            for arg in node.args:
                self.visit_ast(arg)
            callee_name = None
            if isinstance(node.callee, VarRefNode):
                callee_name = node.callee.name
            elif isinstance(node.callee, str):
                callee_name = node.callee
            
            func_label = self.qfuncs.get(callee_name, callee_name)
            self.emit(Opcode.CALL, (func_label, callee_name, len(node.args)))

        elif isinstance(node, IfNode):
            self._compile_condition(node.condition_left, node.op, node.condition_right)
            
            else_label = Label("if_else")
            end_label = Label("if_end")
            self.emit(Opcode.JMP_IF_FALSE, else_label)
            
            for stmt in node.body:
                self.visit_ast(stmt)
                
            if hasattr(node, "else_body") and node.else_body:
                self.emit(Opcode.JMP, end_label)
                
            self.emit_label(else_label)
            if hasattr(node, "else_body") and node.else_body:
                for stmt in node.else_body:
                    self.visit_ast(stmt)
                self.emit_label(end_label)

        elif isinstance(node, ReturnNode):
            if hasattr(node, "expr") and node.expr is not None:
                self.visit_ast(node.expr)
            else:
                self.emit(Opcode.LOAD_CONST, None)
            self.emit(Opcode.RET)

        elif isinstance(node, TraceNode):
            self.emit(Opcode.Q_TRACE)

        elif isinstance(node, PrintNode):
            self.visit_ast(node.expr)
            self.emit(Opcode.PRINT)

        elif isinstance(node, AssertNode):
            self._compile_condition(node.condition_left, node.op, node.condition_right)
            
            ok_label = Label("assert_ok")
            self.emit(Opcode.JMP_IF_TRUE, ok_label)
            self.emit(Opcode.LOAD_CONST, f"Assertion Failed: {node.condition_left.to_source()} {node.op} {node.condition_right.to_source()}")
            self.emit(Opcode.THROW)
            self.emit_label(ok_label)

        elif isinstance(node, WhileNode):
            start_label = Label("while_start")
            end_label = Label("while_end")
            
            self.loop_stack.append((start_label, end_label))
            
            self.emit_label(start_label)
            self.visit_ast(node.condition)
            self.emit(Opcode.JMP_IF_FALSE, end_label)
            
            for stmt in node.body:
                self.visit_ast(stmt)
                
            self.emit(Opcode.JMP, start_label)
            self.emit_label(end_label)
            self.loop_stack.pop()

        elif isinstance(node, ForNode):
            start_label = Label("for_start")
            end_label = Label("for_end")
            
            self.loop_stack.append((start_label, end_label))
            
            arr_ref = self.get_temp_var()
            arr_idx = self.get_temp_var()
            
            self.visit_ast(node.iterable)
            self.emit(Opcode.STORE_VAR, arr_ref)
            
            self.emit(Opcode.LOAD_CONST, 0)
            self.emit(Opcode.STORE_VAR, arr_idx)
            
            self.emit_label(start_label)
            
            self.emit(Opcode.LOAD_VAR, arr_idx)
            self.emit(Opcode.LOAD_VAR, arr_ref)
            self.emit(Opcode.LEN)
            self.emit(Opcode.LT)
            self.emit(Opcode.JMP_IF_FALSE, end_label)
            
            self.emit(Opcode.LOAD_VAR, arr_ref)
            self.emit(Opcode.LOAD_VAR, arr_idx)
            self.emit(Opcode.GET_INDEX)
            self.emit(Opcode.STORE_VAR, node.variable)
            
            for stmt in node.body:
                self.visit_ast(stmt)
                
            self.emit(Opcode.LOAD_VAR, arr_idx)
            self.emit(Opcode.LOAD_CONST, 1)
            self.emit(Opcode.ADD)
            self.emit(Opcode.STORE_VAR, arr_idx)
            
            self.emit(Opcode.JMP, start_label)
            self.emit_label(end_label)
            self.loop_stack.pop()

        elif isinstance(node, BreakNode):
            if not self.loop_stack:
                raise ValueError("Break statement outside loop")
            self.emit(Opcode.JMP, self.loop_stack[-1][1])

        elif isinstance(node, ContinueNode):
            if not self.loop_stack:
                raise ValueError("Continue statement outside loop")
            self.emit(Opcode.JMP, self.loop_stack[-1][0])

        elif isinstance(node, TryCatchNode):
            catch_label = Label("catch_block")
            end_label = Label("try_end")
            
            self.emit(Opcode.PUSH_TRY, catch_label)
            for stmt in node.try_body:
                self.visit_ast(stmt)
            self.emit(Opcode.POP_TRY)
            self.emit(Opcode.JMP, end_label)
            
            self.emit_label(catch_label)
            if node.catch_var:
                self.emit(Opcode.STORE_VAR, node.catch_var)
            else:
                self.emit(Opcode.STORE_VAR, "_unused_exception")
                
            for stmt in node.catch_body:
                self.visit_ast(stmt)
                
            self.emit_label(end_label)

        elif isinstance(node, ThrowNode):
            self.visit_ast(node.expr)
            self.emit(Opcode.THROW)

        # StructLiteralNode
        elif isinstance(node, StructLiteralNode):
            s_decl = self.global_structs.get(node.struct_name)
            bindings = node.field_bindings
            if isinstance(bindings, list):
                bindings = dict(bindings)
                
            if s_decl:
                for f_name, f_type in s_decl.fields:
                    self.visit_ast(bindings[f_name])
                self.emit(Opcode.ALLOC_STRUCT, [f_name for f_name, _ in s_decl.fields])
            else:
                for name, val in bindings.items():
                    self.visit_ast(val)
                self.emit(Opcode.ALLOC_STRUCT, list(bindings.keys()))

        elif isinstance(node, DotAccessNode):
            self.visit_ast(node.obj)
            self.emit(Opcode.GET_FIELD, node.member)

        elif isinstance(node, IndexAccessNode):
            self.visit_ast(node.obj)
            self.visit_ast(node.index)
            self.emit(Opcode.GET_INDEX)

        elif isinstance(node, ArrayLiteralNode):
            for elem in node.elements:
                self.visit_ast(elem)
            self.emit(Opcode.ALLOC_ARRAY, len(node.elements))

        elif isinstance(node, TupleLiteralNode):
            for elem in node.elements:
                self.visit_ast(elem)
            self.emit(Opcode.ALLOC_ARRAY, len(node.elements))

        elif isinstance(node, NoiseNode):
            self.visit_ast(node.expr)
            self.emit(Opcode.Q_NOISE, (node.noise_type, node.targets))

        elif isinstance(node, AssignmentNode):
            is_compound = node.op != "="
            op_map = {
                "+=": Opcode.ADD,
                "-=": Opcode.SUB,
                "*=": Opcode.MUL,
                "/=": Opcode.DIV
            }
            
            if isinstance(node.target, VarRefNode):
                if is_compound:
                    self.emit(Opcode.LOAD_VAR, node.target.name)
                    self.visit_ast(node.value)
                    self.emit(op_map[node.op])
                else:
                    self.visit_ast(node.value)
                self.emit(Opcode.STORE_VAR, node.target.name)
                
            elif isinstance(node.target, DotAccessNode):
                if is_compound:
                    temp_obj = self.get_temp_var()
                    self.visit_ast(node.target.obj)
                    self.emit(Opcode.STORE_VAR, temp_obj)
                    self.emit(Opcode.LOAD_VAR, temp_obj)
                    self.emit(Opcode.LOAD_VAR, temp_obj)
                    self.emit(Opcode.GET_FIELD, node.target.member)
                    self.visit_ast(node.value)
                    self.emit(op_map[node.op])
                    self.emit(Opcode.SET_FIELD, node.target.member)
                else:
                    self.visit_ast(node.target.obj)
                    self.visit_ast(node.value)
                    self.emit(Opcode.SET_FIELD, node.target.member)
                
            elif isinstance(node.target, IndexAccessNode):
                if is_compound:
                    temp_obj = self.get_temp_var()
                    temp_idx = self.get_temp_var()
                    self.visit_ast(node.target.obj)
                    self.emit(Opcode.STORE_VAR, temp_obj)
                    self.visit_ast(node.target.index)
                    self.emit(Opcode.STORE_VAR, temp_idx)
                    self.emit(Opcode.LOAD_VAR, temp_obj)
                    self.emit(Opcode.LOAD_VAR, temp_idx)
                    self.emit(Opcode.LOAD_VAR, temp_obj)
                    self.emit(Opcode.LOAD_VAR, temp_idx)
                    self.emit(Opcode.GET_INDEX)
                    self.visit_ast(node.value)
                    self.emit(op_map[node.op])
                    self.emit(Opcode.SET_INDEX)
                else:
                    self.visit_ast(node.target.obj)
                    self.visit_ast(node.target.index)
                    self.visit_ast(node.value)
                    self.emit(Opcode.SET_INDEX)
            else:
                raise ValueError(f"Invalid assignment target: {node.target}")

        # Compatibility nodes for test_vm.py
        elif isinstance(node, StructAllocNode):
            for val in node.values:
                self.visit_ast(val)
            self.emit(Opcode.ALLOC_STRUCT, node.field_names)

        elif isinstance(node, StructGetNode):
            self.visit_ast(node.struct_expr)
            self.emit(Opcode.GET_FIELD, node.field_name)

        elif isinstance(node, StructSetNode):
            self.visit_ast(node.struct_expr)
            self.visit_ast(node.value_expr)
            self.emit(Opcode.SET_FIELD, node.field_name)

        elif isinstance(node, MapAllocNode):
            for k, v in zip(node.keys, node.values):
                self.visit_ast(k)
                self.visit_ast(v)
            self.emit(Opcode.ALLOC_MAP, len(node.keys))

        elif isinstance(node, MapGetNode):
            self.visit_ast(node.map_expr)
            self.visit_ast(node.key_expr)
            self.emit(Opcode.GET_INDEX)

        elif isinstance(node, MapSetNode):
            self.visit_ast(node.map_expr)
            self.visit_ast(node.key_expr)
            self.visit_ast(node.value_expr)
            self.emit(Opcode.SET_INDEX)

        elif isinstance(node, ArrayAllocNode):
            for elem in node.elements:
                self.visit_ast(elem)
            self.emit(Opcode.ALLOC_ARRAY, len(node.elements))

        elif isinstance(node, ArrayGetNode):
            self.visit_ast(node.array_expr)
            self.visit_ast(node.index_expr)
            self.emit(Opcode.GET_INDEX)

        elif isinstance(node, ArraySetNode):
            self.visit_ast(node.array_expr)
            self.visit_ast(node.index_expr)
            self.visit_ast(node.value_expr)
            self.emit(Opcode.SET_INDEX)

        elif isinstance(node, ParallelBlockNode):
            # Emit SPAWN for each task, then JOIN
            task_labels = []
            for task in node.tasks:
                if isinstance(task, TaskStatementNode):
                    call = task.call
                    if isinstance(call, CallNode):
                        callee_name = call.callee.name if isinstance(call.callee, VarRefNode) else str(call.callee)
                        for arg in call.args:
                            self.visit_ast(arg)
                        func_label = self.qfuncs.get(callee_name, callee_name)
                        self.emit(Opcode.SPAWN, (func_label, callee_name, len(call.args)))
                    elif isinstance(call, QFuncCallNode):
                        for arg_name in call.args:
                            self.emit(Opcode.LOAD_VAR, arg_name)
                        func_label = self.qfuncs.get(call.name, call.name)
                        self.emit(Opcode.SPAWN, (func_label, call.name, len(call.args)))
                else:
                    self.visit_ast(task)
            self.emit(Opcode.JOIN, len([t for t in node.tasks if isinstance(t, TaskStatementNode)]))

        elif isinstance(node, MatchNode):
            end_labels = []
            for pattern_expr, body in node.cases:
                self.visit_ast(node.expr)
                self.visit_ast(pattern_expr)
                self.emit(Opcode.EQ)
                case_label = Label("case_match")
                self.emit(Opcode.JMP_IF_TRUE, case_label)
                end_labels.append(Label("case_end"))
                self.emit(Opcode.JMP, end_labels[-1])
                self.emit_label(case_label)
                for stmt in body:
                    self.visit_ast(stmt)
                self.emit(Opcode.JMP, Label("match_end"))
            if node.default_body:
                for stmt in node.default_body:
                    self.visit_ast(stmt)
            match_end = Label("match_end")
            self.emit_label(match_end)
            for el in end_labels:
                self.emit_label(el)

        elif isinstance(node, StringInterpolationNode):
            if len(node.parts) == 0:
                self.emit(Opcode.LOAD_CONST, "")
            elif len(node.parts) == 1 and isinstance(node.parts[0], str):
                self.emit(Opcode.LOAD_CONST, node.parts[0])
            else:
                first = True
                for part in node.parts:
                    if isinstance(part, str):
                        self.emit(Opcode.LOAD_CONST, part)
                    else:
                        self.visit_ast(part)
                        # Convert to string at runtime
                        self.emit(Opcode.LOAD_CONST, "str")
                        self.emit(Opcode.CALL, (None, "std.string.format_int", 1))
                    if not first:
                        self.emit(Opcode.LOAD_CONST, "concat")
                        self.emit(Opcode.CALL, (None, "std.string.concat", 2))
                    first = False

        else:
            raise ValueError(f"Unknown AST node class: {type(node).__name__}")

    def compile_eqir(self, graph: EQIRGraph) -> list[Instruction]:
        self.raw_code = []
        nodes = graph.topological_sort()

        for node in nodes:
            skip_label = Label("skip_cond")
            if node.condition:
                cbit_name, op, expected_val = node.condition
                self.emit(Opcode.LOAD_VAR, cbit_name)
                self.emit(Opcode.LOAD_CONST, expected_val)
                if op == '==':
                    self.emit(Opcode.EQ)
                elif op == '!=':
                    self.emit(Opcode.NEQ)
                self.emit(Opcode.JMP_IF_FALSE, skip_label)

            # Node operation compilation
            if node.type == 'ALLOC':
                self.emit(Opcode.Q_ALLOC, node.targets[0])
            elif node.type == 'GATE':
                for arg in node.args:
                    if isinstance(arg, (int, float, complex)):
                        self.emit(Opcode.LOAD_CONST, arg)
                    elif isinstance(arg, str):
                        try:
                            tokens = Lexer(arg).tokenize()
                            if tokens and tokens[-1].type.name == 'EOF':
                                tokens.pop()
                            expr_node = Parser(tokens).parse_expr()
                            self.visit_ast(expr_node)
                        except Exception:
                            self.emit(Opcode.LOAD_CONST, arg)
                self.emit(Opcode.Q_GATE, (node.gate_name, node.targets))
            elif node.type == 'MEASURE':
                self.emit(Opcode.Q_MEASURE, (node.targets[0], node.cbit_name))
            elif node.type == 'TRACE':
                self.emit(Opcode.Q_TRACE)
            elif node.type == 'PRINT':
                if isinstance(node.print_expr, str):
                    self.emit(Opcode.LOAD_VAR, node.print_expr)
                else:
                    self.emit(Opcode.LOAD_CONST, node.print_expr)
                self.emit(Opcode.PRINT)
            elif node.type == 'ASSERT':
                left, op, right = node.assert_cond
                if isinstance(left, str):
                    self.emit(Opcode.LOAD_VAR, left)
                else:
                    self.emit(Opcode.LOAD_CONST, left)

                if isinstance(right, str):
                    self.emit(Opcode.LOAD_VAR, right)
                else:
                    self.emit(Opcode.LOAD_CONST, right)

                if op == '==':
                    self.emit(Opcode.EQ)
                elif op == '!=':
                    self.emit(Opcode.NEQ)

                ok_label = Label("assert_ok")
                self.emit(Opcode.JMP_IF_TRUE, ok_label)
                self.emit(Opcode.LOAD_CONST, f"Assertion Failed: {left} {op} {right}")
                self.emit(Opcode.THROW)
                self.emit_label(ok_label)

            if node.condition:
                self.emit_label(skip_label)

        self.emit(Opcode.HALT)
        return self.resolve_labels()

    def optimize_peephole_raw(self):
        if not self.peephole:
            return
        optimized = []
        i = 0
        n = len(self.raw_code)
        while i < n:
            item = self.raw_code[i]
            if isinstance(item, Label):
                optimized.append(item)
                i += 1
                continue
            
            # Pattern 1: LOAD_VAR + LOAD_CONST + ADD/SUB/LT/GT/LTE/GTE
            if (i + 2 < n and 
                isinstance(self.raw_code[i], Instruction) and 
                isinstance(self.raw_code[i+1], Instruction) and 
                isinstance(self.raw_code[i+2], Instruction)):
                
                inst1 = self.raw_code[i]
                inst2 = self.raw_code[i+1]
                inst3 = self.raw_code[i+2]
                
                if inst1.opcode == Opcode.LOAD_VAR and inst2.opcode == Opcode.LOAD_CONST:
                    if inst3.opcode == Opcode.ADD:
                        new_inst = Instruction(Opcode.LOAD_VAR_LOAD_CONST_ADD, (inst1.arg, inst2.arg))
                        new_inst.line = inst1.line
                        optimized.append(new_inst)
                        i += 3
                        continue
                    elif inst3.opcode == Opcode.SUB:
                        new_inst = Instruction(Opcode.LOAD_VAR_LOAD_CONST_SUB, (inst1.arg, inst2.arg))
                        new_inst.line = inst1.line
                        optimized.append(new_inst)
                        i += 3
                        continue
                    elif inst3.opcode == Opcode.LT:
                        new_inst = Instruction(Opcode.LOAD_VAR_LOAD_CONST_LT, (inst1.arg, inst2.arg))
                        new_inst.line = inst1.line
                        optimized.append(new_inst)
                        i += 3
                        continue
                    elif inst3.opcode == Opcode.GT:
                        new_inst = Instruction(Opcode.LOAD_VAR_LOAD_CONST_GT, (inst1.arg, inst2.arg))
                        new_inst.line = inst1.line
                        optimized.append(new_inst)
                        i += 3
                        continue
                    elif inst3.opcode == Opcode.LTE:
                        new_inst = Instruction(Opcode.LOAD_VAR_LOAD_CONST_LTE, (inst1.arg, inst2.arg))
                        new_inst.line = inst1.line
                        optimized.append(new_inst)
                        i += 3
                        continue
                    elif inst3.opcode == Opcode.GTE:
                        new_inst = Instruction(Opcode.LOAD_VAR_LOAD_CONST_GTE, (inst1.arg, inst2.arg))
                        new_inst.line = inst1.line
                        optimized.append(new_inst)
                        i += 3
                        continue

            # Pattern 2: LOAD_CONST + STORE_VAR
            if (i + 1 < n and 
                isinstance(self.raw_code[i], Instruction) and 
                isinstance(self.raw_code[i+1], Instruction)):
                
                inst1 = self.raw_code[i]
                inst2 = self.raw_code[i+1]
                
                if inst1.opcode == Opcode.LOAD_CONST and inst2.opcode == Opcode.STORE_VAR:
                    new_inst = Instruction(Opcode.LOAD_CONST_STORE, (inst1.arg, inst2.arg))
                    new_inst.line = inst1.line
                    optimized.append(new_inst)
                    i += 2
                    continue
            
            optimized.append(item)
            i += 1
            
        self.raw_code = optimized

    def resolve_labels(self) -> list[Instruction]:
        self.optimize_peephole_raw()
        final_instructions = []
        label_to_index = {}

        # First pass: map label positions and filter labels out of the final list
        for item in self.raw_code:
            if isinstance(item, Label):
                label_to_index[item] = len(final_instructions)
            else:
                final_instructions.append(item)

        # Second pass: update the arg of instructions that jump to labels
        for instr in final_instructions:
            if instr.opcode in (Opcode.CALL, Opcode.SPAWN):
                func_target, func_name, num_args = instr.arg
                if isinstance(func_target, Label):
                    resolved_idx = label_to_index.get(func_target)
                    if resolved_idx is None:
                        raise ValueError(f"Unresolved label: {func_target}")
                    instr.arg = (resolved_idx, func_name, num_args)
            elif instr.opcode in (Opcode.JMP, Opcode.JMP_IF_FALSE, Opcode.JMP_IF_TRUE, Opcode.PUSH_TRY):
                if isinstance(instr.arg, Label):
                    resolved_idx = label_to_index.get(instr.arg)
                    if resolved_idx is None:
                        raise ValueError(f"Unresolved label: {instr.arg}")
                    instr.arg = resolved_idx

        self.resolved_qfuncs = {name: label_to_index[label] for name, label in self.qfuncs.items() if label in label_to_index}

        return final_instructions
