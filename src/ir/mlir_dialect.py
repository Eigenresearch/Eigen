"""MLIR Dialect Layer for Eigen 2.3 — Helios.

Defines MLIR structure (Module, Function, Block, Op, Value) and provides
translation passes: AST -> MLIR Dialect -> EQIR.
"""

from src.frontend.ast import (
    ASTNode, ProgramNode, QFuncDeclNode, LetNode, VarDeclNode,
    BinaryOpNode, LiteralNode, VarRefNode, QFuncCallNode, GateNode,
    MeasureNode, IfNode, ReturnNode, TraceNode, PrintNode, AssertNode,
    FuncDeclNode, CallNode, ParallelBlockNode, TaskStatementNode
)
from src.ir.ir_graph import EQIRGraph, EQIRNode


class MLIRValue:
    def __init__(self, name: str, value_type: str):
        self.name = name
        self.value_type = value_type

    def __repr__(self) -> str:
        return f"%{self.name} : {self.value_type}"


class MLIROp:
    def __init__(self, op_name: str, operands: list[MLIRValue] = None, results: list[MLIRValue] = None, attributes: dict = None, condition=None):
        self.op_name = op_name
        self.operands = operands or []
        self.results = results or []
        self.attributes = attributes or {}
        self.condition = condition  # None or (cbit_name, op, val) for conditional execution

    def __repr__(self) -> str:
        res_str = ", ".join(repr(r) for r in self.results)
        res_prefix = f"{res_str} = " if self.results else ""
        ops_str = ", ".join(r.name if isinstance(r, MLIRValue) else str(r) for r in self.operands)
        attrs_str = f" {self.attributes}" if self.attributes else ""
        cond_str = f" [cond: {self.condition}]" if self.condition else ""
        return f"{res_prefix}{self.op_name}({ops_str}){attrs_str}{cond_str}"


class MLIRBlock:
    def __init__(self, label: str = "^bb0"):
        self.label = label
        self.operations: list[MLIROp] = []

    def add_operation(self, op: MLIROp):
        self.operations.append(op)

    def __repr__(self) -> str:
        ops_str = "\n  ".join(repr(op) for op in self.operations)
        return f"{self.label}:\n  {ops_str}"


class MLIRFunction:
    def __init__(self, name: str, params: list[tuple[str, str]] = None, return_types: list[str] = None):
        self.name = name
        self.params = params or []  # List of (name, type)
        self.return_types = return_types or []
        self.blocks: list[MLIRBlock] = [MLIRBlock("^bb0")]

    def add_op(self, op: MLIROp, block_idx: int = 0):
        while len(self.blocks) <= block_idx:
            self.blocks.append(MLIRBlock(f"^bb{len(self.blocks)}"))
        self.blocks[block_idx].add_operation(op)

    def __repr__(self) -> str:
        params_str = ", ".join(f"%{name}: {t}" for name, t in self.params)
        ret_str = f" -> {', '.join(self.return_types)}" if self.return_types else ""
        blocks_str = "\n".join(repr(b) for b in self.blocks)
        return f"func.func @{self.name}({params_str}){ret_str} {{\n{blocks_str}\n}}"


class MLIRModule:
    def __init__(self):
        self.functions: list[MLIRFunction] = []

    def add_function(self, func: MLIRFunction):
        self.functions.append(func)

    def __repr__(self) -> str:
        return "\n\n".join(repr(f) for f in self.functions)


class ASTToMLIRConverter:
    def __init__(self):
        self.module = MLIRModule()
        self.temp_counter = 0
        self.current_function = None
        self.current_block_idx = 0
        self.current_condition = None

    def get_temp(self, prefix: str = "v") -> str:
        self.temp_counter += 1
        return f"{prefix}{self.temp_counter}"

    def convert(self, program: ProgramNode) -> MLIRModule:
        # Create a main function for the program execution root
        main_func = MLIRFunction("main")
        self.current_function = main_func
        self.current_block_idx = 0
        self.module.add_function(main_func)

        # First process all non-function decls into the main block, and register function decls as separate functions
        for node in program.body:
            if isinstance(node, QFuncDeclNode):
                # QFunc Decl
                qfunc = MLIRFunction(node.name, params=node.params)
                prev_func = self.current_function
                prev_block = self.current_block_idx
                self.current_function = qfunc
                self.current_block_idx = 0
                
                for stmt in node.body:
                    self.convert_node(stmt)
                
                self.module.add_function(qfunc)
                self.current_function = prev_func
                self.current_block_idx = prev_block
            elif isinstance(node, FuncDeclNode):
                # Classic Func Decl
                cfunc = MLIRFunction(node.name, params=node.params, return_types=[node.return_type])
                prev_func = self.current_function
                prev_block = self.current_block_idx
                self.current_function = cfunc
                self.current_block_idx = 0
                
                for stmt in node.body:
                    self.convert_node(stmt)
                
                self.module.add_function(cfunc)
                self.current_function = prev_func
                self.current_block_idx = prev_block
            else:
                # Main body statement
                self.convert_node(node)

        return self.module

    def convert_node(self, node: ASTNode) -> MLIRValue | None:
        if isinstance(node, VarDeclNode):
            if node.type_name == "qubit":
                res = MLIRValue(node.name, "!quantum.qubit")
                op = MLIROp("quantum.alloc", results=[res], condition=self.current_condition)
                self.current_function.add_op(op, self.current_block_idx)
                return res
            else:
                res = MLIRValue(node.name, node.type_name)
                op = MLIROp("memref.alloc", results=[res], condition=self.current_condition)
                self.current_function.add_op(op, self.current_block_idx)
                return res

        elif isinstance(node, LetNode):
            val_val = self.convert_node(node.value)
            if val_val:
                res = MLIRValue(node.name, node.type_name)
                op = MLIROp("memref.store", operands=[val_val, res], condition=self.current_condition)
                self.current_function.add_op(op, self.current_block_idx)
                return res

        elif isinstance(node, LiteralNode):
            res = MLIRValue(self.get_temp("c"), node.type_name)
            op = MLIROp("arith.constant", attributes={"value": node.value}, results=[res], condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return res

        elif isinstance(node, VarRefNode):
            # Treat as MLIR value
            return MLIRValue(node.name, "unknown")

        elif isinstance(node, BinaryOpNode):
            l_val = self.convert_node(node.left)
            r_val = self.convert_node(node.right)
            res = MLIRValue(self.get_temp("t"), "unknown")
            op_name = f"arith.{node.op}"
            if node.op == '+': op_name = "arith.addi"
            elif node.op == '-': op_name = "arith.subi"
            elif node.op == '*': op_name = "arith.muli"
            elif node.op == '/': op_name = "arith.divi"
            elif node.op == '**': op_name = "math.pow"
            elif node.op == '%': op_name = "arith.remi"
            op = MLIROp(op_name, operands=[l_val, r_val], results=[res], condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return res

        elif isinstance(node, GateNode):
            operands = [self.convert_node(VarRefNode(t)) for t in node.targets]
            # Convert rotation angle arguments
            arg_vals = []
            for arg in node.args:
                arg_val = self.convert_node(arg)
                if arg_val:
                    arg_vals.append(arg_val)
            
            op = MLIROp("quantum.gate", operands=operands + arg_vals, attributes={"gate": node.gate_name}, condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return None

        elif isinstance(node, MeasureNode):
            q_val = MLIRValue(node.qubit_name, "!quantum.qubit")
            c_val = MLIRValue(node.cbit_name, "!quantum.cbit")
            op = MLIROp("quantum.measure", operands=[q_val], results=[c_val], condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return c_val

        elif isinstance(node, QFuncCallNode):
            args = [MLIRValue(arg, "unknown") for arg in node.args]
            op = MLIROp("func.call", operands=args, attributes={"callee": node.name}, condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return None

        elif isinstance(node, CallNode):
            callee_name = node.callee.name if isinstance(node.callee, VarRefNode) else str(node.callee)
            args = []
            for arg in node.args:
                arg_val = self.convert_node(arg)
                if arg_val:
                    args.append(arg_val)
            op = MLIROp("func.call", operands=args, attributes={"callee": callee_name}, condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return None

        elif isinstance(node, ParallelBlockNode):
            # Compile task spawning structure
            tasks = []
            for task in node.tasks:
                if isinstance(task, TaskStatementNode):
                    callee_name = task.call.callee
                    if isinstance(callee_name, VarRefNode):
                        callee_name = callee_name.name
                    tasks.append(f"task_{callee_name}")
                elif isinstance(task, CallNode):
                    callee_name = task.callee
                    if isinstance(callee_name, VarRefNode):
                        callee_name = callee_name.name
                    tasks.append(f"task_{callee_name}")
            
            op = MLIROp("cf.parallel", attributes={"tasks": tasks}, condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return None

        elif isinstance(node, IfNode):
            # Parse condition
            l_val = self.convert_node(node.condition_left)
            r_val = self.convert_node(node.condition_right)
            
            # Setup current condition block-level filtering
            prev_cond = self.current_condition
            cbit_name = l_val.name if l_val else "unknown_cbit"
            expected_val = r_val.attributes.get("value") if r_val and hasattr(r_val, "attributes") else 0
            # If r_val was created by arith.constant op, retrieve attribute
            self.current_condition = (cbit_name, node.op, expected_val)

            # Compile body statements under this condition
            for stmt in node.body:
                self.convert_node(stmt)
                
            if hasattr(node, "else_body") and node.else_body:
                opp_map = {
                    "==": "!=",
                    "!=": "==",
                    "<": ">=",
                    ">": "<=",
                    "<=": ">",
                    ">=": "<",
                }
                self.current_condition = (cbit_name, opp_map.get(node.op, "!="), expected_val)
                for stmt in node.else_body:
                    self.convert_node(stmt)
 
            self.current_condition = prev_cond
            return None

        elif isinstance(node, TraceNode):
            op = MLIROp("quantum.trace", condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return None

        elif isinstance(node, PrintNode):
            expr_val = self.convert_node(node.expr)
            op = MLIROp("func.print", operands=[expr_val] if expr_val else [], condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return None

        elif isinstance(node, AssertNode):
            l_val = self.convert_node(node.condition_left)
            r_val = self.convert_node(node.condition_right)
            op = MLIROp("cf.assert", operands=[l_val, r_val] if l_val and r_val else [], attributes={"op": node.op}, condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return None

        elif isinstance(node, ReturnNode):
            expr_val = self.convert_node(node.expr) if node.expr else None
            op = MLIROp("func.return", operands=[expr_val] if expr_val else [], condition=self.current_condition)
            self.current_function.add_op(op, self.current_block_idx)
            return None

        return None


class MLIRToEQIRConverter:
    def __init__(self):
        self.graph = EQIRGraph()
        self.qfuncs = {}  # name -> func definition for inlining
        self.constants = {} # name -> value

    def convert(self, module: MLIRModule) -> EQIRGraph:
        # 1. Register all functions first (for inlining calls)
        for func in module.functions:
            if func.name != "main":
                self.qfuncs[func.name] = func

        # 2. Convert main function
        main_func = next((f for f in module.functions if f.name == "main"), None)
        if main_func:
            self.convert_function(main_func)
            
        return self.graph

    def convert_function(self, func: MLIRFunction, param_map: dict = None):
        if param_map is None:
            param_map = {}

        def resolve_name(name: str) -> str:
            return param_map.get(name, name)

        for block in func.blocks:
            for op in block.operations:
                cond = op.condition
                if cond:
                    cond_cbit, cond_op, cond_val = cond
                    cond = (resolve_name(cond_cbit), cond_op, cond_val)

                if op.op_name == "arith.constant":
                    const_val = op.attributes.get("value")
                    res_name = op.results[0].name
                    self.constants[resolve_name(res_name)] = const_val

                elif op.op_name == "quantum.alloc":
                    qname = resolve_name(op.results[0].name)
                    self.graph.add_operation('ALLOC', targets=[qname], condition=cond)

                elif op.op_name == "quantum.gate":
                    gate_name = op.attributes.get("gate", "H")
                    targets = [resolve_name(oper.name) for oper in op.operands]
                    
                    # Separate gate targets from angles
                    gate_targets = []
                    args = []
                    for t in targets:
                        resolved_t = resolve_name(t)
                        if resolved_t in self.constants:
                            args.append(self.constants[resolved_t])
                        elif t.startswith("c") or t.startswith("v") or t.isdigit():
                            # If it's a numeric constant or temporary variable, parse it as arg
                            try:
                                args.append(float(t))
                            except ValueError:
                                args.append(t)
                        else:
                            gate_targets.append(t)
                            
                    self.graph.add_operation(
                        'GATE',
                        gate_name=gate_name,
                        targets=gate_targets,
                        args=args,
                        condition=cond
                    )

                elif op.op_name == "quantum.measure":
                    qname = resolve_name(op.operands[0].name)
                    cname = resolve_name(op.results[0].name)
                    self.graph.add_operation('MEASURE', targets=[qname], cbit_name=cname, condition=cond)

                elif op.op_name == "quantum.trace":
                    self.graph.add_operation('TRACE', condition=cond)

                elif op.op_name == "func.print":
                    print_val = resolve_name(op.operands[0].name) if op.operands else ""
                    self.graph.add_operation('PRINT', print_expr=print_val, condition=cond)

                elif op.op_name == "cf.assert":
                    l_val = resolve_name(op.operands[0].name) if len(op.operands) > 0 else "true"
                    r_val = resolve_name(op.operands[1].name) if len(op.operands) > 1 else "true"
                    assert_op = op.attributes.get("op", "==")
                    self.graph.add_operation(
                        'ASSERT',
                        assert_cond=(l_val, assert_op, r_val),
                        condition=cond
                    )

                elif op.op_name == "func.call":
                    callee = op.attributes.get("callee")
                    if callee in self.qfuncs:
                        target_func = self.qfuncs[callee]
                        # Map params to operands
                        local_map = {}
                        for param, operand in zip(target_func.params, op.operands):
                            local_map[param[0]] = resolve_name(operand.name)
                        # Inline target function operations
                        self.convert_function(target_func, local_map)
