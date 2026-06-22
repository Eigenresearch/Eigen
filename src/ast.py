class ASTNode:
    pass

class ProgramNode(ASTNode):
    def __init__(self, version: float, module_name: str | None, imports: list['ImportNode'], body: list[ASTNode]):
        self.version = version
        self.module_name = module_name
        self.imports = imports
        self.body = body

    def __repr__(self):
        return f"ProgramNode(version={self.version}, module={self.module_name}, imports={self.imports}, body={self.body})"

class ImportNode(ASTNode):
    def __init__(self, module_path: str):
        self.module_path = module_path

    def __repr__(self):
        return f"ImportNode({self.module_path})"

class QFuncDeclNode(ASTNode):
    def __init__(self, name: str, params: list[tuple[str, str]], body: list[ASTNode]):
        self.name = name
        self.params = params  # List of (name, type_name)
        self.body = body

    def __repr__(self):
        return f"QFuncDeclNode({self.name}, params={self.params}, body={self.body})"

class LetNode(ASTNode):
    def __init__(self, name: str, type_name: str, value: ASTNode):
        self.name = name
        self.type_name = type_name
        self.value = value

    def __repr__(self):
        return f"LetNode({self.name}: {self.type_name} = {self.value})"

class VarDeclNode(ASTNode):
    def __init__(self, name: str, type_name: str):
        self.name = name
        self.type_name = type_name

    def __repr__(self):
        return f"VarDeclNode({self.name}: {self.type_name})"

class BinaryOpNode(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"BinaryOpNode({self.left} {self.op} {self.right})"

class LiteralNode(ASTNode):
    def __init__(self, value: float | int | str, type_name: str):
        self.value = value
        self.type_name = type_name

    def __repr__(self):
        return f"LiteralNode({self.value}: {self.type_name})"

class VarRefNode(ASTNode):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"VarRefNode({self.name})"

class QFuncCallNode(ASTNode):
    def __init__(self, name: str, args: list[str]):
        self.name = name
        self.args = args  # List of identifier strings

    def __repr__(self):
        return f"QFuncCallNode({self.name}, args={self.args})"

class GateNode(ASTNode):
    def __init__(self, gate_name: str, targets: list[str], args: list[ASTNode]):
        self.gate_name = gate_name
        self.targets = targets  # List of target/control qubit names
        self.args = args        # Expressions for rotation angles (if any)

    def __repr__(self):
        return f"GateNode({self.gate_name}, targets={self.targets}, args={self.args})"

class MeasureNode(ASTNode):
    def __init__(self, qubit_name: str, cbit_name: str):
        self.qubit_name = qubit_name
        self.cbit_name = cbit_name

    def __repr__(self):
        return f"MeasureNode({self.qubit_name} -> {self.cbit_name})"

class IfNode(ASTNode):
    def __init__(self, condition_left: ASTNode, op: str, condition_right: ASTNode, body: list[ASTNode]):
        self.condition_left = condition_left
        self.op = op  # "=="
        self.condition_right = condition_right
        self.body = body

    def __repr__(self):
        return f"IfNode({self.condition_left} {self.op} {self.condition_right}, body={self.body})"

class ReturnNode(ASTNode):
    def __repr__(self):
        return "ReturnNode()"

class TraceNode(ASTNode):
    def __repr__(self):
        return "TraceNode()"

class PrintNode(ASTNode):
    def __init__(self, expr: ASTNode):
        self.expr = expr

    def __repr__(self):
        return f"PrintNode({self.expr})"

class AssertNode(ASTNode):
    def __init__(self, condition_left: ASTNode, op: str, condition_right: ASTNode):
        self.condition_left = condition_left
        self.op = op
        self.condition_right = condition_right

    def __repr__(self):
        return f"AssertNode({self.condition_left} {self.op} {self.condition_right})"
