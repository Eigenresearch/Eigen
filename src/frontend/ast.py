import abc

class ASTNode(metaclass=abc.ABCMeta):
    def to_source(self) -> str:
        return repr(self)

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

    def to_source(self) -> str:
        return f"{self.left.to_source()} {self.op} {self.right.to_source()}"

class LiteralNode(ASTNode):
    def __init__(self, value: float | int | str, type_name: str):
        self.value = value
        self.type_name = type_name

    def __repr__(self):
        return f"LiteralNode({self.value}: {self.type_name})"

    def to_source(self) -> str:
        if self.type_name == "string":
            return f'"{self.value}"'
        return str(self.value)

class VarRefNode(ASTNode):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"VarRefNode({self.name})"

    def to_source(self) -> str:
        return self.name

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
    def __init__(self, condition_left: ASTNode, op: str, condition_right: ASTNode, body: list[ASTNode], else_body: list[ASTNode] | None = None):
        self.condition_left = condition_left
        self.op = op  # "=="
        self.condition_right = condition_right
        self.body = body
        self.else_body = else_body if else_body is not None else []

    def __repr__(self):
        return f"IfNode({self.condition_left} {self.op} {self.condition_right}, body={self.body}, else_body={self.else_body})"

class ReturnNode(ASTNode):
    def __init__(self, expr: ASTNode | None = None):
        self.expr = expr

    def __repr__(self):
        return f"ReturnNode({self.expr})"

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

class FuncDeclNode(ASTNode):
    def __init__(self, name: str, generic_params: list[str], params: list[tuple[str, str]], return_type: str, body: list[ASTNode]):
        self.name = name
        self.generic_params = generic_params  # e.g. ["T"]
        self.params = params  # list of (param_name, type_name)
        self.return_type = return_type  # e.g. "int"
        self.body = body

    def __repr__(self):
        return f"FuncDeclNode({self.name}, generics={self.generic_params}, params={self.params}, return={self.return_type}, body={self.body})"

class ForNode(ASTNode):
    def __init__(self, variable: str, iterable: ASTNode, body: list[ASTNode]):
        self.variable = variable
        self.iterable = iterable
        self.body = body

    def __repr__(self):
        return f"ForNode(var={self.variable}, iter={self.iterable}, body={self.body})"

class WhileNode(ASTNode):
    def __init__(self, condition: ASTNode, body: list[ASTNode]):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileNode(cond={self.condition}, body={self.body})"

class BreakNode(ASTNode):
    def __repr__(self):
        return "BreakNode()"

class ContinueNode(ASTNode):
    def __repr__(self):
        return "ContinueNode()"

class StructDeclNode(ASTNode):
    def __init__(self, name: str, generic_params: list[str], fields: list[tuple[str, str]]):
        self.name = name
        self.generic_params = generic_params  # e.g. ["T"]
        self.fields = fields  # list of (field_name, field_type)

    def __repr__(self):
        return f"StructDeclNode({self.name}, generics={self.generic_params}, fields={self.fields})"

class StructLiteralNode(ASTNode):
    def __init__(self, struct_name: str, field_bindings: dict[str, ASTNode] | list[tuple[str, ASTNode]]):
        self.struct_name = struct_name
        self.field_bindings = field_bindings

    def __repr__(self):
        return f"StructLiteralNode({self.struct_name}, bindings={self.field_bindings})"

class DotAccessNode(ASTNode):
    def __init__(self, obj: ASTNode, member: str):
        self.obj = obj
        self.member = member

    def __repr__(self):
        return f"DotAccessNode({self.obj}.{self.member})"

    def to_source(self) -> str:
        return f"{self.obj.to_source()}.{self.member}"

class ArrayLiteralNode(ASTNode):
    def __init__(self, elements: list[ASTNode]):
        self.elements = elements

    def __repr__(self):
        return f"ArrayLiteralNode({self.elements})"

class TupleLiteralNode(ASTNode):
    def __init__(self, elements: list[ASTNode]):
        self.elements = elements

    def __repr__(self):
        return f"TupleLiteralNode({self.elements})"

class TryCatchNode(ASTNode):
    def __init__(self, try_body: list[ASTNode], catch_var: str | None, catch_body: list[ASTNode]):
        self.try_body = try_body
        self.catch_var = catch_var
        self.catch_body = catch_body

    def __repr__(self):
        return f"TryCatchNode(try={self.try_body}, catch_var={self.catch_var}, catch={self.catch_body})"

class ThrowNode(ASTNode):
    def __init__(self, expr: ASTNode):
        self.expr = expr

    def __repr__(self):
        return f"ThrowNode({self.expr})"

class EnumDeclNode(ASTNode):
    def __init__(self, name: str, variants: list[str]):
        self.name = name
        self.variants = variants  # list of variant names

    def __repr__(self):
        return f"EnumDeclNode({self.name}, variants={self.variants})"

class NoiseNode(ASTNode):
    def __init__(self, noise_type: str, expr: ASTNode, targets: list[str]):
        self.noise_type = noise_type
        self.expr = expr
        self.targets = targets

    def __repr__(self):
        return f"NoiseNode({self.noise_type}({self.expr}) targets={self.targets})"

class AssignmentNode(ASTNode):
    def __init__(self, target: ASTNode, op: str, value: ASTNode):
        self.target = target
        self.op = op
        self.value = value

    def __repr__(self):
        return f"AssignmentNode({self.target} {self.op} {self.value})"

class CallNode(ASTNode):
    def __init__(self, callee: str | ASTNode, args: list[ASTNode]):
        self.callee = callee
        self.args = args

    def __repr__(self):
        return f"CallNode({self.callee}, args={self.args})"

    def to_source(self) -> str:
        callee_str = self.callee.to_source() if isinstance(self.callee, ASTNode) else str(self.callee)
        args_str = ", ".join(a.to_source() if isinstance(a, ASTNode) else str(a) for a in self.args)
        return f"{callee_str}({args_str})"

class IndexAccessNode(ASTNode):
    def __init__(self, obj: ASTNode, index: ASTNode):
        self.obj = obj
        self.index = index

    def __repr__(self):
        return f"IndexAccessNode({self.obj}[{self.index}])"

    def to_source(self) -> str:
        return f"{self.obj.to_source()}[{self.index.to_source()}]"

class MapAllocNode(ASTNode):
    def __init__(self, keys: list[ASTNode], values: list[ASTNode]):
        self.keys = keys
        self.values = values

    def __repr__(self):
        return f"MapAllocNode(keys={self.keys}, values={self.values})"

class StructAllocNode(ASTNode):
    def __init__(self, field_names: list[str], values: list[ASTNode]):
        self.field_names = field_names
        self.values = values

class StructGetNode(ASTNode):
    def __init__(self, struct_expr: ASTNode, field_name: str):
        self.struct_expr = struct_expr
        self.field_name = field_name

class StructSetNode(ASTNode):
    def __init__(self, struct_expr: ASTNode, field_name: str, value_expr: ASTNode):
        self.struct_expr = struct_expr
        self.field_name = field_name
        self.value_expr = value_expr

class MapGetNode(ASTNode):
    def __init__(self, map_expr: ASTNode, key_expr: ASTNode):
        self.map_expr = map_expr
        self.key_expr = key_expr

class MapSetNode(ASTNode):
    def __init__(self, map_expr: ASTNode, key_expr: ASTNode, value_expr: ASTNode):
        self.map_expr = map_expr
        self.key_expr = key_expr
        self.value_expr = value_expr

class ArrayAllocNode(ASTNode):
    def __init__(self, elements: list[ASTNode]):
        self.elements = elements

class ArrayGetNode(ASTNode):
    def __init__(self, array_expr: ASTNode, index_expr: ASTNode):
        self.array_expr = array_expr
        self.index_expr = index_expr

class ArraySetNode(ASTNode):
    def __init__(self, array_expr: ASTNode, index_expr: ASTNode, value_expr: ASTNode):
        self.array_expr = array_expr
        self.index_expr = index_expr
        self.value_expr = value_expr

class ParallelBlockNode(ASTNode):
    def __init__(self, tasks: list[ASTNode]):
        self.tasks = tasks

    def __repr__(self):
        return f"ParallelBlockNode(tasks={self.tasks})"

class TaskStatementNode(ASTNode):
    def __init__(self, call: ASTNode):
        self.call = call

    def __repr__(self):
        return f"TaskStatementNode(call={self.call})"

class MatchNode(ASTNode):
    def __init__(self, expr: ASTNode, cases: list[tuple[ASTNode, list[ASTNode]]], default_body: list[ASTNode] | None = None):
        self.expr = expr
        self.cases = cases  # list of (pattern_expr, body)
        self.default_body = default_body or []

    def __repr__(self):
        return f"MatchNode(expr={self.expr}, cases={len(self.cases)}, default={bool(self.default_body)})"

class StringInterpolationNode(ASTNode):
    def __init__(self, parts: list):
        self.parts = parts  # list of str or ASTNode

    def __repr__(self):
        return f"StringInterpolationNode(parts={self.parts})"


# §3.1 — Trait/Interface System (partial: AST/parser only).

class TraitMethodSignatureNode(ASTNode):
    """A single method signature inside a trait declaration.

    Reuses FuncDeclNode's parameter/return-type shape but with an empty
    body, since traits only declare method *signatures*, not
    implementations. The parser may later attach a default body for
    optional methods — for now `body` is always `[]`.
    """
    def __init__(self, name: str, generic_params: list[str],
                 params: list[tuple[str, str]], return_type: str):
        self.name = name
        self.generic_params = generic_params
        self.params = params
        self.return_type = return_type
        self.body = []  # trait signatures have no default body

    def __repr__(self):
        return (f"TraitMethodSignatureNode({self.name}, "
                f"generics={self.generic_params}, "
                f"params={self.params}, return={self.return_type})")


class TraitDeclNode(ASTNode):
    """A trait declaration: `trait Foo { fn bar(...) -> ...; ... }`.

    Traits collect a set of method signatures. Concrete types implement
    them via `ImplBlockNode`. Trait bounds on functions / structs are
    not enforced at type-check time in this partial P2 implementation —
    the goal is to get the language surface in place so user code can
    cite traits by name in impl blocks and downstream tools (a future
    Rust-type-check system) can verify conformance.
    """
    def __init__(self, name: str, generic_params: list[str],
                 methods: list["TraitMethodSignatureNode"]):
        self.name = name
        self.generic_params = generic_params
        self.methods = methods

    def __repr__(self):
        return (f"TraitDeclNode({self.name}, "
                f"generics={self.generic_params}, "
                f"methods={len(self.methods)})")

    def method_names(self) -> set:
        return {m.name for m in self.methods}


class ImplBlockNode(ASTNode):
    """An `impl Trait for Type { ... }` block.

    Carries the bound trait name (or `None` when the impl is not
    constrained, i.e. `impl Type { ... }` as in Rust's "inherent impl"),
    the target type name being provided with methods, and the function
    declarations that fulfill the trait's method signatures.
    """
    def __init__(self, trait_name: str | None,
                 target_type: str,
                 methods: list["FuncDeclNode"]):
        self.trait_name = trait_name
        self.target_type = target_type
        self.methods = methods

    def __repr__(self):
        if self.trait_name is None:
            return f"ImplBlockNode(inherent for {self.target_type}, methods={len(self.methods)})"
        return (f"ImplBlockNode(impl {self.trait_name} for "
                f"{self.target_type}, methods={len(self.methods)})")


class TypeAliasDeclNode(ASTNode):
    """§3.3 — A `type Name = Target;` declaration.

    Carries the alias name and the target type as a raw string (parsed
    loosely — anything after `=` up to a `;` or newline block). The type
    checker substitutes the alias name for the target type lazily at
    every type-reference site, allowing users to give meaningful names
    to complex composite types like `(Qubit, Qubit)` or `Map<string,
    int>` without committing to a specific structural shape in source.

    This is a P2 surface-level extension: aliases are stored and
    resolved textually. Higher-kinded / generic aliases (`type
    Pair[T] = (T, T)`) are NOT supported yet — generic aliases would
    require either AST rewriting or a substitution context.
    """
    def __init__(self, name: str, target_type: str):
        self.name = name
        self.target_type = target_type

    def __repr__(self):
        return f"TypeAliasDeclNode({self.name} = {self.target_type})"


try:
    import eigen_native
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
