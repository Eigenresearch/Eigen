class Opcode:
    # Arithmetic and Comparisons
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    GT = "GT"
    LTE = "LTE"
    GTE = "GTE"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # Load/Store values and variables
    LOAD_CONST = "LOAD_CONST"
    LOAD_VAR = "LOAD_VAR"
    STORE_VAR = "STORE_VAR"

    # Scoping and function execution
    CALL = "CALL"
    RET = "RET"
    ENTER_FRAME = "ENTER_FRAME"
    EXIT_FRAME = "EXIT_FRAME"

    # Structs
    ALLOC_STRUCT = "ALLOC_STRUCT"
    GET_FIELD = "GET_FIELD"
    SET_FIELD = "SET_FIELD"

    # Maps and Arrays
    ALLOC_MAP = "ALLOC_MAP"
    ALLOC_ARRAY = "ALLOC_ARRAY"
    LEN = "LEN"
    GET_INDEX = "GET_INDEX"
    SET_INDEX = "SET_INDEX"

    # Exceptions
    THROW = "THROW"
    PUSH_TRY = "PUSH_TRY"
    POP_TRY = "POP_TRY"

    # Quantum operations
    Q_ALLOC = "Q_ALLOC"
    Q_GATE = "Q_GATE"
    Q_MEASURE = "Q_MEASURE"
    Q_TRACE = "Q_TRACE"
    Q_NOISE = "Q_NOISE"

    # Control
    JMP = "JMP"
    JMP_IF_FALSE = "JMP_IF_FALSE"
    JMP_IF_TRUE = "JMP_IF_TRUE"
    HALT = "HALT"

    # Parallel execution
    SPAWN = "SPAWN"
    JOIN = "JOIN"

    # Additional helpers
    PRINT = "PRINT"  # Convenient opcode for printing top of stack


class Instruction:
    def __init__(self, opcode: str, arg=None, line: int = None):
        self.opcode = opcode
        self.arg = arg
        self.line = line

    def __repr__(self) -> str:
        arg_str = f" {self.arg}" if self.arg is not None else ""
        line_str = f" (line {self.line})" if self.line is not None else ""
        return f"{self.opcode}{arg_str}{line_str}"

    def to_dict(self) -> dict:
        return {
            "opcode": self.opcode,
            "arg": self.arg,
            "line": self.line
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Instruction":
        return cls(data["opcode"], data.get("arg"), data.get("line"))
