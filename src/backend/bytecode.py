class UnsupportedBytecodeVersionError(Exception):
    pass


class Opcode:
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

    # Modulo and Bitwise opcodes
    MOD = "MOD"
    POW = "POW"
    BIT_AND = "BIT_AND"
    BIT_OR = "BIT_OR"
    BIT_XOR = "BIT_XOR"
    BIT_NOT = "BIT_NOT"
    SHL = "SHL"
    SHR = "SHR"

    # Superinstructions
    LOAD_CONST_STORE = "LOAD_CONST_STORE"
    LOAD_VAR_LOAD_CONST_ADD = "LOAD_VAR_LOAD_CONST_ADD"
    LOAD_VAR_LOAD_CONST_SUB = "LOAD_VAR_LOAD_CONST_SUB"
    LOAD_VAR_LOAD_CONST_LT = "LOAD_VAR_LOAD_CONST_LT"
    LOAD_VAR_LOAD_CONST_GT = "LOAD_VAR_LOAD_CONST_GT"
    LOAD_VAR_LOAD_CONST_LTE = "LOAD_VAR_LOAD_CONST_LTE"
    LOAD_VAR_LOAD_CONST_GTE = "LOAD_VAR_LOAD_CONST_GTE"


# Define a stable list order for all string opcodes
OPCODE_LIST = [
    Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV,
    Opcode.EQ, Opcode.NEQ, Opcode.LT, Opcode.GT, Opcode.LTE, Opcode.GTE,
    Opcode.AND, Opcode.OR, Opcode.NOT,
    Opcode.LOAD_CONST, Opcode.LOAD_VAR, Opcode.STORE_VAR,
    Opcode.CALL, Opcode.RET, Opcode.ENTER_FRAME, Opcode.EXIT_FRAME,
    Opcode.ALLOC_STRUCT, Opcode.GET_FIELD, Opcode.SET_FIELD,
    Opcode.ALLOC_MAP, Opcode.ALLOC_ARRAY, Opcode.LEN, Opcode.GET_INDEX, Opcode.SET_INDEX,
    Opcode.THROW, Opcode.PUSH_TRY, Opcode.POP_TRY,
    Opcode.Q_ALLOC, Opcode.Q_GATE, Opcode.Q_MEASURE, Opcode.Q_TRACE, Opcode.Q_NOISE,
    Opcode.JMP, Opcode.JMP_IF_FALSE, Opcode.JMP_IF_TRUE, Opcode.HALT,
    Opcode.SPAWN, Opcode.JOIN, Opcode.PRINT,
    Opcode.MOD, Opcode.POW, Opcode.BIT_AND, Opcode.BIT_OR, Opcode.BIT_XOR, Opcode.BIT_NOT, Opcode.SHL, Opcode.SHR,
    Opcode.LOAD_CONST_STORE,
    Opcode.LOAD_VAR_LOAD_CONST_ADD, Opcode.LOAD_VAR_LOAD_CONST_SUB,
    Opcode.LOAD_VAR_LOAD_CONST_LT, Opcode.LOAD_VAR_LOAD_CONST_GT,
    Opcode.LOAD_VAR_LOAD_CONST_LTE, Opcode.LOAD_VAR_LOAD_CONST_GTE
]

OPCODE_TO_INT = {op: i for i, op in enumerate(OPCODE_LIST)}
INT_TO_OPCODE = list(OPCODE_LIST)

BYTECODE_VERSION = 1

def validate_bytecode_version(data: dict) -> bool:
    if isinstance(data, dict):
        version = data.get("bytecode_version", 0)
        if version > BYTECODE_VERSION:
            raise UnsupportedBytecodeVersionError(
                f"Bytecode version {version} is not supported (max supported: {BYTECODE_VERSION})"
            )
        return True
    return True


class Instruction:
    __slots__ = ('opcode', 'arg', 'line', 'opcode_int')

    def __init__(self, opcode: str, arg=None, line: int = None):
        self.opcode = opcode
        self.arg = arg
        self.line = line
        self.opcode_int = OPCODE_TO_INT.get(opcode, -1)

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

