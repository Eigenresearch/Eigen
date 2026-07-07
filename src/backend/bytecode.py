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


# === §4.4 Bytecode Versioning ==============================================
# Major.minor versioning with forward-compatible handling:
#   * Major increments when the bytecode wire format changes in a way
#     that breaks backward/forward compatibility (e.g., opcodes removed,
#     instruction layout restructured).
#   * Minor increments when new opcodes/fields are ADDED in a way that
#     older interpreters can still load the file (forward-compatibility).
# An interpreter compiled against (1, 0) can safely execute a (1, k)
# bytecode file for any k >= 0: unknown opcodes are reported lazily as
# "unsupported opcode" rather than rejected up-front.

BYTECODE_VERSION_MAJOR = 1
BYTECODE_VERSION_MINOR = 0
# Scalar fallback retained for legacy code paths and existing tests that
# compare `version > BYTECODE_VERSION`.
BYTECODE_VERSION = BYTECODE_VERSION_MAJOR


class BytecodeVersion:
    """Major.minor bytecode version with rich comparison & formatting."""

    __slots__ = ("major", "minor")

    def __init__(self, major: int, minor: int = 0):
        if not isinstance(major, int) or isinstance(major, bool):
            raise TypeError("major must be an int")
        if not isinstance(minor, int) or isinstance(minor, bool):
            raise TypeError("minor must be an int")
        if major < 0 or minor < 0:
            raise ValueError("version components must be non-negative")
        self.major = major
        self.minor = minor

    @classmethod
    def from_int(cls, value: int) -> "BytecodeVersion":
        return cls(int(value), 0)

    @classmethod
    def from_tuple(cls, value) -> "BytecodeVersion":
        if len(value) < 1:
            raise ValueError("version tuple must have at least one element")
        major = int(value[0])
        minor = int(value[1]) if len(value) > 1 else 0
        return cls(major, minor)

    @classmethod
    def from_str(cls, value: str) -> "BytecodeVersion":
        text = value.strip().lstrip("v")
        if not text:
            raise ValueError("empty version string")
        parts = text.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return cls(major, minor)

    @classmethod
    def parse(cls, value) -> "BytecodeVersion":
        """Best-effort parse from int, tuple/list, str, or BytecodeVersion."""
        if isinstance(value, BytecodeVersion):
            return cls(value.major, value.minor)
        if isinstance(value, bool):
            raise TypeError("bool is not a valid version")
        if isinstance(value, int):
            return cls.from_int(value)
        if isinstance(value, (tuple, list)):
            return cls.from_tuple(value)
        if isinstance(value, str):
            return cls.from_str(value)
        raise TypeError(f"unsupported version type: {type(value).__name__}")

    def as_tuple(self):
        return (self.major, self.minor)

    def as_int(self) -> int:
        """Backward-compatible int form (major only)."""
        return self.major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    def __repr__(self) -> str:
        return f"BytecodeVersion({self.major}, {self.minor})"

    def __eq__(self, other) -> bool:
        other = BytecodeVersion.parse(other)
        return self.major == other.major and self.minor == other.minor

    def __lt__(self, other) -> bool:
        other = BytecodeVersion.parse(other)
        return self.as_tuple() < other.as_tuple()

    def __le__(self, other) -> bool:
        other = BytecodeVersion.parse(other)
        return self.as_tuple() <= other.as_tuple()

    def __gt__(self, other) -> bool:
        other = BytecodeVersion.parse(other)
        return self.as_tuple() > other.as_tuple()

    def __ge__(self, other) -> bool:
        other = BytecodeVersion.parse(other)
        return self.as_tuple() >= other.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())


# The supported (max) bytecode version this interpreter understands.
SUPPORTED_BYTECODE_VERSION = BytecodeVersion(
    BYTECODE_VERSION_MAJOR, BYTECODE_VERSION_MINOR
)


class CompatibilityStatus:
    """Enum-like class describing how a requested version relates
    to the interpreter's supported version."""
    EXACT = "exact"                 # same (major, minor)
    FORWARD_MINOR = "forward_minor"  # same major, higher minor (forward-compat)
    BACKWARD = "backward"           # strictly older (same or lower major)
    INCOMPATIBLE_MAJOR = "incompatible_major"  # different major (forward)
    INCOMPATIBLE_FUTURE = "incompatible_future"  # higher major than supported


def parse_bytecode_version(value) -> BytecodeVersion:
    """Public entrypoint for parsing a version from any supported form."""
    return BytecodeVersion.parse(value)


def check_bytecode_compatibility(requested) -> str:
    """Return the CompatibilityStatus label for ``requested`` against
    the interpreter's supported version.  Never raises."""
    try:
        rv = BytecodeVersion.parse(requested)
    except (TypeError, ValueError):
        return CompatibilityStatus.INCOMPATIBLE_MAJOR
    sv = SUPPORTED_BYTECODE_VERSION
    if rv == sv:
        return CompatibilityStatus.EXACT
    if rv.major == sv.major:
        if rv.minor > sv.minor:
            return CompatibilityStatus.FORWARD_MINOR
        return CompatibilityStatus.BACKWARD
    if rv.major < sv.major:
        return CompatibilityStatus.BACKWARD
    return CompatibilityStatus.INCOMPATIBLE_FUTURE


def format_version_error(requested, supported=SUPPORTED_BYTECODE_VERSION) -> str:
    """Produce a clear, actionable error message for a version mismatch."""
    rv = BytecodeVersion.parse(requested)
    sv = BytecodeVersion.parse(supported) if not isinstance(
        supported, BytecodeVersion) else supported
    return (
        f"Bytecode version {rv} is not supported by this interpreter "
        f"(max supported: {sv}). Major version mismatch — bytecode was "
        f"produced by a newer, incompatible release. Upgrade the runtime "
        f"or recompile the source with a compatible toolchain."
    )


def is_bytecode_compatible(requested) -> bool:
    """True if the requested version can be loaded by this interpreter
    without raising UnsupportedBytecodeVersionError. Forward-compatible
    minor bumps are considered compatible."""
    status = check_bytecode_compatibility(requested)
    return status in (
        CompatibilityStatus.EXACT,
        CompatibilityStatus.FORWARD_MINOR,
        CompatibilityStatus.BACKWARD,
    )


def validate_bytecode_version(data: dict) -> bool:
    """Validate ``data["bytecode_version"]`` against the supported
    bytecode version.

    Returns True if the bytecode can be loaded by this interpreter.
    Raises UnsupportedBytecodeVersionError if the requested version
    has a HIGHER major than the interpreter supports (incompatible).

    Forward-compatible handling: a bytecode file with a higher MINOR
    version (same major) is loadable; the interpreter will only fail
    later if it encounters an unknown opcode, which is the proper
    forward-compatible behaviour.
    """
    if isinstance(data, dict):
        version = data.get("bytecode_version", 0)
        status = check_bytecode_compatibility(version)
        if status in (
            CompatibilityStatus.INCOMPATIBLE_MAJOR,
            CompatibilityStatus.INCOMPATIBLE_FUTURE,
        ):
            raise UnsupportedBytecodeVersionError(
                format_version_error(version)
            )
        return True
    return True


def load_bytecode(data: dict) -> tuple[list, str]:
    """Load a bytecode dictionary with full version validation.

    Returns (instructions, compatibility_status).  Raises
    UnsupportedBytecodeVersionError for incompatible major versions.

    For forward-compatible minor versions (same major, higher minor),
    the instructions are loaded normally — the VM will raise
    InvalidOpcodeError lazily if it encounters an unknown opcode from
    the newer minor version.
    """
    status = "exact"
    if isinstance(data, dict):
        version = data.get("bytecode_version", 0)
        status = check_bytecode_compatibility(version)
        if status in (
            CompatibilityStatus.INCOMPATIBLE_MAJOR,
            CompatibilityStatus.INCOMPATIBLE_FUTURE,
        ):
            raise UnsupportedBytecodeVersionError(
                format_version_error(version)
            )
        raw_instructions = data.get("instructions", [])
        instructions = [Instruction.from_dict(d) for d in raw_instructions]
        return instructions, status
    return [], "exact"


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

