# Eigen 2.4 вЂ” Mone Language Specification & Developer Reference

This document provides the authoritative language specification and developer reference for **Eigen 2.4 вЂ” Mone**, a domain-specific, hybrid classical-quantum programming language. Eigen integrates a robust classical execution runtime (supporting structures, dynamic collections, recursion, and exception handling) with quantum circuit execution, automatic SSA and graph-based optimization, formal verification, and native acceleration.

---

## 1. Language Philosophy & Execution Model

Eigen is designed with a **runtime-first** philosophy. Unlike quantum assembly formats or embedded library SDKs, Eigen programs run inside a managed classical-quantum Virtual Machine (VM). 

### 1.1 Pure VM vs. Backend Compatibility
1. **Source of Truth**: The classical runtime execution loop, variables, memory space, call frames, and exception handlers are managed directly by the Eigen Virtual Machine (EBC) or Eigen Runtime.
2. **Backend Transparency**: Target backends (such as IBM OpenQASM 3.0, Qiskit Aer, AWS Braket, and Azure QIR) are optional output targets. Since typical quantum hardware backends do not support rich classical structures like recursion, maps, or exception handling, Eigen supports:
   - **Hybrid Simulation**: Running the full hybrid classical-quantum pipeline on the optimized classical Eigen VM.
   - **Exporters / Transpilation**: Extracting pure quantum gate operations (after static path analysis or execution tracing) and exporting them to target quantum architectures.
   - **Strict Verification**: In `--strict` audit mode, programs containing classical constructs incompatible with the chosen hardware backend will raise compilation errors rather than executing silently with semantic loss.

---

## 2. Lexical Structure

### 2.1 Character Set & Whitespace
Eigen source files are encoded in standard UTF-8. 
- **Whitespace**: Spaces, tabs, and newlines are used as delimiters but do not define block scopes (which are explicitly defined by braces `{}`).
- **Line Endings**: Standard Unix (`\n`) and Windows (`\r\n`) line endings are supported.

### 2.2 Comments
Comments in Eigen are single-line and begin with a `#` symbol or `//`. They can appear anywhere on a line. Everything from the `#` or `//` to the end of that line is ignored by the lexer.
```eigen
# This is a full-line comment
let x: int = 10  # This is an inline comment
// This is also a comment
```
Doc-comments begin with three hash marks `###` or three slashes `///` and are processed by the documentation generator.

### 2.3 Identifiers
Identifiers are names given to variables, functions, quantum subroutines, fields, modules, and structures.
- **Regex Rule**: `[a-zA-Z_][a-zA-Z0-9_\.]*`
- Identifiers must start with an ASCII letter or underscore `_`. They can contain letters, numbers, underscores, and dots (used for module paths like `std.math.sin`).

### 2.4 Keywords
The following tokens are reserved keywords and cannot be used as identifiers:
```text
eigen       module      import      qfunc       func
struct      enum        let         if          else
elif        for         in          while       break
continue    try         catch       throw       noise
return      trace       print       assert      parallel
task        match       case        default     qubit
cbit        int         float       string      bool
array       map         null        true        false
and         or          not
```

### 2.5 Literal Types
- **Integer Literals**: Sequence of digits (e.g., `42`, `0`, `1000`).
- **Hex Literals**: `0x` prefix followed by hex digits (e.g., `0xFF` → 255, `0xDEAD` → 57005).
- **Binary Literals**: `0b` prefix followed by binary digits (e.g., `0b1010` → 10).
- **Octal Literals**: `0o` prefix followed by octal digits (e.g., `0o77` → 63).
- **Float Literals**: Sequence of digits containing a single decimal point (e.g., `3.14159`, `0.0`, `1.0`).
- **Scientific Notation**: `1.23e-5`, `6.022e23` — parsed as float literals.
- **String Literals**: Double-quoted character sequences (e.g., `"Hello, Eigen!"`). Escape sequences like `\n`, `\t`, `\r`, `\0`, `\\`, `\"` are evaluated.
- **String Interpolation**: `${expr}` inside string literals is parsed and evaluated at runtime (e.g., `"Result: ${x}"` concatenates the string "Result: " with the value of `x`).
- **Boolean Literals**: `true` and `false`.
- **Null Literal**: `null` (used as an uninitialized reference for structures).

---

## 3. Static Type System

Eigen uses a static, checked type system. The type checker enforces correctness at compile time, checking variable bounds, assignment types, function argument types, and generic collection parameters.

### 3.1 Primitive Types
- **`int`**: 64-bit signed integer.
- **`float`**: Double-precision (64-bit) floating-point number.
- **`string`**: Immutable sequence of UTF-8 characters.
- **`bool`**: Boolean value (`true` or `false`).

### 3.2 Quantum Types
- **`qubit`**: A reference to a physical or simulated quantum bit.
  - **Non-Copiable / Linear Type**: Qubits cannot be copied, reassigned, or passed by value. They must be allocated and referenced by name. This guarantees physical conservation laws.
- **`cbit`**: A classical bit (storing `0` or `1`) representing the output of a quantum measurement.
  - **Compatibility**: `cbit` variables can be compared directly with integer constants or variables, and can be coerced to `int` in assignments.

### 3.3 Composite Types
#### Structures (`struct`)
Structures are user-defined data structures consisting of named, typed fields.
```eigen
struct Particle {
    x: float,
    y: float,
    charge: int
}
```
Structure instantiation and member access:
```eigen
let p: Particle = Particle { x: 0.0, y: 1.5, charge: 1 }
p.x = 2.4
print p.x
```

#### Enumerations (`enum`)
Enumerations define a finite set of named integer constants.
```eigen
enum State {
    INIT,
    RUNNING,
    HALTED
}
```

#### Tuples
Tuples represent fixed-size, heterogenous collections.
```eigen
let pair: (int, float) = (42, 3.14)
```

### 3.4 Collections (Generics)
Eigen supports parameterized collection types which are strictly checked at compile-time to guarantee type safety:

#### Arrays (`array<T>`)
A dynamically sized, contiguous sequence of elements of type `T`.
```eigen
let numbers: array<int> = [1, 2, 3, 4]
let first: int = numbers[0]
```

#### Maps (`map<K, V>`)
A hash-map key-value store mapping keys of type `K` to values of type `V`.
```eigen
let scores: map<string, float> = {"Alice": 98.5, "Bob": 84.0}
let alice_score: float = scores["Alice"]
```

### 3.5 Compatibility, Promotion, and Coercion
- **Cbit & Int Coercion**: Comparisons (`==`, `!=`) allow comparing a `cbit` directly with `int` (e.g. `if c0 == 1`).
- **Implicit Number Promotion**: In classical expressions, `int` is promoted to `float` if matched with another float (e.g., `let x: float = 3.0 + 2`). This is also supported in rotation angles (e.g. `RX q, 2`).
- **Strict Bounds Verification**: Array indexes are checked by the VM, and invalid accesses raise runtime errors.

---

## 4. Extended Backus-Naur Form (EBNF) Grammar

Below is the formal syntactic grammar of Eigen 2.4:

```ebnf
Program             = "eigen" Version [ ModuleDecl ] { ImportDecl } { Statement } EOF ;
Version             = FloatLiteral | IntLiteral ;
ModuleDecl          = "module" Identifier ;
ImportDecl          = "import" Identifier ;

Statement           = QFuncDecl 
                    | FuncDecl 
                    | StructDecl 
                    | EnumDecl 
                    | VarDecl 
                    | LetStmt 
                    | AssignmentStmt 
                    | GateStmt 
                    | QFuncCall 
                    | CallNode 
                    | MeasureStmt 
                    | IfStmt 
                    | ForStmt 
                    | WhileStmt 
                    | BreakStmt 
                    | ContinueStmt 
                    | TryCatchStmt 
                    | ThrowStmt 
                    | NoiseStmt 
                    | ReturnStmt 
                    | TraceStmt 
                    | PrintStmt 
                    | AssertStmt 
                    | ParallelBlock ;

QFuncDecl           = "qfunc" Identifier "(" [ ParamList ] ")" "{" { Statement } "}" ;
FuncDecl            = "func" Identifier "(" [ ParamList ] ")" "->" Type "{" { Statement } "}" ;
StructDecl          = "struct" Identifier "{" FieldList "}" ;
FieldList           = Identifier ":" Type { "," Identifier ":" Type } ;
EnumDecl            = "enum" Identifier "{" Identifier { "," Identifier } "}" ;

VarDecl             = Type Identifier ;
LetStmt             = "let" Identifier ":" Type "=" Expression ;
AssignmentStmt      = AccessExpr "=" Expression
                    | AccessExpr "+=" Expression
                    | AccessExpr "-=" Expression
                    | AccessExpr "*=" Expression
                    | AccessExpr "/=" Expression ;

GateStmt            = SingleQubitGate Identifier
                    | TwoQubitGate Identifier "," Identifier
                    | RotationGate Identifier "," Expression ;

SingleQubitGate     = "H" | "X" | "Y" | "Z" | "S" | "T" ;
TwoQubitGate        = "CNOT" | "CZ" | "SWAP" ;
RotationGate        = "RX" | "RY" | "RZ" ;

QFuncCall           = Identifier "(" [ ArgList ] ")" ;
CallNode            = Identifier "(" [ ExprList ] ")" ;
ArgList             = Identifier { "," Identifier } ;
ExprList            = Expression { "," Expression } ;

MeasureStmt         = "measure" Identifier "->" Identifier ;
IfStmt              = "if" Expression "{" { Statement } "}" [ "else" "{" { Statement } "}" ] ;
ForStmt             = "for" Identifier "in" Expression "{" { Statement } "}" ;
WhileStmt           = "while" Expression "{" { Statement } "}" ;
BreakStmt           = "break" ;
ContinueStmt        = "continue" ;

TryCatchStmt        = "try" "{" { Statement } "}" "catch" Identifier "{" { Statement } "}" ;
ThrowStmt           = "throw" Expression ;
NoiseStmt           = "noise" ( "depolarizing" | "bitflip" ) "(" Expression ")" Identifier { "," Identifier } ;

ReturnStmt          = "return" [ Expression ] ;
TraceStmt           = "trace" ;
PrintStmt           = "print" Expression ;
AssertStmt          = "assert" Expression ;
ParallelBlock       = "parallel" "{" { TaskStatement } "}" ;
TaskStatement       = "task" CallNode ;

Type                = ( "int" | "float" | "string" | "bool" | "qubit" | "cbit" | "array" | "map" | Identifier ) [ "<" Type { "," Type } ">" ] ;

Expression          = LogicalOrExpr ;
LogicalOrExpr       = LogicalAndExpr { "or" LogicalAndExpr } ;
LogicalAndExpr      = EqualityExpr { "and" EqualityExpr } ;
EqualityExpr        = RelationalExpr { ( "==" | "!=" ) RelationalExpr } ;
RelationalExpr      = AdditiveExpr { ( "<" | ">" | "<=" | ">=" ) AdditiveExpr } ;
AdditiveExpr        = MultiplicativeExpr { ( "+" | "-" ) MultiplicativeExpr } ;
AdditiveExpr        = MultiplicativeExpr { ( "*" | "/" ) UnaryExpr } ;
UnaryExpr           = [ "-" | "+" | "not" ] AccessExpr ;
AccessExpr          = PrimaryExpr { "." Identifier | "[" Expression "]" | "(" [ ExprList ] ")" } ;

PrimaryExpr         = IntLiteral
                    | FloatLiteral
                    | StringLiteral
                    | BooleanLiteral
                    | "null"
                    | Identifier
                    | Constant
                    | ArrayLiteral
                    | MapLiteral
                    | TupleLiteral
                    | StructLiteral
                    | "(" Expression ")" ;

ArrayLiteral        = "[" [ ExprList ] "]" ;
MapLiteral          = "{" [ KeyValuePair { "," KeyValuePair } ] "}" ;
KeyValuePair        = Expression ":" Expression ;
TupleLiteral        = "(" Expression "," Expression { "," Expression } ")" ;
StructLiteral       = Identifier "{" FieldBindingList "}" ;
FieldBindingList    = Identifier ":" Expression { "," Identifier ":" Expression } ;

Constant            = "PI" | "TAU" | "E" ;
BooleanLiteral      = "true" | "false" ;
StringLiteral       = '"' { AnyCharacter } '"' ;
Identifier          = ( Letter | "_" ) { Letter | Digit | "_" | "." } ;
Digit               = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
Letter              = "a" | ... | "z" | "A" | ... | "Z" ;
IntLiteral          = Digit { Digit } ;
FloatLiteral        = Digit { Digit } "." Digit { Digit } ;
```

---

## 5. Quantum Execution Model

### 5.1 Qubit Allocation & Lifetime
Quantum execution requires defining named references to qubits and classical bits.
```eigen
qubit q0  # Allocates a new quantum bit in state |0>
cbit c0   # Allocates a classical bit in state 0
```
Qubits are managed by the quantum simulator and are cleaned up upon leaving their declared scope.

### 5.2 Quantum Gate Operations
Eigen supports native single-qubit, two-qubit, and parameterized rotation gates. All gates must match target connectivity layouts if hardware routing constraints are active:

#### Single-Qubit Gates
- **`H q`**: Hadamard gate (creates superposition).
- **`X q`**: Pauli-X gate (bit-flip, logical NOT).
- **`Y q`**: Pauli-Y gate.
- **`Z q`**: Pauli-Z gate (phase-flip).
- **`S q`**: Phase gate (square root of Z).
- **`T q`**: $\pi/8$ gate (square root of S).

#### Two-Qubit Gates
- **`CNOT q_ctrl, q_target`**: Controlled-NOT gate.
- **`CZ q_ctrl, q_target`**: Controlled-Z gate.
- **`SWAP q1, q2`**: Swap states of two qubits.

#### Parameterized Rotation Gates
- **`RX q, angle`**: Rotate around X-axis by `angle` (float expression).
- **`RY q, angle`**: Rotate around Y-axis by `angle`.
- **`RZ q, angle`**: Rotate around Z-axis by `angle`.

```eigen
# Rotates qubit q0 by pi/2 around X axis
RX q0, 1.57079632679
```

### 5.3 Quantum Measurements
Measurements collapse a qubit's quantum state onto classical outcomes (`0` or `1`), saving the result to a classical bit:
```eigen
measure q0 -> c0
```

### 5.4 Custom Quantum Functions (`qfunc`)
Quantum operations can be bundled inside reusable `qfunc` structures. Unlike classical functions, `qfunc` subroutines do not return values and can only contain quantum allocations, gate operations, and sub-calls.
```eigen
qfunc bell_state(qubit a, qubit b) {
    H a
    CNOT a, b
}
```

### 5.5 Noise Modeling
Decoherence and hardware readout errors can be simulated using the native `noise` statement, which applies noise channels:
- **`noise depolarizing(p) q0`**: Applies a depolarizing channel with probability `p` $[0.0, 1.0]$.
- **`noise bitflip(p) q0`**: Applies a probabilistic Pauli-X gate with probability `p`.

```eigen
H q0
noise depolarizing(0.02) q0  # 2% depolarizing channel noise
```

### 5.6 Qubit Indexing & Memory Layout Conventions
Eigen uses a **Little-Endian (LSB-first)** convention for mapping physical/simulated qubits to state vector indices and binary output representations.

1. **Bit Mapping**: The first qubit allocated in the program (typically `q0`) corresponds to the least significant bit (LSB) at index position 0. The second qubit allocated (`q1`) corresponds to index position 1, and the $N$-th qubit (`qN`) maps to index position $N-1$.
2. **State Vector Indexing**: An index `idx` in the state vector is structured as a binary integer:
   $$\text{idx} = \sum_{k=0}^{N-1} b_k 2^k$$
   where $b_k \in \{0, 1\}$ represents the state of the $k$-th allocated qubit.
3. **Amplitude String Presentation**: When rendering state probabilities or amplitude dictionaries (e.g., via `get_amplitudes_dict()`), the bitstrings are written from left to right as `q[N-1]q[N-2]...q[1]q[0]`. Thus, for a two-qubit state where `q0` is in state $|1\rangle$ and `q1` is in state $|0\rangle$, the output bitstring is `"01"`.

#### Conversion Utilities
If your workflows or downstream tools expect a **Big-Endian (MSB-first)** layout (where the first allocated qubit `q0` maps to the leftmost bit: `q[0]q[1]...q[N-1]`), you can import and use the built-in converters:
```python
from src.utils.converters import to_msb_first_dict, reorder_state_vector

# Convert amplitude dictionary keys
msb_dict = to_msb_first_dict(simulator.get_amplitudes_dict())

# Reorder raw state vector lists
msb_state_vector = reorder_state_vector(raw_state, num_qubits, source_convention="lsb", target_convention="msb")
```

---

## 6. Classical Functions & Control Flow

### 6.1 Classical Functions (`func`)
Classical subroutines are declared using `func` and must specify typed parameters and an explicit return type using the `->` operator.
```eigen
func add_numbers(a: int, b: int) -> int {
    return a + b
}
```

### 6.2 Scope Rules
Eigen resolves variables lexically. Block statements bounded by braces `{}` form closures; variables declared within blocks cannot be accessed outside them. 

### 6.3 Recursion & Memory Limits
Eigen functions natively support recursion. Activation frames are allocated on the VM call stack. The maximum call frame recursion depth under the VM is capped at $10,000$ to prevent system stack overflow:
```eigen
func fibonacci(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}
```

### 6.4 Local Variable Bindings
Variables are defined using the `let` keyword, requiring a type annotation and initialization:
```eigen
let count: int = 0
let weight: float = 78.3
let name: string = "Mone"
```

### 6.5 Conditional Branching
`if` and `else` blocks allow conditional execution based on boolean expressions.
```eigen
if weight > 100.0 {
    print "Heavy"
} else {
    print "Normal"
}
```

### 6.6 Iteration (Loops)
Eigen supports both `while` loops and collection iteration with `for`.

```eigen
# While loop
let idx: int = 0
while idx < 5 {
    print idx
    idx += 1
}

# For loop iterating through an array
let arr: array<int> = [10, 20, 30]
for x in arr {
    print x
}
```

`break` and `continue` keywords provide standard early loop exit and skip controls.

---

## 7. Exception Handling

Eigen implements structured exception handling through `try`-`catch`-`throw` blocks. This allows developers to catch runtime errors or emit user-defined exceptions.

```eigen
try {
    let result: int = compute_value()
    if result < 0 {
        throw "NegativeResultError"
    }
} catch err {
    print "Caught error:"
    print err
}
```

### 7.1 Throw Semantics
The `throw` keyword takes any valid expression (typically a string or error structure). The VM pauses execution, unwinds the active call stack frames, and checks for catch handlers. If no handler is resolved, the program aborts with a stack dump.

---

## 8. Concurrency & Parallelism

For hybrid quantum tasks, Eigen introduces parallel execution constructs. These are compiled using special VM instruction sets (`SPAWN` and `JOIN`) and execute concurrently via a DAG task scheduler.

### 8.1 Parallel Blocks
The `parallel` keyword groups multiple independent calls to classical functions into concurrent execution tasks.
```eigen
func perform_simulation(sim_id: int) -> int {
    # Perform complex simulation task
    return sim_id * 2
}

parallel {
    task perform_simulation(1)
    task perform_simulation(2)
}
```

### 8.2 Execution & Scheduling Model
- **Thread Pool**: The parallel task scheduler (`TaskScheduler`) manages a thread pool (configured via max workers).
- **DAG Dependency Resolution**: Tasks within the parallel block are run concurrently. In the intermediate compiler representation, tasks with dependencies form a Directed Acyclic Graph (DAG) and are scheduled automatically once all their prerequisites complete.

---

## 9. Standard Library (stdlib)

Eigen features a built-in standard library. Standard library modules are imported using the `import` statement.

### 9.1 `std.math`
Mathematical operations and trigonometric functions (angle inputs in radians):
- `sin(x: float) -> float`: Sine.
- `cos(x: float) -> float`: Cosine.
- `tan(x: float) -> float`: Tangent.
- `sqrt(x: float) -> float`: Square root.
- `log(x: float) -> float`: Natural logarithm.
- `exp(x: float) -> float`: Exponential.
- `abs(x: float) -> float`: Absolute value.

### 9.2 `std.io`
File operations and formatting:
- `read_file(filepath: string) -> string`: Reads text file content.
- `write_file(filepath: string, content: string) -> int`: Writes content to a file, returns bytes written.
- `print_format(fmt: string, val: int) -> int`: Prints a formatted integer string.

### 9.3 `std.collections`
Collection helpers:
- `append_int(arr: array<int>, val: int) -> int`: Appends integer element to array.
- `remove_at(arr: array<int>, idx: int) -> int`: Removes element at given index.

### 9.4 `std.random`
Random number generation:
- `rand_float() -> float`: Returns a random float between `0.0` and `1.0`.
- `rand_int(min_val: int, max_val: int) -> int`: Returns random integer in range.

### 9.5 `std.stats`
Statistical computations:
- `mean(arr: array<float>) -> float`: Computes arithmetic mean.
- `variance(arr: array<float>) -> float`: Computes variance.

### 9.6 `std.string`
String operations:
- `concat(s1: string, s2: string) -> string`: Concatenates two strings.
- `format_int(val: int) -> string`: String representation of integer.

### 9.7 `std.time`
Timing functions:
- `now() -> float`: Unix epoch timestamp.
- `sleep(ms: int) -> int`: Blocks the thread for `ms` milliseconds.

---

## 10. Compiler & Optimization Pipeline

```
[ Eigen Source ]
      в”‚
      в–ј
   [ AST ]  <в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ [ Incremental AST Cache ]
      в”‚
      в–ј
  [ MLIR ]  в”Ђв”Ђ(Dialects: func, arith, quantum, cf)
      в”‚
      в–ј
  [ EQIR ]  в”Ђв”Ђ(Quantum DAG IR representation) <в”Ђв”Ђв”Ђв”Ђ [ EQIR Cache ]
  в”њв”Ђв”Ђв”Ђ Optimizer Passes (Gate Fusion, ZX, etc.)
  в””в”Ђв”Ђв”Ђ Equivalence Verification (ZX-Calculus & Unitary)
      в”‚
      в”њв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”ђ
      в–ј                            в–ј
[ EBC Bytecode ]           [ LLVM IR / QIR ]
      в”‚                            в”‚
      в–ј                            в–ј
[ Eigen VM / JIT ]        [ Native Target Code ]
```

### 10.1 Abstract Syntax Tree (AST) & Zero-Copy Pratt Parsing
The native compiler front-end (implemented in Rust) utilizes a zero-copy Pratt parser that parses token references from byte streams without text duplication. It produces 100% compliant and mutable AST objects.

### 10.2 MLIR Dialect Layer
The compiler translates the AST into a structured Multi-Level Intermediate Representation (MLIR) matching standard dialects:
- **`func` dialect**: Defines execution function boundaries and parameters.
- **`arith` dialect**: Low-level classic integer and floating-point computations.
- **`quantum` dialect**: Allocations and quantum gate operations.
- **`cf` dialect**: Control flow transitions between basic blocks.

### 10.3 EQIR DAG
Quantum gates are extracted into the **Equivalent Quantum Intermediate Representation (EQIR)** graph. This is a Directed Acyclic Graph representing gate dependencies, enabling gate fusion, commutation rewrites, and dead-gate cancellation.

### 10.4 Caching System & Query DB
To accelerate incremental rebuilds, the compilation pipeline serializes intermediate stages to disk under `.eigen_cache/` matching cryptographic file hashes (SHA-256):
- `*.ast`: Binary pickle format of parsed AST.
- `*.ssa`: Basic block structures.
- `*.eqir`: Serialized JSON graph layout.
- `*.zx`: Simplified Clifford graph layouts.
- `*.ebc`: Instruction sets.

### 10.5 JIT v2 Loop Optimizations
Traces are monitored by an adaptive VM loop compiler:
- **Loop-Invariant Code Motion (LICM):** Code instructions that produce identical values inside loops are automatically moved out of the loop block.
- **Trace Specialization:** Inserts shape and type guards. In the case of guard failure, execution jumps back to deoptimized classical paths.

### 10.6 Standalone LLVM & QIR Generation
Using `eigen build <file.eig> --aot --qir`, basic SSA blocks are converted into LLVM assembly conforming to standard QIR specification schemas. The resulting code compiles directly to standalone machine executables (`.exe` on Windows, native binaries on Linux/macOS) free of CPython dependencies.

