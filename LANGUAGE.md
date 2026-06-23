# Eigen 2.3 — Helios Language Specification & Developer Reference

This document provides the authoritative language specification and developer reference for **Eigen 2.3 — Helios**, a domain-specific, hybrid classical-quantum programming language. Eigen integrates a robust classical execution runtime (supporting structures, dynamic collections, recursion, and exception handling) with quantum circuit execution, automatic SSA and graph-based optimization, formal verification, and native acceleration.

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
Comments in Eigen are single-line and begin with a `#` symbol. They can appear anywhere on a line. Everything from the `#` to the end of that line is ignored by the lexer.
```eigen
# This is a full-line comment
let x: int = 10  # This is an inline comment
```

### 2.3 Identifiers
Identifiers are names given to variables, functions, quantum subroutines, fields, modules, and structures.
- **Regex Rule**: `[a-zA-Z_][a-zA-Z0-9_\.]*`
- Identifiers must start with an ASCII letter or underscore `_`. They can contain letters, numbers, underscores, and dots (used for module paths like `std.math.sin`).

### 2.4 Keywords
The following tokens are reserved keywords and cannot be used as identifiers:
```text
eigen       module      import      qfunc       func
struct      enum        let         if          else
for         in          while       break       continue
try         catch       throw       noise       return
trace       print       assert      parallel    task
qubit       cbit        int         float       string
bool        array       map         null        true
false       and         or          not
```

### 2.5 Literal Types
- **Integer Literals**: Sequence of digits (e.g., `42`, `0`, `1000`).
- **Float Literals**: Sequence of digits containing a single decimal point (e.g., `3.14159`, `0.0`, `1.0`).
- **String Literals**: Double-quoted character sequences (e.g., `"Hello, Eigen!"`). Escape sequences like `\n` and `\t` are evaluated.
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
  - **Non-Copiable**: Qubits cannot be copied, reassigned, or passed by value. They must be allocated and referenced by name.
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

Below is the formal syntactic grammar of Eigen 2.3:

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
MultiplicativeExpr  = UnaryExpr { ( "*" | "/" ) UnaryExpr } ;
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

---

## 6. Classical Functions & Control Flow

### 6.1 Classical Functions (`func`)
Classical subroutines are declared using `func` and must specify typed parameters and an explicit return type using the `->` operator.
```eigen
func add_numbers(a: int, b: int) -> int {
    return a + b
}
```

### 6.2 Recursion
Eigen functions natively support recursion. Activation frames are allocated on the VM call stack and resolved upon returning.
```eigen
func fibonacci(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}
```

### 6.3 Local Variable Bindings
Variables are defined using the `let` keyword, requiring a type annotation and initialization:
```eigen
let count: int = 0
let weight: float = 78.3
let name: string = "Helios"
```

### 6.4 Conditional Branching
`if` and `else` blocks allow conditional execution based on boolean expressions.
```eigen
if weight > 100.0 {
    print "Heavy"
} else {
    print "Normal"
}
```

### 6.5 Iteration (Loops)
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

### 9.8 Quantum Library Helpers
Predefined quantum algorithms and circuit structures:
- `quantum.bell`: Bell state creation helper.
- `quantum.deutsch`: Oracles for Deutsch-Jozsa algorithms.
- `quantum.ghz`: GHZ (Greenberger-Horne-Zeilinger) state algorithms.
- `quantum.grover`: Iteration structures for Grover's search.

---

## 10. Compiler & IR Pipeline

Eigen's compiler transforms source code through multiple levels of intermediate representation before execution:

```
[ Eigen Source ]
      │
      ▼
   [ AST ]  <─────── [ Incremental AST Cache ]
      │
      ▼
  [ MLIR ]  ──(Dialects: func, arith, quantum, cf)
      │
      ▼
  [ EQIR ]  ──(Quantum DAG IR representation) <──── [ EQIR Cache ]
  ├─── Optimizer Passes (Gate Fusion, ZX, etc.)
  └─── Equivalence Verification (ZX-Calculus & Unitary)
      │
      ├────────────────────────────┐
      ▼                            ▼
[ EBC Bytecode ]           [ LLVM IR / QIR ]
      │                            │
      ▼                            ▼
[ Eigen VM / JIT ]        [ Native Target Code ]
```

### 10.1 Abstract Syntax Tree (AST)
The parser checks the grammar and generates a hierarchical tree structure. The AST includes structures for loops, variables, quantum gates, parallel blocks, and exceptions.

### 10.2 MLIR Dialect Layer
The compiler translates the AST into a structured Multi-Level Intermediate Representation (MLIR) matching standard dialects:
- **`func` dialect**: Defines execution function boundaries and parameters.
- **`arith` dialect**: Low-level classic integer and floating-point computations.
- **`quantum` dialect**: Allocations and quantum gate operations.
- **`cf` dialect**: Control flow transitions between basic blocks.

This intermediate layer simplifies target compiler mapping and code optimization.

### 10.3 EQIR DAG
Quantum gates are extracted into the **Equivalent Quantum Intermediate Representation (EQIR)** graph. This is a Directed Acyclic Graph representing gate dependencies, enabling gate fusion, commutation rewrites, and dead-gate cancellation.

### 10.4 EBC Bytecode & Virtual Machine
The default compilation target is **Eigen Bytecode (EBC)**. The bytecode is compiled to instruction objects executed by the VM dispatch loop.
- **Trace-Based JIT**: The JIT compiler monitors active basic blocks. If a loop is executed repeatedly, the tracer captures the execution bytecode and compiles it to native Python routines, accelerating execution by 2x to 5x.
- **Native Rust Execution**: When PyO3-based `eigen_native` is compiled, EBC execution is offloaded directly to a highly optimized native C/Rust execution thread loop.

### 10.5 LLVM / QIR Generation
Using `eigen build --llvm`, the compiler generates standard LLVM IR / Quantum Intermediate Representation (QIR). This generates clean SSA LLVM files referencing QIR runtime function declarations (e.g. `@__quantum__qis__h__body`).

### 10.6 Caching System
To accelerate incremental rebuilds, the compilation pipeline serializes intermediate stages to disk under `.eigen_cache/` matching file hashes:
- `*.ast`: Binary pickle format of parsed AST.
- `*.ssa`: Basic block structures.
- `*.eqir`: Serialized JSON graph layout.
- `*.zx`: Simplified Clifford graph layouts.
- `*.ebc`: Instruction sets.

---

## 11. Code Examples

### 11.1 Complete Bell State & Classical Assertion
```eigen
eigen 2.3
module quantum.bell

# Create Bell State, measure, and assert entanglement
qubit q0
qubit q1
cbit c0
cbit c1

H q0
CNOT q0, q1

measure q0 -> c0
measure q1 -> c1

print c0
print c1
assert c0 == c1
```

### 11.2 Classical Recursion, Struct Mutation, and Exception Handling
```eigen
eigen 2.3

struct Configuration {
    max_steps: int,
    tolerance: float
}

func calculate_factorial(n: int) -> int {
    if n <= 0 {
        return 1
    }
    return n * calculate_factorial(n - 1)
}

try {
    let conf: Configuration = Configuration { max_steps: 100, tolerance: 0.0001 }
    conf.max_steps = 150
    
    let result: int = calculate_factorial(5)
    print result
    assert result == 120
    
    if conf.max_steps > 100 {
        throw "ConfigStepsExceededException"
    }
} catch err {
    print "Exception trapped successfully:"
    print err
}
```

### 11.3 Parallel Simulation Block
```eigen
eigen 2.3

func run_phase_estimation(angle: float) -> int {
    # Simulation code
    let steps: int = 1000
    return steps
}

parallel {
    task run_phase_estimation(0.314)
    task run_phase_estimation(0.785)
    task run_phase_estimation(1.570)
}
```
