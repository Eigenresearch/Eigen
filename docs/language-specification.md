# Eigen 2.3 Language Specification

This document provides the formal language specification for Eigen version 2.3.

## 1. Syntax Grammar (EBNF)

Below is the formal Extended Backus-Naur Form (EBNF) grammar specification for Eigen 2.3:

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
                    | AssertStmt ;

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

## 2. Type System

Eigen implements a static type system comprising primitive, quantum, hybrid, and composite structures.

### 2.1 Primitive and Collection Types
- **`qubit`**: A non-copiable quantum bit reference.
- **`cbit`**: A classical bit (storing `0` or `1`), compatible with `int` for assignments and comparisons.
- **`int`**: 64-bit signed integer.
- **`float`**: Double-precision floating-point number.
- **`string`**: Character strings (immutable).
- **`bool`**: Boolean value (`true` or `false`).
- **`array<T>`**: Dynamically allocated list of elements of type `T`.
- **`map<K, V>`**: Key-value store mapping keys of type `K` to values of type `V`.
- **`struct`**: User-defined structured collection of named, typed fields.
- **`enum`**: User-defined finite set of named constants.

### 2.2 Compatibility and Coercion
- **Cbit & Int Compatibility**: Comparators (`==`, `!=`) and assignments permit comparing `cbit` directly with `int` literals.
- **Implicit Promotion**: Only `int` promotes to `float` where expected (e.g. gate rotation angles).
- **Generics**: Fully checked at compile-time to guarantee collection homogeneity.

---

## 3. Runtime Guarantees

Eigen Runtime and VM provide full execution guarantees. Every language construct—including recursive functions, loops, structures, arrays, maps, and exception catch blocks—is executed natively by the Eigen VM. Classical execution is considered the source of truth, whereas backend exporters (like the Qiskit backend) are optional compatibility targets.

---

## 4. Backend Compatibility Matrix

The table below specifies the level of support (`FULL`, `PARTIAL`, or `NONE`) for language features across execution targets:

| Feature / Capability | Eigen Runtime / VM | Qiskit Transpiler | EQIR v1.1 DAG |
| -------------------- | ------------------ | ----------------- | ----------- |
| Quantum Gates        | `FULL`             | `FULL`            | `FULL`      |
| Measurements         | `FULL`             | `FULL`            | `FULL`      |
| Noise Channels       | `FULL`             | `NONE`            | `NONE`      |
| Recursive Functions  | `FULL`             | `NONE`            | `NONE`      |
| Loops                | `FULL`             | `NONE`            | `NONE`      |
| Structs              | `FULL`             | `NONE`            | `NONE`      |
| Maps                 | `FULL`             | `NONE`            | `NONE`      |
| Arrays (Dynamic)     | `FULL`             | `NONE`            | `NONE`      |
| Try-Catch Exceptions | `FULL`             | `NONE`            | `NONE`      |
| Debug Directives     | `FULL`             | `PARTIAL` (Comm)  | `FULL`      |

---

## 5. Operational Semantics

### 5.1 Exception Handling (`try`-`catch`-`throw`)
Exceptions are dynamically thrown using `throw expression`. The VM call stack is unwound step-by-step until the closest matching catch handler is resolved. Uncaught exceptions cause VM termination with trace outputs.

### 5.2 Noise Modeling
The simulator supports applying decoherence noise directly on qubits.
- `noise depolarizing(p) q0`: Applies a depolarizing channel with error probability `p`.
- `noise bitflip(p) q0`: Applies a bit-flip channel (probabilistic X gate) with probability `p`.
These operations are executed natively by the VM simulator.
