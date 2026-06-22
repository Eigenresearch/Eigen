# Eigen Language Specification

This document provides the formal language specification for Eigen version 1.0.

## 1. Syntax Grammar (EBNF)

Below is the formal Extended Backus-Naur Form (EBNF) grammar specification for Eigen:

```ebnf
Program             = "eigen" Version [ ModuleDecl ] { ImportDecl } { Statement } EOF ;
Version             = FloatLiteral | IntLiteral ;
ModuleDecl          = "module" Identifier ;
ImportDecl          = "import" Identifier ;

Statement           = QFuncDecl 
                    | VarDecl 
                    | LetStmt 
                    | GateStmt 
                    | QFuncCall 
                    | MeasureStmt 
                    | IfStmt 
                    | "return" 
                    | "trace" 
                    | "print" Expression 
                    | AssertStmt ;

QFuncDecl           = "qfunc" Identifier "(" [ ParamList ] ")" "{" { Statement } "}" ;
ParamList           = Param { "," Param } ;
Param               = Type Identifier ;

VarDecl             = Type Identifier ;
Type                = "qubit" | "cbit" | "int" | "float" ;

LetStmt             = "let" Identifier ":" BasicType "=" Expression ;
BasicType           = "int" | "float" | "cbit" ;

GateStmt            = SingleQubitGate Identifier
                    | TwoQubitGate Identifier "," Identifier
                    | RotationGate Identifier "," Expression ;

SingleQubitGate     = "H" | "X" | "Y" | "Z" | "S" | "T" ;
TwoQubitGate        = "CNOT" | "CZ" | "SWAP" ;
RotationGate        = "RX" | "RY" | "RZ" ;

QFuncCall           = Identifier "(" [ ArgList ] ")" ;
ArgList             = Identifier { "," Identifier } ;

MeasureStmt         = "measure" Identifier "->" Identifier ;

IfStmt              = "if" Expression "==" Expression "{" { Statement } "}" ;

AssertStmt          = "assert" Expression "==" Expression ;

Expression          = AdditiveExpr ;
AdditiveExpr        = MultiplicativeExpr { ( "+" | "-" ) MultiplicativeExpr } ;
MultiplicativeExpr  = PrimaryExpr { ( "*" | "/" ) PrimaryExpr } ;
PrimaryExpr         = IntLiteral
                    | FloatLiteral
                    | Identifier
                    | Constant
                    | "(" Expression ")"
                    | "-" PrimaryExpr
                    | "+" PrimaryExpr ;

Constant            = "PI" | "TAU" | "E" ;
Identifier          = ( Letter | "_" ) { Letter | Digit | "_" | "." } ;
Digit               = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
Letter              = "a" | ... | "z" | "A" | ... | "Z" ;
IntLiteral          = Digit { Digit } ;
FloatLiteral        = Digit { Digit } "." Digit { Digit } ;
```

## 2. Type System

Eigen implements a static type system with four primitive types:
- **`qubit`**: Represents a quantum bit state. Qubits are hardware-level resources that cannot be copied or reassigned. They can only be declared and manipulated using unitary operations (gates) or collapsed via measurement.
- **`cbit`**: Represents a classical bit (storing values `0` or `1`). Used primarily for storing measurement outcomes and classical conditional branching.
- **`int`**: Represents standard integer variables used in classical arithmetic expressions.
- **`float`**: Represents standard double-precision floating-point numbers. Used primarily for specifying rotation gate angles.

### Type Compatibility and Coercion
- Implicit type coercion is restricted. Only `int` values can be implicitly promoted to `float` in numeric expressions (e.g., `let theta: float = PI / 2` promotes `2` to `2.0`).
- Any attempt to apply a quantum gate to a non-`qubit` variable triggers a compilation-time type error.
- Any attempt to store a measurement outcome in a non-`cbit` variable triggers a compilation-time type error.

## 3. Scope and Modularity

### Files and Versioning
Every valid Eigen file must start with the `eigen 1.0` header directive, identifying the version of the compiler requested.

### Module System
- A file can optionally declare its namespace via `module <module_path>`.
- Other files can import this module using `import <module_path>`. Dotted paths correspond directly to directory structures (e.g., `import quantum.bell` maps to `quantum/bell.eig`).
- Importing compiles the target module and merges its `qfunc` declarations into the current file's global compiler namespace, making subroutines available for invocation.

## 4. Operational Semantics

### Variable Binding (`let`)
Variables of type `int`, `float`, and `cbit` are declared and initialized using `let name: type = expression`. Bindings are evaluated statically during AST-to-EQIR compilation if they represent constants, or tracked dynamically if they represent runtime states (e.g., classical bits).

### Quantum Functions (`qfunc`)
`qfunc` declarations define parameterized, reusable quantum subroutines. They cannot allocate qubits internally. Instead, they accept qubits as arguments and manipulate them. During compilation, all `qfunc` invocations are inlined into a flat execution graph (EQIR v1), resolving parameter names to argument names.

### Built-in Quantum Gates
Unitary transformations alter the quantum state vector according to:
- **Hadamard (`H`)**: Creates equal superposition states:
  \[H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\]
- **Pauli-X (`X`)**: Quantum NOT gate:
  \[X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\]
- **Rotation-X (`RX`)**: Rotation around the X-axis:
  \[RX(\theta) = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}\]

### Measurement (`measure`)
Measurement collapses the quantum state of a targeted qubit:
- Extracts a classical bit outcome with probability \(P(b) = |\langle b | \psi \rangle|^2\).
- Erases entanglement of the measured qubit and updates the classical store.
- Wavefunction collapse is simulated probabilistically using a pseudo-random number generator.
