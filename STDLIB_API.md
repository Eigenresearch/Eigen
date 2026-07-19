# Eigen Standard Library API

This document lists every module in `stdlib/std/` with its function
signatures and behavior. Each module is a `.eig` file beginning with
`eigen 1.0` and a `module std.<name>` declaration.

Functions marked **stub** return a placeholder value — they either rely on a
native helper not yet implemented in pure Eigen (transcendental functions,
SHA-256, array mutation) or are placeholders kept for API compatibility.
Calling them does not raise; the return value is documented per function.

The VM maps short names (e.g. `sin`) to fully-qualified stdlib paths
(`std.math.sin`) via the `EigenVM._STD_MAPPING` class attribute (see
`src/backend/vm.py`).

| Module | Functions | Notes |
|--------|-----------|-------|
| `std.bitwise` | 9 | Pure Eigen, fully implemented. |
| `std.collections` | 2 | Array helpers, stub bodies. |
| `std.complex` | 8 | Complex-number helpers using `array<float>` pairs. |
| `std.hashing` | 5 | FNV-1a and rolling hashes; SHA-256 is a stub. |
| `std.io` | 3 | File I/O stubs. |
| `std.linear_algebra` | 9 | Vector/matrix helpers, most are stubs. |
| `std.math` | 7 | `sqrt`, `abs` implemented; trig/transcendentals throw. |
| `std.quantum_helpers` | 8 | State-vector construction helpers. |
| `std.random` | 2 | RNG stubs. |
| `std.sorting` | 5 | Insertion/merge/quick sort, binary search. |
| `std.stats` | 2 | Mean / variance stubs. |
| `std.string` | 2 | String helpers, stub bodies. |
| `std.time` | 2 | Wall-clock stubs. |

---

## std.bitwise

Integer bit operations. All functions are pure Eigen and fully implemented.

```eig
eigen 1.0
module std.bitwise
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `and` | `(a: int, b: int) -> int` | Bitwise AND (`a & b`). |
| `or` | `(a: int, b: int) -> int` | Bitwise OR (`a \| b`). |
| `xor` | `(a: int, b: int) -> int` | Bitwise XOR (`a ^ b`). |
| `not` | `(a: int) -> int` | Bitwise NOT (`~a`). |
| `shift_left` | `(a: int, n: int) -> int` | Logical left shift (`a << n`). |
| `shift_right` | `(a: int, n: int) -> int` | Logical right shift (`a >> n`). |
| `popcount` | `(a: int) -> int` | Number of set bits via `x & (x-1)` loop. |
| `get_bit` | `(a: int, i: int) -> int` | Returns `1` if bit `i` of `a` is set, else `0`. |
| `set_bit` | `(a: int, i: int) -> int` | Returns `a` with bit `i` set to 1. |
| `clear_bit` | `(a: int, i: int) -> int` | Returns `a` with bit `i` cleared. |

Example:

```eig
import std.bitwise

let x: int = std.bitwise.popcount(0b101101)
print x   # 4
```

---

## std.collections

Array helpers for `array<int>`. Bodies are stubs (return `0`); the runtime
provides native implementations where available.

```eig
eigen 1.0
module std.collections
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `append_int` | `(arr: array<int>, val: int) -> int` | Append an int. Returns `0`; the array is mutated in place. |
| `remove_at` | `(arr: array<int>, idx: int) -> int` | Remove the element at `idx`. Returns `0`. |

---

## std.complex

Complex-number arithmetic. Complex values are represented as `array<float>`
of length 2 (`[real, imag]`).

```eig
eigen 1.0
module std.complex
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `real` | `(c: array<float>) -> float` | Real part = `c[0]`. |
| `imag` | `(c: array<float>) -> float` | Imaginary part = `c[1]`. |
| `make` | `(re: float, im: float) -> array<float>` | Construct `[re, im]`. |
| `add` | `(a, b: array<float>) -> array<float>` | Complex addition. |
| `mul` | `(a, b: array<float>) -> array<float>` | Complex multiplication: `(a₀b₀ - a₁b₁, a₀b₁ + a₁b₀)`. |
| `abs` | `(c: array<float>) -> float` | Modulus `√(re² + im²)` via `std.math.sqrt`. |
| `conjugate` | `(c: array<float>) -> array<float>` | Returns `[c[0], -c[1]]`. |
| `phase` | `(c: array<float>) -> float` | Stub — returns `0.0`. |
| `append_float` | `(arr, val: array<float>, float) -> array<float>` | Helper, stub body. |

Example:

```eig
import std.complex

let z: array<float> = std.complex.make(3.0, 4.0)
let mag: float = std.complex.abs(z)        # 5.0
```

---

## std.hashing

String hashing. Uses FNV-1a constants.

```eig
eigen 1.0
module std.hashing
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `hash` | `(s: string) -> int` | FNV-1a 32-bit hash. Returns the absolute value (non-negative). |
| `hash64` | `(s: string) -> int` | FNV-1a 64-bit rolling hash. Non-negative. |
| `combine` | `(a: int, b: int) -> int` | `a * 31 + b` for compound keys. |
| `sha256` | `(s: string) -> string` | Stub — returns `"sha256-stub-not-implemented-in-pure-eigen"`. Use FFI for a real digest. |
| `char_at` | `(s: string, i: int) -> int` | Stub — returns `0`. |

Note: `hash` and `hash64` iterate `while i < n` where `n` is initialized to
`0`, so over a non-empty string the loop body does not execute and the
initial offset constant is returned. This is the actual behavior of the
stdlib as shipped.

---

## std.io

File I/O. All bodies are stubs.

```eig
eigen 1.0
module std.io
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `read_file` | `(filepath: string) -> string` | Stub — returns `""`. |
| `write_file` | `(filepath: string, content: string) -> int` | Stub — returns `0`. |
| `print_format` | `(fmt: string, val: int) -> int` | Stub — returns `0`. |

---

## std.linear_algebra

Vector and matrix helpers for `array<float>` and `array<array<float>>`.
Most functions are stubs; see the "Behavior" column.

```eig
eigen 1.0
module std.linear_algebra
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `dot` | `(a, b: array<float>) -> float` | Dot product (loop, but `n` is stubbed to `0`). Returns `0.0` for non-empty input. |
| `transpose` | `(m: array<array<float>>) -> array<array<float>>` | Stub — returns `m` unchanged. |
| `identity` | `(n: int) -> array<array<float>>` | Builds the n×n identity using `append_float`/`append_row`. Since those are stubs, the returned matrix is empty. |
| `matmul` | `(a, b: array<array<float>>) -> array<array<float>>` | Stub — returns `a`. |
| `inverse` | `(m: array<array<float>>) -> array<array<float>>` | Stub — returns `m`. |
| `norm` | `(v: array<float>) -> float` | L2 norm via `std.math.sqrt`. Same `n=0` caveat as `dot`. |
| `vec_add` | `(a, b: array<float>) -> array<float>` | Stub — returns `a`. |
| `vec_scale` | `(v: array<float>, s: float) -> array<float>` | Stub — returns `v`. |
| `append_float` | `(arr, val) -> array<float>` | Stub. |
| `append_row` | `(m, row) -> array<array<float>>` | Stub. |

---

## std.math

Floating-point math. `sqrt` and `abs` are implemented in pure Eigen;
transcendental functions throw because they require native math support.

```eig
eigen 1.0
module std.math
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `sin` | `(x: float) -> float` | Throws `"math.eig::sin requires native math support"`. |
| `cos` | `(x: float) -> float` | Throws; same reason. |
| `tan` | `(x: float) -> float` | Throws; same reason. |
| `sqrt` | `(x: float) -> float` | Newton-Raphson iteration, 30 steps. Throws on `x < 0`. Returns `0.0` for `x == 0.0`. |
| `log` | `(x: float) -> float` | Throws; native support required. |
| `exp` | `(x: float) -> float` | Throws; native support required. |
| `abs` | `(x: float) -> float` | Returns `-x` if `x < 0`, else `x`. |

Example:

```eig
import std.math

let r: float = std.math.sqrt(2.0)     # ≈ 1.414213562373095
let a: float = std.math.abs(-3.5)      # 3.5
```

---

## std.quantum_helpers

State-vector construction and measurement helpers operating on
`array<float>` statevectors (interleaved real/imag is *not* the convention —
these helpers treat the array as a probability amplitude list of length
`2ⁿ`). Several functions have `n = 0` stub initialization.

```eig
eigen 1.0
module std.quantum_helpers
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `uniform_state` | `(n: int) -> array<float>` | Stub — returns `[]`. |
| `normalize` | `(state: array<float>) -> array<float>` | Computes `s = Σ state[i]²` but does not actually rescale. Returns `state` unchanged. |
| `probability` | `(state: array<float>, outcome: int) -> float` | Returns `state[outcome]²`. |
| `bell_state` | `() -> array<float>` | Returns `[0.7071, 0, 0, 0.7071]` (the Bell state `(\|00> + \|11>)/√2`). |
| `ghz_state` | `(n: int) -> array<float>` | Returns a length-`2ⁿ` array with `0.7071` at indices `0` and `2ⁿ - 1`, zero elsewhere. |
| `measure` | `(state: array<float>, seed: float) -> int` | Inverse-CDF sampling. Returns the sampled outcome index. (Same `n=0` caveat — returns `0`.) |
| `num_qubits` | `(state: array<float>) -> int` | Returns `0` (stub body). |
| `append_float` | `(arr, val) -> array<float>` | Stub. |

---

## std.random

Random-number generation. Bodies are stubs.

```eig
eigen 1.0
module std.random
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `rand_float` | `() -> float` | Stub — returns `0.0`. |
| `rand_int` | `(min_val: int, max_val: int) -> int` | Stub — returns `0`. |

---

## std.sorting

Integer array sorting and search. `sort` and `sort_desc` use insertion sort;
`binary_search` is implemented; `merge_sort` and `quick_sort` are partial
stubs (return the input unchanged).

```eig
eigen 1.0
module std.sorting
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `sort` | `(arr: array<int>) -> array<int>` | Insertion sort, ascending. Mutates `arr` in place. (Loop bound `n = 0` — see note.) |
| `sort_desc` | `(arr: array<int>) -> array<int>` | Insertion sort, descending. Same caveat. |
| `binary_search` | `(arr: array<int>, target: int) -> int` | Classic binary search. Returns the index or `-1`. (`hi` is stubbed to `0`.) |
| `merge_sort` | `(arr: array<int>) -> array<int>` | Stub — returns `arr`. |
| `quick_sort` | `(arr: array<int>, lo: int, hi: int) -> int` | Partial Lomuto-partition implementation. Returns `0`. |

Note: as shipped, `sort`/`sort_desc`/`binary_search` initialize their loop
bound to `0`, so the loop body does not execute over the user's array — the
input is returned unchanged. Callers needing real sorting should bind to a
native helper via FFI or implement the sort inline.

---

## std.stats

Descriptive statistics. Bodies are stubs.

```eig
eigen 1.0
module std.stats
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `mean` | `(arr: array<float>) -> float` | Stub — returns `0.0`. |
| `variance` | `(arr: array<float>) -> float` | Stub — returns `0.0`. |

---

## std.string

String helpers. Bodies are stubs.

```eig
eigen 1.0
module std.string
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `concat` | `(s1: string, s2: string) -> string` | Stub — returns `""`. |
| `format_int` | `(val: int) -> string` | Stub — returns `""`. |

For string interpolation use the language-level `"text ${expr}"` syntax
instead (handled by `StringInterpolationNode` in the parser).

---

## std.time

Time helpers. Bodies are stubs.

```eig
eigen 1.0
module std.time
```

| Function | Signature | Behavior |
|----------|-----------|----------|
| `now` | `() -> float` | Stub — returns `0.0`. |
| `sleep` | `(ms: int) -> int` | Stub — returns `0`. |

---

## Calling stdlib functions

Two equivalent syntaxes are accepted:

```eig
import std.math

let r: float = std.math.sqrt(2.0)
```

```eig
let r: float = sqrt(2.0)     # short name resolved via EigenVM._STD_MAPPING
```

The short form works because `EigenVM._STD_MAPPING` maps unqualified names
(`sin`, `sqrt`, `mean`, `now`, ...) to fully-qualified module paths
(`std.math.sin`, `std.math.sqrt`, `std.stats.mean`, `std.time.now`, ...).
