"""§3.1 — Eigen Language Extensions package.

This subpackage collects the surface-level / API-envelope
implementations of the language-extension items in sol.md §3.1:

  * `default_named_variadic` — Default / Named / Variadic arguments.
  * `pattern_matching_guards` — Pattern matching with optional guard
    expressions per case.
  * `algebraic_data_types` — Algebraic data types (sum types, tagged
    unions), plus a runtime pattern-matching interpreter for ADTs.
  * `module_system` — Module path resolution, visibility, re-exports.
  * `macro_system` — AST-level macro registry and expansion.
  * `operator_overloading` — User-defined operator dispatch table.
  * `async_await` — `AsyncTask` wrapper providing the API surface
    for `async func` / `await` (executed synchronously).

Each module is independently importable. The package exports a
small "convenience" surface so callers can write
`from src.language_extensions import FunctionSignature, validate_call`.
"""
from .default_named_variadic import (  # noqa: F401
    Parameter,
    ParameterKind,
    FunctionSignature,
    FunctionSignatureError,
    ArgumentCountError,
    UnknownKeywordArgumentError,
    DuplicateArgumentError,
    MissingArgumentError,
    bind_arguments,
    positional_list,
    validate_call,
    parameter,
    signature,
)
from .pattern_matching_guards import (  # noqa: F401
    Pattern,
    LiteralPattern,
    WildcardPattern,
    RangePattern,
    IsInstancePattern,
    ConstructorPattern,
    BindPattern,
    Guard,
    MatchCase,
    match_with_guards,
)
from .algebraic_data_types import (  # noqa: F401
    Variant,
    AlgebraicDataType,
    ADTValue,
    ADTValueError,
    ADTRegistry,
    StandardADTs,
    make_option_adt,
    make_result_adt,
    make_either_adt,
    make_list_adt,
)
from .module_system import (  # noqa: F401
    ModuleVisibility,
    ModuleVisibilityError,
    ModuleLookupError,
    CircularReExportError,
    ModuleSymbol,
    Module,
    ModuleRegistry,
)
from .macro_system import (  # noqa: F401
    MacroExpansionError,
    MacroContext,
    Macro,
    MacroTable,
    MacroExpander,
    prelude_macros,
)
from .operator_overloading import (  # noqa: F401
    Operator,
    OperatorOverload,
    OperatorOverloadTable,
    OperatorOverloadError,
)
from .async_await import (  # noqa: F401
    AsyncTask,
    AsyncError,
    AsyncStateError,
    await_,
    yield_,
)
from .type_system_extensions import (  # noqa: F401
    TypeKind,
    TypeRef,
    Protocol,
    UnionType,
    OptionalType,
    ResultType,
    ConstGeneric,
    HigherKindedType,
)
from .quantum_constructs import (  # noqa: F401
    FeedbackError,
    MidCircuitFeedback,
    feed_forward,
    RepeatUntilSuccess,
    RusFailure,
    QecCode,
    repetition_code_x,
    repetition_code_z,
    shor_code,
    steane_code,
    PulseEntry,
    PulseSchedule,
    DynamicStep,
    DynamicStepKind,
    DynamicCircuit,
    conditional_gate,
)
