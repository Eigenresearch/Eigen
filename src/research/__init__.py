"""P3 §12.1 quantum research tools subpackage.

Submodules:
  * `src.research.quantum_volume`
  * `src.research.randomized_benchmarking`
  * `src.research.entanglement_witness`

We don't re-export functions from here to avoid name-shadows on the
submodules — the test suite accesses them as
`from src.research import quantum_volume as qv` (etc.). Importing
a function with the same name as a submodule into the package
namespace would have made `src.research.X` resolve to the function
instead of the submodule.
"""
