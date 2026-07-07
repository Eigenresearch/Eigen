# Resource Estimator for Eigen programs
#
# Audit §1.5: the previous estimator only counted CNOTs, total gates,
# and T-gates. Real fault-tolerant costing (cf. Azure Resource Estimator)
# depends on Toffoli counts, rotation counts (which decompose to T gates
# in the synthesis model), two-qubit-gate counts (not just CNOT), and the
# SWAP overhead introduced by routing.
#
# The estimator now exposes these metrics. It dispatches gate metadata
# through the shared `gate_registry` so adding a new gate to the registry
# automatically flows the right classification here, instead of needing
# a parallel `if/elif` chain (same audit fix as the exporter backends).
import math
from src.ir.ir_graph import EQIRGraph, EQIRNode
from src.backend.gate_registry import (
    CLIFFORD_GATES,
    GATE_QUBIT_COUNT,
    get_gate_spec,
    CONTROLLED_ROTATION_GATES,
    ROTATION_GATES,
)

# All rotation gates (single + controlled) - these decompose into Clifford+T
# during fault-tolerant synthesis. Counted separately from Clifford/T gates
# so callers can apply their own synthesis cost model.
ALL_ROTATION_GATES = ROTATION_GATES | CONTROLLED_ROTATION_GATES

# Toffoli-equivalent gates: CCX and CSWAP (the latter costs ~2 Toffoli
# equivalents in the standard decomposition; we report both as raw counts
# and let the caller apply their own scaling).
TOFFOLI_GATES = {"CCX", "CSWAP"}


class ResourceEstimator:
    """Computes compile-time resource estimates for an EQIR graph.

    The estimator is *routing-optional*: callers that have already run
    Eigen's router and want to include the SWAP overhead can provide
    `swaps_inserted=int` to the `estimate` call. If absent, the estimator
    assumes the input graph already accounts for SWAP operations that
    should be counted (any `SWAP` gates present in the graph are counted
    under `two_qubit_count` and `swap_count` already).
    """

    def estimate(self, graph: EQIRGraph, *, swaps_inserted: int = 0) -> dict:
        nodes = graph.topological_sort()

        qubits: set[str] = set()
        cbits: set[str] = set()
        gate_count = 0
        single_qubit_count = 0
        two_qubit_count = 0
        three_qubit_count = 0
        cnot_count = 0
        clifford_count = 0
        t_count = 0
        toffoli_count = 0
        rotation_count = 0
        swap_count = 0
        measurement_count = 0

        # Initialize per-node depth trackers for the four metric streams.
        depths = {node.id: 0 for node in nodes}
        t_depths = {node.id: 0 for node in nodes}
        two_qubit_depths = {node.id: 0 for node in nodes}
        measurement_depths = {node.id: 0 for node in nodes}

        for node in nodes:
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
                continue
            if node.type == 'MEASURE':
                if node.cbit_name:
                    cbits.add(node.cbit_name)
                measurement_count += 1
                parent_ids = [p.id for p in node.parents if p.id in measurement_depths]
                base = max((measurement_depths[pid] for pid in parent_ids), default=0)
                measurement_depths[node.id] = base + 1
                continue
            if node.type != 'GATE':
                continue

            # All gate metadata flows from the shared registry; this avoids
            # the parallel-if-elif-per-quantifier bug class the audit named.
            gate_count += 1
            g_name = node.gate_name
            spec = get_gate_spec(g_name)

            n_qubits = (spec.qubit_count if spec is not None
                        else GATE_QUBIT_COUNT.get(g_name, 1))
            if n_qubits == 1:
                single_qubit_count += 1
            elif n_qubits == 2:
                two_qubit_count += 1
            elif n_qubits == 3:
                three_qubit_count += 1

            if g_name == 'CNOT':
                cnot_count += 1
            if g_name == 'SWAP':
                swap_count += 1
            if g_name in CLIFFORD_GATES:
                clifford_count += 1
            if g_name == 'T':
                t_count += 1
            if g_name in TOFFOLI_GATES:
                toffoli_count += 1
            if g_name in ALL_ROTATION_GATES:
                rotation_count += 1

            # Depth trackers: 1 gate per timestep for any gate; +1 T-depth
            # only for T gates (Clifford+T decomposition assumption).
            # +1 two-qubit depth only when this gate is 2-qubit or 3-qubit.
            parent_ids = [p.id for p in node.parents if p.id in depths]
            base_depth = max((depths[pid] for pid in parent_ids), default=0)
            base_t_depth = max((t_depths[pid] for pid in parent_ids), default=0)
            base_2q_depth = max((two_qubit_depths[pid] for pid in parent_ids), default=0)
            is_t = (g_name == 'T')
            is_2q = n_qubits >= 2
            depths[node.id] = base_depth + 1
            t_depths[node.id] = base_t_depth + (1 if is_t else 0)
            two_qubit_depths[node.id] = base_2q_depth + (1 if is_2q else 0)

        total_depth = max(depths.values()) if depths else 0
        total_t_depth = max(t_depths.values()) if t_depths else 0
        total_2q_depth = max(two_qubit_depths.values()) if two_qubit_depths else 0
        total_measurement_depth = max(measurement_depths.values()) if measurement_depths else 0

        # Audit specifically called out routing overhead: include both the
        # SWAP gates present in the EQIR graph (`swap_count`) AND any extra
        # SWAPs that the caller reports were inserted after a routing pass
        # (`swaps_inserted`). Callers using the bare estimator on a
        # pre-routing graph see `0` for the second term.
        total_swap_count = swap_count + max(0, int(swaps_inserted))

        # Approximate T-count contribution from rotation synthesis.
        # Exact synthesis is gate-fidelity dependent, but the common
        # Heisenberg-style synthesis for `R_theta` uses ~ ceil(log2(1/eps))
        # T gates for an infidelity of `eps`; for a default eps of 1e-3,
        # ~10 T gates per rotation is a typical engineering estimate.
        # Reported separately as `rotation_t_estimate` so the user can
        # override or recompute it against their own accuracy target.
        rotation_t_estimate = rotation_count * 10

        return {
            # Qubit & cbit allocation
            'logical_qubits': len(qubits),
            'classical_bits': len(cbits),
            # Depths
            'circuit_depth': total_depth,
            't_depth': total_t_depth,
            'two_qubit_depth': total_2q_depth,
            'measurement_depth': total_measurement_depth,
            # Counts
            'gate_count': gate_count,
            'single_qubit_count': single_qubit_count,
            'two_qubit_count': two_qubit_count,
            'three_qubit_count': three_qubit_count,
            'cnot_count': cnot_count,
            'toffoli_count': toffoli_count,
            'swap_count': swap_count,
            'total_swap_count': total_swap_count,
            'post_routing_swaps_count': max(0, int(swaps_inserted)),
            't_count': t_count,
            'rotation_count': rotation_count,
            'rotation_t_estimate': rotation_t_estimate,
            'clifford_count': clifford_count,
            'measurements': measurement_count,
            # Backwards-compatible aliases (kept so existing callers /
            # CLI commands see the field they expect).
            'clifford_gates': clifford_count,
        }
