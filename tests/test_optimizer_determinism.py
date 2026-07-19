"""
Audit §2.3 — byte-identical compile + determinism regression tests.

The previous optimizer worklist used `set.pop()` (Python) and the Rust
optimizer popped an arbitrary element from a `HashSet<usize>`. Both choices
made the set of nodes that survive optimization depend on hash-seed and
process-local memory layout, producing different optimized graphs from
identical source across runs. The canonicalizer's `topological_sort` was
already sorted by id, but the *contents* that it visited could differ.

These tests pin down the fix:

  * The canonical hash for the same input must be byte-identical across
    multiple invocations (in-process determinism).

  * When run in subprocesses seeded with different `PYTHONHASHSEED` values,
    the canonical hash must be identical across processes (cross-process
    determinism — the actual audit requirement).

  * The optimizer must not regress either of those properties on simple
    peephole / commutation / rotation-merging patterns.
"""

import os
import subprocess
import sys
import textwrap
import unittest


def _build_optimizable_graph():
    """A small graph exercising all the optimizer's rewrite rules."""
    from src.ir.ir_graph import EQIRGraph
    g = EQIRGraph()
    # Two qubits
    g.add_operation('ALLOC', targets=['q0'])
    g.add_operation('ALLOC', targets=['q1'])
    # Self-inverse H H -> identity (Rule 1)
    g.add_operation('GATE', gate_name='H', targets=['q0'])
    g.add_operation('GATE', gate_name='H', targets=['q0'])
    # Rotation merge: RX(0.3) + RX(0.4) -> RX(0.7) (Rule 2)
    g.add_operation('GATE', gate_name='RX', targets=['q0'], args=[0.3])
    g.add_operation('GATE', gate_name='RX', targets=['q0'], args=[0.4])
    # Trivial rotation: RZ(0) is dead -> eliminated (Rule 3)
    g.add_operation('GATE', gate_name='RZ', targets=['q0'], args=[0.0])
    # Peephole: H -> X -> H becomes Z (Rule 4)
    g.add_operation('GATE', gate_name='H', targets=['q0'])
    g.add_operation('GATE', gate_name='X', targets=['q0'])
    g.add_operation('GATE', gate_name='H', targets=['q0'])
    # Peephole: S -> S becomes Z (Rule 5)
    g.add_operation('GATE', gate_name='S', targets=['q1'])
    g.add_operation('GATE', gate_name='S', targets=['q1'])
    # Commutation: Z -> CNOT(q0,q1) -> Z on q0 (Rule 6)
    g.add_operation('GATE', gate_name='Z', targets=['q0'])
    g.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q1'])
    g.add_operation('GATE', gate_name='Z', targets=['q0'])
    # Terminal measurement
    g.add_operation('MEASURE', targets=['q1'], cbit_name='c1')
    return g


class TestOptimizerDeterminism(unittest.TestCase):
    def _hash(self):
        from src.canonicalizer import Canonicalizer
        g = _build_optimizable_graph()
        return Canonicalizer().hash_circuit(g)

    def test_in_process_deterministic_hash(self):
        """Multiple invocations within the same process produce equal hashes."""
        h1 = self._hash()
        h2 = self._hash()
        h3 = self._hash()
        self.assertEqual(h1, h2)
        self.assertEqual(h2, h3)
        self.assertEqual(len(h1), 64)

    def test_optimizer_runs_full_pass_without_crash(self):
        """Optimizer completes a full pass on the optimizable graph."""
        from src.ir.optimizer import EQIROptimizer
        from copy import deepcopy
        g = deepcopy(_build_optimizable_graph())
        opt = EQIROptimizer()
        before = len(g.nodes)
        opt.optimize(g)
        # All rewrites should remove at least a few nodes (Rule 1 alone removes 2).
        self.assertLess(len(g.nodes), before)
        self.assertGreater(opt.iterations_count, 0)

    def test_optimizer_idempotent_second_pass_is_no_op(self):
        """Optimizing an already-optimized graph is a fixed point.

        (Both the Python and Rust optimizer code paths overwrite
        `optimizations_count` on each call, so we cannot compare that counter
        for idempotence. The genuine invariant is that the set of surviving
        nodes - and therefore the canonical hash - does not change.)
        """
        from src.ir.optimizer import EQIROptimizer
        from src.canonicalizer import Canonicalizer
        from copy import deepcopy
        g = deepcopy(_build_optimizable_graph())
        opt = EQIROptimizer()
        opt.optimize(g)
        first_node_count = len(g.nodes)
        first_hash = Canonicalizer().hash_circuit(g)

        # Second pass should neither remove nodes nor change the hash.
        opt.optimize(g)
        self.assertEqual(len(g.nodes), first_node_count)
        self.assertEqual(Canonicalizer().hash_circuit(g), first_hash)


class TestCrossProcessDeterminism(unittest.TestCase):
    """
    Spawn fresh Python interpreters under each of several PYTHONHASHSEED
    values and confirm the canonical hash is identical across them. This is
    the actual audit requirement: the canonical hash of a fixed source file
    must not depend on the interpreter's hash-seed choice.
    """

    GRAPH_BUILDER_SRC = textwrap.dedent("""
        import sys, os
        sys.path.insert(0, {repo!r})
        from src.ir.ir_graph import EQIRGraph
        from src.canonicalizer import Canonicalizer

        g = EQIRGraph()
        g.add_operation('ALLOC', targets=['q0'])
        g.add_operation('ALLOC', targets=['q1'])
        g.add_operation('GATE', gate_name='H', targets=['q0'])
        g.add_operation('GATE', gate_name='H', targets=['q0'])
        g.add_operation('GATE', gate_name='RX', targets=['q0'], args=[0.3])
        g.add_operation('GATE', gate_name='RX', targets=['q0'], args=[0.4])
        g.add_operation('GATE', gate_name='RZ', targets=['q0'], args=[0.0])
        g.add_operation('GATE', gate_name='H', targets=['q0'])
        g.add_operation('GATE', gate_name='X', targets=['q0'])
        g.add_operation('GATE', gate_name='H', targets=['q0'])
        g.add_operation('GATE', gate_name='S', targets=['q1'])
        g.add_operation('GATE', gate_name='S', targets=['q1'])
        g.add_operation('GATE', gate_name='Z', targets=['q0'])
        g.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q1'])
        g.add_operation('GATE', gate_name='Z', targets=['q0'])
        g.add_operation('MEASURE', targets=['q1'], cbit_name='c1')
        print(Canonicalizer().hash_circuit(g), end='')
    """).format(repo=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    def _hash_under_seed(self, seed):
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = str(seed)
        # -u: unbuffered so print is flushed before the process exits
        proc = subprocess.run(
            [sys.executable, '-u', '-c', self.GRAPH_BUILDER_SRC],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            self.fail(f"hasher exited {proc.returncode}; stderr=\n{proc.stderr}")
        return proc.stdout

    def test_identical_hash_across_pythonhashseed_values(self):
        seeds = ['0', '1', '2', '1337', 'random']
        hashes = {s: self._hash_under_seed(s) for s in seeds}
        all_hashes = list(hashes.values())
        for i in range(len(all_hashes) - 1):
            self.assertEqual(
                all_hashes[i], all_hashes[-1],
                msg=f"hash diverged across PYTHONHASHSEED values: {hashes}",
            )


class TestToDictDeterministicSerialization(unittest.TestCase):
    """The serialized node list (used as input to the Rust optimizer)
    must be byte-identical across runs of the same input graph."""

    def _to_dict_bytes(self):
        g = _build_optimizable_graph()
        import io
        io.StringIO()
        # Stable key ordering in the dicts + sorted children_ids already
        # guarantees this; we additionally assert by comparing the raw
        # repr in two independent builds.
        d1 = g.to_dict()
        d2 = (_build_optimizable_graph()).to_dict()
        import json
        return json.dumps(d1, sort_keys=True), json.dumps(d2, sort_keys=True)

    def test_to_dict_is_deterministic(self):
        a, b = self._to_dict_bytes()
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
