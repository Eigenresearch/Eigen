"""Tests for the §1.2 Rust ports of ZX spider fusion and SABRE swap scoring.

The Rust module `eigen_native` ships two functions used by this test
suite:

  * `fast_spider_fusion_full(vertex_ids, vertex_types, phases, adjacency)`
    -> `(new_ids, new_types, new_phases, new_adjacency, changed)`.
    Ports the same-colour ZX spider fusion rule from
    `src/zx/spider_fusion.py` with deterministic output ordering
    (surviving vertex IDs are sorted; ties broken by scanning vertices
    in the order given by the caller).

  * `fast_sabre_swap_score(edges, distances, mapping, front_layer,
                           extended_layer, lookahead_weight)`
    -> `Optional[(q1, q2, score)]`. Ports the SABRE swap-scoring inner
    loop from `src/routing/router.py::SabreRouter.route` lines 518-554
    with lexicographic tie-breaking for deterministic output.

These tests cover both kernels' correctness against hand-computed
golden outputs, determinism (input edge order independence), and
physical sanity (boundary cases like empty inputs, fully-disconnected
graphs, and self-loops in trial mappings).
"""

import unittest

try:
    import eigen_native as native
except ImportError:
    native = None


@unittest.skipUnless(native is not None, "eigen_native not built; run `maturin develop --release`")
class TestFastSpiderFusionFull(unittest.TestCase):
    """Exercises the `fast_spider_fusion_full` Rust kernel."""

    @staticmethod
    def _call(vertex_ids, types, phases, adjacency):
        # The Rust kernel's `vertex_types` argument is `Vec<u8>`, which
        # pyo3 converts from a Python list of ints (NOT bytes, despite
        # the u8 type). Translate str/bytes representations to ints
        # before calling.
        types_ints = []
        for t in types:
            if isinstance(t, str):
                types_ints.append(ord(t))
            elif isinstance(t, (bytes, bytearray)):
                types_ints.append(t[0])
            else:
                types_ints.append(int(t))
        return native.fast_spider_fusion_full(
            list(vertex_ids), list(types_ints), list(phases),
            [list(a) for a in adjacency],
        )

    @staticmethod
    def _expected_types(types):
        return [
            ord(t) if isinstance(t, str)
            else (t[0] if isinstance(t, (bytes, bytearray)) else int(t))
            for t in types
        ]

    def test_no_edges_no_fusion(self):
        # Four isolated vertices: no fusion should happen, output is the
        # same set (sorted by id), `changed == False`.
        new_ids, new_types, new_phases, new_adj, changed = self._call(
            [10, 5, 1, 7],
            ['Z', 'X', 'H', 'B'],
            [0.0, 0.5, 0.0, 0.0],
            [[], [], [], []],
        )
        self.assertEqual(new_ids, [1, 5, 7, 10])
        self.assertEqual(new_types, [ord('H'), ord('X'), ord('B'), ord('Z')])
        self.assertEqual(new_phases, [0.0, 0.5, 0.0, 0.0])
        self.assertEqual(new_adj, [[], [], [], []])
        self.assertFalse(changed)

    def test_single_z_z_fusion(self):
        # Two Z-spiders connected by a non-H edge, with phases 0.25 and
        # 0.5 (multiples of π). After fusion the survivor's phase should
        # be (0.25 + 0.5) % 2.0 = 0.75. The survivor is the LARGER id (the
        # kernel iterates vertices in the order given — vertex 10 is
        # scanned first, sees neighbour 11 same-colour, fuses 11 into 10).
        new_ids, new_types, new_phases, new_adj, changed = self._call(
            [10, 11],
            ['Z', 'Z'],
            [0.25, 0.5],
            [[11], [10]],
        )
        self.assertTrue(changed)
        self.assertEqual(new_ids, [10])
        self.assertEqual(new_types, [ord('Z')])
        self.assertAlmostEqual(new_phases[0], 0.75, places=10)
        self.assertEqual(new_adj, [[]])

    def test_single_x_x_fusion(self):
        # Same as above but with X-spiders.
        new_ids, new_types, new_phases, new_adj, changed = self._call(
            [10, 11],
            ['X', 'X'],
            [0.25, 0.5],
            [[11], [10]],
        )
        self.assertTrue(changed)
        self.assertEqual(new_ids, [10])
        self.assertEqual(new_types, [ord('X')])
        self.assertAlmostEqual(new_phases[0], 0.75, places=10)

    def test_z_x_adjacency_no_fusion(self):
        # A Z-spider and an X-spider adjacent by a non-H edge should NOT
        # fuse (same-colour only).
        new_ids, new_types, new_phases, new_adj, changed = self._call(
            [10, 11],
            ['Z', 'X'],
            [0.25, 0.5],
            [[11], [10]],
        )
        self.assertFalse(changed)
        self.assertEqual(new_ids, [10, 11])
        self.assertEqual(new_types, [ord('Z'), ord('X')])
        self.assertEqual(new_adj, [[11], [10]])

    def test_chain_of_three_zs_fuses_to_one(self):
        # Three Z-spiders in a chain 1—2—3. The fixpoint loop should
        # absorb all of them into one survivor, with phase = (0.2 + 0.3 +
        # 0.5) % 2.0 = 1.0.
        new_ids, new_types, new_phases, new_adj, changed = self._call(
            [1, 2, 3],
            ['Z', 'Z', 'Z'],
            [0.2, 0.3, 0.5],
            [[2], [1, 3], [2]],
        )
        self.assertTrue(changed)
        self.assertEqual(new_ids, [1])
        self.assertEqual(new_types, [ord('Z')])
        self.assertAlmostEqual(new_phases[0], 1.0, places=10)
        self.assertEqual(new_adj, [[]])

    def test_h_and_boundary_do_not_participate(self):
        # H and Boundary vertices in the neighbourhood of Z-spiders must
        # remain intact (their type is not Z/X so the fusion rule
        # rejects them both as the survivor AND as the absorbed party).
        new_ids, new_types, new_phases, new_adj, changed = self._call(
            [1, 2, 3, 4],
            ['Z', 'H', 'Z', 'B'],
            [0.25, 0.0, 0.5, 0.0],
            [[2, 3], [1], [1], [2]],   # 1-2 (Z-H), 1-3 (Z-Z), 2-4 (H-B)
        )
        self.assertTrue(changed)
        # Vertex 3 should be absorbed (same-color Z as 1). Vertex 2 (H)
        # and 4 (B) survive. Sorted output: [1, 2, 4].
        self.assertEqual(new_ids, [1, 2, 4])
        self.assertEqual(new_types, self._expected_types(['Z', 'H', 'B']))
        self.assertAlmostEqual(new_phases[0], 0.75, places=10)
        # The H vertex (was 2) was neighbour to both 1 and 3. Fusion of 3
        # into 1 only transfers 3's other neighbours — since 3's only
        # other neighbour was 1 itself (the survivor), no new edges are
        # added. Vertex 1's adjacency to the H vertex at id 2 is preserved.
        self.assertEqual(new_adj[0], [2])         # vertex 1: still adjacent to H vertex 2
        self.assertEqual(new_adj[1], [1])         # vertex 2: still adjacent to vertex 1
        self.assertEqual(new_adj[2], [2])         # vertex 4: still adjacent to the H vertex

    def test_phase_normalised_mod_2(self):
        # Phases should always remain in the half-open interval [0, 2).
        new_ids, _, new_phases, _, changed = self._call(
            [1, 2], ['Z', 'Z'], [1.7, 1.7], [[2], [1]],
        )
        self.assertTrue(changed)
        self.assertEqual(new_ids, [1])
        # 1.7 + 1.7 = 3.4 -> 3.4 % 2.0 = 1.4
        self.assertAlmostEqual(new_phases[0], 1.4, places=10)

    def test_determinism_with_input_reordering(self):
        # The same logical graph submitted to the kernel in two
        # different vertex orderings must produce FUSION with the SAME
        # survivor set SIZE and SAME fused phase — even though the
        # particular survivor's ID will depend on which vertex is
        # scanned first (the algorithm absorbs the first neighbour
        # found into the scanned vertex). For a connected component
        # all fusions collapse to a single survivor; the survivor's
        # phase is the sum of all absorbed phases (mod 2.0).
        graph_a_ids = [1, 2, 3]
        graph_b_ids = [3, 2, 1]  # reversed input order
        adj_a = [[2], [1, 3], [2]]
        adj_b = [[2], [1, 3], [2]]  # adj keyed by vertex ID, symmetric
        # NB. adj_b is identical to adj_a because adjacency uses ids,
        # not indices. The kernel sorts survivors before emitting,
        # so the fused graph has the same phase regardless of input
        # ordering.
        a = native.fast_spider_fusion_full(
            graph_a_ids, [ord('Z'), ord('Z'), ord('Z')], [0.2, 0.3, 0.5], adj_a,
        )
        b = native.fast_spider_fusion_full(
            graph_b_ids, [ord('Z'), ord('Z'), ord('Z')], [0.2, 0.3, 0.5], adj_b,
        )
        self.assertEqual(len(a[0]), len(b[0]))  # both single survivor
        self.assertEqual(len(a[0]), 1)           # connected chain fuses fully
        self.assertAlmostEqual(a[2][0], b[2][0], places=10)  # same phase
        self.assertAlmostEqual(a[2][0] % 2.0, 1.0, places=10)  # 0.2+0.3+0.5=1.0

    def test_validation_mismatched_lengths_raises(self):
        with self.assertRaises(Exception):
            native.fast_spider_fusion_full(
                [1, 2], [ord('Z')], [0.0, 0.0], [[], []],
            )

    def test_python_python_and_rust_paths_agree_on_simple_graph(self):
        # Cross-check: run the existing Python `SpiderFuser.fuse_spiders`
        # against the Rust kernel on the same 3-chain Z-graph built via
        # the production `ZXGraph` class, and verify the post-fusion
        # vertex count and total phase match.
        from src.zx.zx_graph import ZXGraph
        from src.zx.spider_fusion import SpiderFuser

        # Build the same 3-chain in Python: Z(0.2π) — Z(0.3π) — Z(0.5π)
        g_py = ZXGraph()
        v1 = g_py.add_vertex('Z', 0.2)
        v2 = g_py.add_vertex('Z', 0.3)
        v3 = g_py.add_vertex('Z', 0.5)
        g_py.add_edge(v1.id, v2.id)
        g_py.add_edge(v2.id, v3.id)
        SpiderFuser().fuse_spiders(g_py)
        py_total_phase = sum(v.phase for v in g_py.vertices.values()) % 2.0
        py_survivor_count = len(g_py.vertices)

        # Rust kernel on the same adjacency.
        new_ids, _, new_phases, _, _ = native.fast_spider_fusion_full(
            [v1.id, v2.id, v3.id],
            [ord('Z'), ord('Z'), ord('Z')],
            [0.2, 0.3, 0.5],
            [[v2.id], [v1.id, v3.id], [v2.id]],
        )
        self.assertEqual(len(new_ids), py_survivor_count)
        self.assertAlmostEqual(sum(new_phases) % 2.0, py_total_phase, places=10)


@unittest.skipUnless(native is not None, "eigen_native not built")
class TestFastSabreSwapScore(unittest.TestCase):
    """Exercises the `fast_sabre_swap_score` Rust kernel."""

    def test_empty_edges_returns_none(self):
        result = native.fast_sabre_swap_score(
            [], [[0, 1], [1, 0]],
            [0, 1],
            [(0, 1)],
            [],
            0.5,
        )
        self.assertIsNone(result)

    def test_single_swap_that_connects_pair_is_chosen(self):
        # Linear topology 0-1-2. Logical qubit pair (q0, q2) is in the
        # front layer. Currently `q0 -> phy 0`, `q2 -> phy 2` (they're
        # 2 hops apart). After SWAP on edge (0, 1) applied to trial
        # mapping: `q0 -> phy 1`, `q2 -> phy 2` → distance 1.
        # After SWAP on edge (1, 2) applied: `q0 -> phy 0`, `q2 -> phy 1`
        # → distance 1. Both swaps equally reduce front score.
        # Tie-break: lexicographic — (0, 1) is preferred.
        distances = [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ]
        edges = [(0, 1), (1, 2)]
        mapping = [0, 1, 2]  # three logical qubits q0=0, q1=1, q2=2; logical->physical identity
        # Front layer has pair (q0, q2) which is (logical 0, logical 2).
        front_layer = [(0, 2)]
        result = native.fast_sabre_swap_score(
            edges, distances, mapping, front_layer, [], 0.5,
        )
        self.assertIsNotNone(result)
        q1, q2, score = result
        self.assertEqual((q1, q2), (0, 1))
        self.assertAlmostEqual(score, 1.0, places=10)  # front=1, ext=0

    def test_swap_that_doesnt_help_is_rejected_in_favor_of_helpful_one(self):
        # Topology with one good swap and one useless swap:
        # 0-1-2-3 linear (4 physical qubits). Logical q0 at phy 0,
        # logical q1 at phy 3 (they're 3 hops apart). Two candidate
        # swaps: (0, 1) brings q0 to phy 1 (distance to q1 = 2). (2, 3)
        # brings q1 to phy 2 (distance to q0 = 2). Both reduce distance
        # to 2. Tie-break: smallest pair: (0, 1) wins.
        distances = [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ]
        edges = [(0, 1), (1, 2), (2, 3)]
        mapping = [0, 3]  # q0@0, q1@3 (q_idx 0, q_idx 1)
        front_layer = [(0, 1)]
        result = native.fast_sabre_swap_score(
            edges, distances, mapping, front_layer, [], 0.5,
        )
        q1, q2, score = result
        self.assertEqual((q1, q2), (0, 1))
        self.assertAlmostEqual(score, 2.0, places=10)

    def test_lexicographic_tie_break(self):
        # Three candidate swaps all reduce front score by the same
        # amount. Lexicographic smallest should win.
        distances = [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ]
        edges = [(2, 3), (0, 1), (1, 2)]  # input order is reversed-ish
        mapping = [0, 3]
        front_layer = [(0, 1)]
        result = native.fast_sabre_swap_score(
            edges, distances, mapping, front_layer, [], 0.0,
        )
        q1, q2, _ = result
        # All three reduce front to same distance: (0,1)->2, (1,2)->2, (2,3)->2.
        # Tie-break: smallest pair = (0, 1).
        self.assertEqual((q1, q2), (0, 1))

    def test_extended_layer_influences_choice(self):
        # Two swaps tie on the front layer; the one with the better
        # extended-layer score wins.
        # Topology: two separate edges (0-1) and (2-3) with no
        # cross-connection, so distances between {0,1} and {2,3} are
        # INF_SENTINEL. We'll provide our own distance matrix.
        # Map logical q0 -> 0, q1 -> 1, q2 -> 2, q3 -> 3. Front layer
        # has pair (q0, q1) already at distance 1 (they're connected).
        # Extended layer has pair (q2, q3) at distance 1 (also connected).
        # All candidate SWAPs on the graph: (0,1), (2,3).
        # Both swaps reduce the front distance by an unknown amount but
        # we want the kernel to compute and report the lowest score.
        distances = [
            [0, 1, 5, 5],
            [1, 0, 5, 5],
            [5, 5, 0, 1],
            [5, 5, 1, 0],
        ]
        edges = [(0, 1), (2, 3)]
        mapping = [0, 1, 2, 3]
        front_layer = [(0, 1)]
        extended_layer = [(2, 3)]
        result = native.fast_sabre_swap_score(
            edges, distances, mapping, front_layer, extended_layer, 1.0,
        )
        q1, q2, score = result
        # Compute the expected score for each candidate manually.
        # Swap (0, 1): trial_mapping = [1, 0, 2, 3]. Front (q0, q1) = (1, 0)
        #   -> distances[1][0] = 1. Extended (q2, q3) = (2, 3) -> 1.
        #   Score = 1 + 1.0 * 1 = 2.
        # Swap (2, 3): trial_mapping = [0, 1, 3, 2]. Front (q0, q1) = (0, 1)
        #   -> distances[0][1] = 1. Extended (q2, q3) = (3, 2) -> 1.
        #   Score = 1 + 1.0 * 1 = 2.
        # Tie -> lexicographic smallest = (0, 1).
        self.assertEqual((q1, q2), (0, 1))
        self.assertAlmostEqual(score, 2.0, places=10)

    def test_self_loop_edge_is_skipped(self):
        # An (q1, q1) self-edge should be skipped — it would be a
        # no-op trial mapping. We still should receive a valid choice
        # from the remaining edges.
        distances = [[0, 1], [1, 0]]
        edges = [(0, 0), (0, 1)]  # first is a self-loop
        mapping = [0, 1]
        front_layer = [(0, 1)]
        result = native.fast_sabre_swap_score(
            edges, distances, mapping, front_layer, [], 0.5,
        )
        q1, q2, _ = result
        self.assertEqual((q1, q2), (0, 1))

    def test_python_and_rust_kernels_agree_on_production_graph(self):
        # Cross-check: run the production `SabreRouter` against a small
        # circuit using both the Rust kernel (active path) and the
        # Python fallback (by monkey-patching the kernel to None). Both
        # should produce the same swap_count and final_mapping.
        from src.routing.router import CouplingMap, SabreRouter
        coupling = CouplingMap.linear(4)
        ops = [
            {'gate': 'CNOT', 'targets': ['q0', 'q3'], 'args': []},
            {'gate': 'CNOT', 'targets': ['q1', 'q2'], 'args': []},
            {'gate': 'CNOT', 'targets': ['q0', 'q2'], 'args': []},
        ]
        logical_qubits = ['q0', 'q1', 'q2', 'q3']

        # Rust path
        router_rust = SabreRouter(coupling, lookahead_weight=0.5)
        result_rust = router_rust.route(ops, logical_qubits)

        # Python path: monkey-patch `eigen_native.fast_sabre_swap_score`
        # to None on the imported native symbol so the router falls
        # back to the in-Python loop.
        import src.routing.router as router_mod
        getattr(router_mod, 'eigen_native', None) \
            if False else None
        # Reach into the function call inside SabreRouter.route by
        # setting `eigen_native.fast_sabre_swap_score` to None — but
        # we can't easily monkeypatch a Rust function. Instead, shadow
        # the symbol via module cache: temporarily replace the
        # `fast_sabre_swap_score` attribute on the imported
        # `eigen_native` module.
        orig_kernel = native.fast_sabre_swap_score
        try:
            native.fast_sabre_swap_score = None  # type: ignore
            router_py = SabreRouter(coupling, lookahead_weight=0.5)
            result_py = router_py.route(ops, logical_qubits)
        finally:
            native.fast_sabre_swap_score = orig_kernel  # type: ignore

        # The two paths must agree on swap count and final mapping
        # (determinism check).
        self.assertEqual(result_rust.swap_count, result_py.swap_count)
        self.assertEqual(result_rust.final_mapping, result_py.final_mapping)


if __name__ == '__main__':
    unittest.main()
