"""Tests for Hardware Connectivity Mapper (routing module)."""
import unittest
from src.routing.router import (
    CouplingMap, RoutedCircuit, BasicSwapRouter, GreedyRouter, SabreRouter, route_eqir_graph
)
from src.ir.ir_graph import EQIRGraph


class TestCouplingMap(unittest.TestCase):

    def test_linear_creation(self):
        for n in range(2, 8):
            with self.subTest(n=n):
                cm = CouplingMap.linear(n)
                self.assertEqual(cm.num_qubits, n)

    def test_linear_connectivity(self):
        cm = CouplingMap.linear(5)
        for i in range(4):
            self.assertTrue(cm.are_connected(i, i + 1))
        self.assertFalse(cm.are_connected(0, 2))
        self.assertFalse(cm.are_connected(0, 3))
        self.assertFalse(cm.are_connected(0, 4))

    def test_grid_creation(self):
        grids = [(2, 2), (2, 3), (3, 3), (4, 4)]
        for rows, cols in grids:
            with self.subTest(rows=rows, cols=cols):
                cm = CouplingMap.grid(rows, cols)
                self.assertEqual(cm.num_qubits, rows * cols)

    def test_grid_connectivity(self):
        cm = CouplingMap.grid(3, 3)
        # Center (4) connected to 1, 3, 5, 7
        self.assertTrue(cm.are_connected(4, 1))
        self.assertTrue(cm.are_connected(4, 3))
        self.assertTrue(cm.are_connected(4, 5))
        self.assertTrue(cm.are_connected(4, 7))
        # Corners not connected diagonally
        self.assertFalse(cm.are_connected(0, 4))
        self.assertFalse(cm.are_connected(2, 4))

    def test_shortest_path_linear(self):
        cm = CouplingMap.linear(5)
        path = cm.shortest_path(0, 4)
        self.assertEqual(path, [0, 1, 2, 3, 4])

    def test_shortest_path_same_node(self):
        cm = CouplingMap.linear(5)
        self.assertEqual(cm.shortest_path(2, 2), [2])

    def test_shortest_path_adjacent(self):
        cm = CouplingMap.linear(5)
        self.assertEqual(cm.shortest_path(1, 2), [1, 2])

    def test_distance_linear(self):
        cm = CouplingMap.linear(5)
        for i in range(5):
            for j in range(5):
                with self.subTest(i=i, j=j):
                    self.assertEqual(cm.distance(i, j), abs(i - j))

    def test_distance_grid(self):
        cm = CouplingMap.grid(2, 2)
        # 0-1
        # 2-3
        self.assertEqual(cm.distance(0, 3), 2)
        self.assertEqual(cm.distance(0, 1), 1)
        self.assertEqual(cm.distance(0, 2), 1)
        self.assertEqual(cm.distance(1, 2), 2)

    def test_neighbors_linear(self):
        cm = CouplingMap.linear(5)
        self.assertEqual(cm.neighbors(0), {1})
        self.assertEqual(cm.neighbors(4), {3})
        self.assertEqual(cm.neighbors(2), {1, 3})

    def test_neighbors_grid(self):
        cm = CouplingMap.grid(3, 3)
        # Corner 0: neighbors are 1, 3
        self.assertEqual(cm.neighbors(0), {1, 3})
        # Center 4: neighbors are 1, 3, 5, 7
        self.assertEqual(cm.neighbors(4), {1, 3, 5, 7})

    def test_heavy_hex(self):
        cm = CouplingMap.heavy_hex(3)
        self.assertEqual(cm.num_qubits, 9)

    def test_bidirectional_edges(self):
        cm = CouplingMap([(0, 1)])
        self.assertTrue(cm.are_connected(0, 1))
        self.assertTrue(cm.are_connected(1, 0))

    def test_custom_topology(self):
        # Ring topology: 0-1-2-3-0
        cm = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 0)])
        self.assertEqual(cm.num_qubits, 4)
        self.assertTrue(cm.are_connected(0, 3))
        self.assertTrue(cm.are_connected(3, 0))
        self.assertEqual(cm.distance(0, 2), 2)


class TestBasicSwapRouter(unittest.TestCase):

    def test_single_qubit_gates_no_swaps(self):
        cm = CouplingMap.linear(3)
        router = BasicSwapRouter(cm)
        ops = [
            {'gate': 'H', 'targets': ['q0']},
            {'gate': 'X', 'targets': ['q1']},
            {'gate': 'Z', 'targets': ['q2']},
        ]
        result = router.route(ops, ['q0', 'q1', 'q2'])
        self.assertEqual(result.swap_count, 0)
        self.assertEqual(len(result.operations), 3)

    def test_adjacent_cnot_no_swaps(self):
        cm = CouplingMap.linear(3)
        router = BasicSwapRouter(cm)
        ops = [{'gate': 'CNOT', 'targets': ['q0', 'q1']}]
        result = router.route(ops, ['q0', 'q1', 'q2'])
        self.assertEqual(result.swap_count, 0)
        self.assertEqual(len(result.operations), 1)

    def test_non_adjacent_cnot_inserts_swaps(self):
        cm = CouplingMap.linear(3)
        router = BasicSwapRouter(cm)
        ops = [{'gate': 'CNOT', 'targets': ['q0', 'q2']}]
        result = router.route(ops, ['q0', 'q1', 'q2'])
        self.assertGreater(result.swap_count, 0)

    def test_non_adjacent_long_distance(self):
        cm = CouplingMap.linear(5)
        router = BasicSwapRouter(cm)
        ops = [{'gate': 'CNOT', 'targets': ['q0', 'q4']}]
        result = router.route(ops, ['q0', 'q1', 'q2', 'q3', 'q4'])
        self.assertEqual(result.swap_count, 3)  # Need 3 swaps: 0-1, 1-2, 2-3

    def test_multiple_gates_routing(self):
        cm = CouplingMap.linear(4)
        router = BasicSwapRouter(cm)
        ops = [
            {'gate': 'H', 'targets': ['q0']},
            {'gate': 'CNOT', 'targets': ['q0', 'q1']},
            {'gate': 'CNOT', 'targets': ['q0', 'q3']},
        ]
        result = router.route(ops, ['q0', 'q1', 'q2', 'q3'])
        self.assertGreater(len(result.operations), 3)  # At least some swaps needed

    def test_initial_and_final_mapping(self):
        cm = CouplingMap.linear(3)
        router = BasicSwapRouter(cm)
        ops = [{'gate': 'H', 'targets': ['q0']}]
        result = router.route(ops, ['q0', 'q1', 'q2'])
        self.assertEqual(result.initial_mapping, {'q0': 0, 'q1': 1, 'q2': 2})
        self.assertEqual(result.final_mapping, {'q0': 0, 'q1': 1, 'q2': 2})

    def test_too_many_qubits_raises(self):
        cm = CouplingMap.linear(2)
        router = BasicSwapRouter(cm)
        ops = [{'gate': 'H', 'targets': ['q0']}]
        with self.assertRaises(ValueError):
            router.route(ops, ['q0', 'q1', 'q2'])

    def test_empty_circuit(self):
        cm = CouplingMap.linear(3)
        router = BasicSwapRouter(cm)
        result = router.route([], ['q0', 'q1'])
        self.assertEqual(result.swap_count, 0)
        self.assertEqual(len(result.operations), 0)

    def test_routed_circuit_summary(self):
        cm = CouplingMap.linear(3)
        router = BasicSwapRouter(cm)
        ops = [{'gate': 'CNOT', 'targets': ['q0', 'q2']}]
        result = router.route(ops, ['q0', 'q1', 'q2'])
        summary = result.summary()
        self.assertIn('total_gates', summary)
        self.assertIn('swap_count', summary)
        self.assertIn('initial_mapping', summary)
        self.assertIn('final_mapping', summary)

    def test_routed_circuit_summary_exact_values(self):
        """Verify exact summary JSON for CNOT(q0,q2) on linear-3."""
        cm = CouplingMap.linear(3)
        router = BasicSwapRouter(cm)
        ops = [{'gate': 'CNOT', 'targets': ['q0', 'q2']}]
        result = router.route(ops, ['q0', 'q1', 'q2'])
        # Trace: path(0,2)=[0,1,2] -> SWAP(0,1) -> CNOT(1,2)
        expected_summary = {
            'total_gates': 2,
            'swap_count': 1,
            'initial_mapping': {'q0': 0, 'q1': 1, 'q2': 2},
            'final_mapping': {'q0': 1, 'q1': 0, 'q2': 2},
        }
        self.assertEqual(result.summary(), expected_summary)
        # Verify exact operation sequence
        self.assertEqual(result.operations, [
            ('SWAP', [0, 1], []),
            ('CNOT', [1, 2], []),
        ])

    def test_long_distance_summary_exact_values(self):
        """Verify exact summary JSON for CNOT(q0,q4) on linear-5."""
        cm = CouplingMap.linear(5)
        router = BasicSwapRouter(cm)
        ops = [{'gate': 'CNOT', 'targets': ['q0', 'q4']}]
        result = router.route(ops, ['q0', 'q1', 'q2', 'q3', 'q4'])
        # Trace: path(0,4)=[0,1,2,3,4] -> SWAP(0,1),SWAP(1,2),SWAP(2,3) -> CNOT(3,4)
        expected_summary = {
            'total_gates': 4,
            'swap_count': 3,
            'initial_mapping': {'q0': 0, 'q1': 1, 'q2': 2, 'q3': 3, 'q4': 4},
            'final_mapping': {'q0': 3, 'q1': 0, 'q2': 1, 'q3': 2, 'q4': 4},
        }
        self.assertEqual(result.summary(), expected_summary)
        self.assertEqual(result.operations, [
            ('SWAP', [0, 1], []),
            ('SWAP', [1, 2], []),
            ('SWAP', [2, 3], []),
            ('CNOT', [3, 4], []),
        ])

    def test_multi_gate_cascade_exact_values(self):
        """Verify exact output for H + adjacent CNOT + non-adjacent CNOT."""
        cm = CouplingMap.linear(4)
        router = BasicSwapRouter(cm)
        ops = [
            {'gate': 'H', 'targets': ['q0']},
            {'gate': 'CNOT', 'targets': ['q0', 'q1']},
            {'gate': 'CNOT', 'targets': ['q0', 'q3']},
        ]
        result = router.route(ops, ['q0', 'q1', 'q2', 'q3'])
        # H(q0) -> phys 0, no swap
        # CNOT(q0,q1) -> phys 0,1 adjacent, no swap
        # CNOT(q0,q3) -> phys 0,3 path=[0,1,2,3] -> SWAP(0,1),SWAP(1,2) -> CNOT(2,3)
        expected_summary = {
            'total_gates': 5,
            'swap_count': 2,
            'initial_mapping': {'q0': 0, 'q1': 1, 'q2': 2, 'q3': 3},
            'final_mapping': {'q0': 2, 'q1': 0, 'q2': 1, 'q3': 3},
        }
        self.assertEqual(result.summary(), expected_summary)
        self.assertEqual(result.operations, [
            ('H', [0], []),
            ('CNOT', [0, 1], []),
            ('SWAP', [0, 1], []),
            ('SWAP', [1, 2], []),
            ('CNOT', [2, 3], []),
        ])


class TestGreedyRouter(unittest.TestCase):

    def test_adjacent_no_swaps(self):
        cm = CouplingMap.linear(3)
        router = GreedyRouter(cm)
        ops = [{'gate': 'CNOT', 'targets': ['q0', 'q1']}]
        result = router.route(ops, ['q0', 'q1', 'q2'])
        self.assertEqual(result.swap_count, 0)

    def test_non_adjacent_inserts_swaps(self):
        cm = CouplingMap.linear(4)
        router = GreedyRouter(cm)
        ops = [{'gate': 'CNOT', 'targets': ['q0', 'q3']}]
        result = router.route(ops, ['q0', 'q1', 'q2', 'q3'])
        self.assertGreater(result.swap_count, 0)

    def test_single_qubit_only(self):
        cm = CouplingMap.linear(3)
        router = GreedyRouter(cm)
        ops = [
            {'gate': 'H', 'targets': ['q0']},
            {'gate': 'X', 'targets': ['q1']},
        ]
        result = router.route(ops, ['q0', 'q1', 'q2'])
        self.assertEqual(result.swap_count, 0)

    def test_grid_topology(self):
        cm = CouplingMap.grid(2, 3)
        router = GreedyRouter(cm)
        ops = [
            {'gate': 'CNOT', 'targets': ['q0', 'q5']},
        ]
        result = router.route(ops, ['q0', 'q1', 'q2', 'q3', 'q4', 'q5'])
        self.assertGreater(result.swap_count, 0)

    def test_greedy_vs_basic_produces_valid_result(self):
        cm = CouplingMap.linear(5)
        ops = [
            {'gate': 'CNOT', 'targets': ['q0', 'q4']},
            {'gate': 'CNOT', 'targets': ['q1', 'q3']},
        ]
        qubits = ['q0', 'q1', 'q2', 'q3', 'q4']

        basic = BasicSwapRouter(cm).route(ops, qubits)
        greedy = GreedyRouter(cm).route(ops, qubits)

        # Both should produce valid results with SWAP gates
        self.assertGreater(basic.swap_count, 0)
        self.assertGreater(greedy.swap_count, 0)


class TestRouteEQIRGraph(unittest.TestCase):

    def test_route_simple_circuit(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('ALLOC', targets=['q1'])
        graph.add_operation('GATE', gate_name='H', targets=['q0'])
        graph.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q1'])

        cm = CouplingMap.linear(2)
        result = route_eqir_graph(graph, cm, 'basic')
        self.assertEqual(result.swap_count, 0)

    def test_route_requires_swaps(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('ALLOC', targets=['q1'])
        graph.add_operation('ALLOC', targets=['q2'])
        graph.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q2'])

        cm = CouplingMap.linear(3)
        result = route_eqir_graph(graph, cm, 'basic')
        self.assertGreater(result.swap_count, 0)

    def test_route_empty_graph(self):
        graph = EQIRGraph()
        cm = CouplingMap.linear(3)
        result = route_eqir_graph(graph, cm)
        self.assertEqual(result.swap_count, 0)

    def test_route_with_greedy(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('ALLOC', targets=['q1'])
        graph.add_operation('ALLOC', targets=['q2'])
        graph.add_operation('GATE', gate_name='H', targets=['q0'])
        graph.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q2'])

        cm = CouplingMap.linear(3)
        result = route_eqir_graph(graph, cm, 'greedy')
        self.assertIsNotNone(result)

    def test_route_with_sabre(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('ALLOC', targets=['q1'])
        graph.add_operation('ALLOC', targets=['q2'])
        graph.add_operation('GATE', gate_name='H', targets=['q0'])
        graph.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q2'])

        cm = CouplingMap.linear(3)
        result = route_eqir_graph(graph, cm, 'sabre')
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.operations), 2)

    def test_route_too_few_physical_qubits(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('ALLOC', targets=['q1'])
        graph.add_operation('ALLOC', targets=['q2'])

        cm = CouplingMap.linear(2)
        with self.assertRaises(ValueError):
            route_eqir_graph(graph, cm)


if __name__ == "__main__":
    unittest.main()
