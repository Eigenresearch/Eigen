"""
P2 §3.2 — Parameterized circuits (surface-level API) tests.

See ``src/parametrized_circuit.py`` for docs. The tests cover:
  * Parameter uniqueness / hashing / equality.
  * `ParametrizedCircuit.parameters()` discovers unique placeholders.
  * `bind()` resolves by Parameter instance or by string key.
  * `bind()` raises ``UnresolvedParameterError`` when the caller
    omits a parameter.
  * `bind({})` on a parameter-free circuit is a no-op.
  * `freeze(theta=...)` keyword sugar matches `bind({p: ...})`.
  * `clone()` produces a structurally equal but distinct circuit.
  * `run_resolved_circuit(simulator, resolved)` actually drives the
    existing QuantumSimulator and produces a Bell state when feeding
    `(H, [q0]), (CNOT, [q0, q1])`.
"""
import math
import unittest

from src.parametrized_circuit import (
    Parameter,
    ParametrizedCircuit,
    ResolvedCircuit,
    UnresolvedParameterError,
    run_resolved_circuit,
)


class TestParameter(unittest.TestCase):

    def test_unique_by_name(self):
        a = Parameter("theta")
        b = Parameter("theta")
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(a, b)

    def test_repr_shows_name(self):
        self.assertEqual(repr(Parameter("alpha")), "Parameter('alpha')")
        self.assertEqual(str(Parameter("alpha")), "$alpha")


class TestParametrizedCircuit(unittest.TestCase):

    def test_empty_circuit_has_no_parameters(self):
        c = ParametrizedCircuit(instructions=[])
        self.assertEqual(c.parameters(), [])

    def test_parameter_discovery_unique(self):
        theta = Parameter("theta")
        phi = Parameter("phi")
        c = ParametrizedCircuit(instructions=[
            ("RX", ["q0"], theta),
            ("RY", ["q0"], phi),
            ("RZ", ["q0"], theta),  # same param reused — should only count once
        ])
        params = c.parameters()
        # Insertion order: theta first, phi second.
        self.assertEqual(params, [theta, phi])

    def test_bind_by_parameter_instance(self):
        theta = Parameter("theta")
        c = ParametrizedCircuit(instructions=[
            ("RX", ["q0"], theta),
        ])
        resolved = c.bind({theta: 0.5})
        self.assertIsInstance(resolved, ResolvedCircuit)
        self.assertEqual(resolved.instructions,
                         [("RX", ["q0"], 0.5)])

    def test_bind_by_string_key(self):
        theta = Parameter("theta")
        c = ParametrizedCircuit(instructions=[
            ("RX", ["q0"], theta),
        ])
        resolved = c.bind({"theta": 0.7})
        self.assertEqual(resolved.instructions, [("RX", ["q0"], 0.7)])
        # Binding map preserved on the ResolvedCircuit.
        self.assertEqual(resolved.binding[theta], 0.7)

    def test_bind_missing_raises(self):
        theta = Parameter("theta")
        c = ParametrizedCircuit(instructions=[("RX", ["q0"], theta)])
        with self.assertRaises(UnresolvedParameterError):
            c.bind({})

    def test_bind_extra_unknown_param_raises(self):
        # Extra unknown strings must surface — silently accepting
        # them would let typos in the cli leak through.
        c = ParametrizedCircuit(instructions=[("H", ["q0"])])
        with self.assertRaises(UnresolvedParameterError):
            c.bind({"unknown_param": 1.0})

    def test_bind_mixed_concrete_and_parameter_values(self):
        # Instructions can have concrete floats alongside Parameter
        # references; bind only substitutes the latter.
        theta = Parameter("theta")
        c = ParametrizedCircuit(instructions=[
            ("RX", ["q0"], theta),
            ("RZ", ["q1"], 0.3),  # concrete float — not a parameter
        ])
        resolved = c.bind({theta: 0.5})
        self.assertEqual(resolved.instructions,
                         [("RX", ["q0"], 0.5),
                          ("RZ", ["q1"], 0.3)])

    def test_bind_no_parameters_no_op(self):
        # Circuit with only concrete values can be bound with empty
        # dict — no parameters to resolve.
        c = ParametrizedCircuit(instructions=[
            ("H", ["q0"]),
            ("CNOT", ["q0", "q1"]),
        ])
        resolved = c.bind({})
        self.assertEqual(resolved.instructions, c.instructions)

    def test_freeze_keyword_sugar(self):
        theta = Parameter("theta")
        c = ParametrizedCircuit(instructions=[("RX", ["q0"], theta)])
        resolved = c.freeze(theta=1.0)
        self.assertEqual(resolved.instructions, [("RX", ["q0"], 1.0)])

    def test_clone_produces_distinct_list(self):
        theta = Parameter("theta")
        c = ParametrizedCircuit(instructions=[("RX", ["q0"], theta)])
        c2 = c.clone()
        self.assertIsNot(c.instructions, c2.instructions)
        self.assertEqual(c.instructions, c2.instructions)

    def test_len_and_iter(self):
        c = ParametrizedCircuit(instructions=[
            ("H", ["q0"]),
            ("CNOT", ["q0", "q1"]),
        ])
        self.assertEqual(len(c), 2)
        self.assertEqual([i for i in c],
                         [("H", ["q0"]), ("CNOT", ["q0", "q1"])])


class TestRunResolvedCircuit(unittest.TestCase):

    def test_bell_state_via_resolved_circuit(self):
        # Build the canonical Bell-state circuit, bind (no params), and
        # replay it through QuantumSimulator — verify H|0> + CNOT gives
        # the Bell amplitudes.
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(sim_type='dense')
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        circuit = ParametrizedCircuit(instructions=[
            ("H", ["q0"]),
            ("CNOT", ["q0", "q1"]),
        ]).bind({})  # no parameters
        run_resolved_circuit(sim, circuit)
        inv_sqrt2 = 1.0 / math.sqrt(2)
        state = sim.get_state_vector()
        self.assertAlmostEqual(state[0].real, inv_sqrt2, places=8)
        self.assertAlmostEqual(state[3].real, inv_sqrt2, places=8)
        self.assertAlmostEqual(abs(state[1]), 0.0, places=8)
        self.assertAlmostEqual(abs(state[2]), 0.0, places=8)

    def test_parametrized_rx_resolves_to_correct_state(self):
        # RX(theta)|0> = cos(theta/2)|0> + i*j sin(theta/2)|1>. Bind
        # theta=π so we expect |1> up to a phase.
        theta = Parameter("theta")
        circuit = ParametrizedCircuit(instructions=[
            ("RX", ["q0"], theta),
        ]).bind({theta: math.pi})
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(sim_type='dense')
        sim.allocate_qubit("q0")
        run_resolved_circuit(sim, circuit)
        state = sim.get_state_vector()
        # cos(pi/2) ≈ 0, sin(pi/2) = 1, so state = i*|1>.
        self.assertAlmostEqual(abs(state[0]), 0.0, places=8)
        self.assertAlmostEqual(abs(state[1]), 1.0, places=8)

    def test_unknown_gate_raises(self):
        sim = None  # we never reach the simulator because of the unknown gate
        circuit = ResolvedCircuit.of([
            ("NOPE", ["q0"]),
        ])
        with self.assertRaises(ValueError):
            # Pass a fake simulator — we expect the dispatch to raise
            # before touching it.
            run_resolved_circuit(sim, circuit)


if __name__ == "__main__":
    unittest.main()
