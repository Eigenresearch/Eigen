"""
Audit §5 — shared gate registry + hardware-backend export regression tests.

The audit fixed two concrete bugs in the four hardware-backend exporters
(`ibm_backend.py`, `ionq_backend.py`, `azure_backend.py`, `braket_backend.py`):

  * SWAP was missing entirely from the Azure QIR exporter (silent drop).
  * The IBM exporter only emitted OpenQASM 2.0, not the 3.0 format that
    Qiskit migrated to as its primary interchange format.

Both were downstream consequences of a structural bug: the four exporters
carried independent `if/elif` chains over gate names. Adding a new gate
required touching four files, and gates were forgotten in some exporters
(SWAP in Azure) while the format choice in another was kept at 2.0 (IBM).

After the fix, the four exporters all dispatch through the shared
`gate_registry.GateSpec` table. These tests pin down:

  * Every gate registered in `gate_registry` is emitted by every backend
    (no silent drops).
  * SWAP specifically is emitted to Azure QIR as
    `__quantum__qis__swap__body`.
  * IBM emits `OPENQASM 3.0;` and `qubit[]` declarations.
  * IBM's optional 2.0 fallback still works for callers that need it.
  * The "no silent drops" rule: unsupported gates are surfaced by name,
    not lost.
"""

import json
import unittest

from src.ir.ir_graph import EQIRGraph
from src.backend.gate_registry import (
    get_gate_spec, all_registered_gates, GATE_QUBIT_COUNT,
)
from src.backend.backends.ibm_backend import IBMBackend
from src.backend.backends.ionq_backend import IonQBackend
from src.backend.backends.azure_backend import AzureBackend
from src.backend.backends.braket_backend import BraketBackend


def _build_graph_with_all_gates():
    """Build an EQIR graph that exercises every gate in the registry.

    Each registered gate is applied once on the appropriate number of
    freshly-allocated qubits, after the audit's `no silent drops` rule.
    """
    g = EQIRGraph()
    # Allocate enough qubits for the largest gate (CSWAP = 3 qubits). The
    # remaining single-/two-qubit gates reuse qubits as their
    # `qubit_last_writer` chain naturally requires.
    g.add_operation('ALLOC', targets=['q0'])
    g.add_operation('ALLOC', targets=['q1'])
    g.add_operation('ALLOC', targets=['q2'])
    g.add_operation('ALLOC', targets=['q3'])

    # We need to use distinct gate names so each gate is observed exactly
    # once by the all-gate check.
    plan = [
        ('H',     ['q0'], None),
        ('X',     ['q0'], None),
        ('Y',     ['q0'], None),
        ('Z',     ['q0'], None),
        ('S',     ['q0'], None),
        ('T',     ['q0'], None),
        ('RX',    ['q0'], [0.1]),
        ('RY',    ['q0'], [0.2]),
        ('RZ',    ['q0'], [0.3]),
        ('CNOT',  ['q0', 'q1'], None),
        ('CZ',    ['q0', 'q1'], None),
        ('SWAP',  ['q0', 'q1'], None),
        ('CCX',   ['q0', 'q1', 'q2'], None),
        ('CSWAP', ['q0', 'q1', 'q2'], None),
        ('CP',    ['q0', 'q1'], [0.4]),
        ('CRX',   ['q0', 'q1'], [0.5]),
        ('CRY',   ['q0', 'q1'], [0.6]),
        ('CRZ',   ['q0', 'q1'], [0.7]),
    ]
    for gate_name, targets, args in plan:
        kwargs = {'gate_name': gate_name, 'targets': targets}
        if args is not None:
            kwargs['args'] = args
        g.add_operation('GATE', **kwargs)

    g.add_operation('MEASURE', targets=['q0'], cbit_name='c0')
    return g


class TestGateRegistryCore(unittest.TestCase):
    def test_get_gate_spec_returns_a_spec_for_each_registered_gate(self):
        specs = {name: get_gate_spec(name) for name in all_registered_gates()}
        for name, spec in specs.items():
            self.assertIsNotNone(spec, msg=f"missing spec for gate {name}")
            # Per-exporter encoding tables must all be populated; the audit
            # explicitly forbids any of the four exporters silently
            # dropping a registered gate.
            self.assertTrue(spec.qasm_name, msg=f"{name} lacks qasm_name")
            self.assertTrue(spec.qir_func,
                             msg=f"{name} lacks qir_func (was previously "
                                 f"the SWAP-on-Azure silent-drop bug)")
            self.assertTrue(spec.braket_method, msg=f"{name} lacks braket_method")
            self.assertTrue(spec.ionq_gate, msg=f"{name} lacks ionq_gate")

    def test_get_gate_spec_is_case_insensitive(self):
        # Internal names are uppercase, but exporters sometimes pass through
        # whatever the user wrote. Case-insensitive lookup keeps the
        # behavior stable.
        self.assertEqual(get_gate_spec('H') and get_gate_spec('h').qasm_name, 'h')
        self.assertIsNone(get_gate_spec(''))
        self.assertIsNone(get_gate_spec(None))
        self.assertIsNone(get_gate_spec('NOT_A_GATE'))

    def test_qubit_count_in_spec_matches_GATE_QUBIT_COUNT_table(self):
        for gate, spec in [(g, get_gate_spec(g)) for g in all_registered_gates()]:
            self.assertEqual(spec.qubit_count, GATE_QUBIT_COUNT[gate],
                             msg=f"{gate} has inconsistent qubit_count in spec vs legacy table")

    def test_get_gate_spec_qir_callee_format(self):
        # Audit specifically called out that SWAP was missing on Azure. The
        # full callee is `__quantum__qis__<func>__body` - pin the
        # naming convention so a typo here surfaces immediately.
        self.assertEqual(get_gate_spec('SWAP').qir_callee(),
                         '__quantum__qis__swap__body')
        self.assertEqual(get_gate_spec('H').qir_callee(),
                         '__quantum__qis__h__body')


class TestIBMBackend(unittest.TestCase):
    def setUp(self):
        self.graph = _build_graph_with_all_gates()
        self.output = IBMBackend().export(self.graph)

    def test_emits_openqasm_3_header(self):
        self.assertIn("OPENQASM 3.0;", self.output)
        self.assertIn('include "stdgates.inc";', self.output)

    def test_emits_3_style_register_decls(self):
        self.assertIn("qubit[4] q;", self.output)
        self.assertIn("bit[1] c;", self.output)

    def test_emits_3_style_measurement(self):
        # OpenQASM 3.0 in-line measurement is `c[0] = measure q[0];`
        self.assertIn("c[0] = measure q[0];", self.output)
        # 2.0 syntax should NOT be present in 3.0 output
        self.assertNotIn("measure q[0] -> c[0];", self.output)

    def test_swap_is_emitted_not_dropped(self):
        # Audit: previously only OpenQASM 2.0 was emitted and the audit
        # called this out specifically.
        self.assertIn("swap q[0], q[1];", self.output)

    def test_all_registered_gates_appear_without_an_unsupported_marker(self):
        # The audit's "no silent drops" rule: every registered gate must be
        # emitted somewhere in the output without an `Unsupported` comment.
        # QASM3 can spell a gate either as `qasm_name` (legacy qelib1
        # spelling, e.g. `cp`) or via `qasm3_rendering` (`ctrl @ phase`),
        # so accept either as evidence the gate was emitted.
        for gate_name in all_registered_gates():
            spec = get_gate_spec(gate_name)
            renderings = {spec.qasm_name, spec.qasm3_rendering}
            renderings.discard(None)
            self.assertTrue(
                any(r in self.output for r in renderings),
                msg=f"gate {gate_name} (qasm_name={spec.qasm_name!r}, "
                    f"qasm3_render={spec.qasm3_rendering!r}) not in IBM "
                    f"3.0 output:\n{self.output}"
            )
        self.assertNotIn("Unsupported", self.output)

    def test_controlled_rotations_use_qasm3_ctrl_at_form(self):
        # The 3.0 preferred spelling for `crx` is `ctrl @ rx(angle)`.
        self.assertIn("ctrl @ rx(0.5)", self.output)
        self.assertIn("ctrl @ phase(0.4)", self.output)

    def test_qasm2_fallback_emits_openqasm_2_header(self):
        legacy = IBMBackend().export_qasm2(self.graph)
        self.assertIn("OPENQASM 2.0;", legacy)
        self.assertIn('include "qelib1.inc";', legacy)
        self.assertIn("qreg q[4];", legacy)
        self.assertIn("measure q[0] -> c[0];", legacy)
        # 2.0 fallback uses the bare `crx`/`cry`/`crz`/`cp` spelling,
        # not the 3.0 `ctrl @` form.
        self.assertIn("crx(0.5)", legacy)
        self.assertNotIn("ctrl @", legacy)


class TestAzureBackend(unittest.TestCase):
    def setUp(self):
        self.graph = _build_graph_with_all_gates()
        self.output = AzureBackend().export(self.graph)

    def test_swap_emitted_via_qir_callee(self):
        # Audit's headline bug for Azure: SWAP was missing entirely from
        # the if/elif chain. Now it must appear as a QIR call instruction
        # with the correct callee, plus a matching `declare` line.
        # The call instruction uses SSA-style `%q0`/`%q1` operands (no type
        # annotation on call sites); the declare line carries the type
        # signature `%Qubit*, %Qubit*`.
        self.assertIn(
            "call void @__quantum__qis__swap__body(%q0, %q1)",
            self.output,
        )
        self.assertIn(
            "declare void @__quantum__qis__swap__body(%Qubit*, %Qubit*)",
            self.output,
        )

    def test_all_registered_gates_have_qir_declares(self):
        # The declarations section must contain exactly the QIR callers
        # that were actually used, but at a minimum must cover every
        # registered gate in the audit's "no silent drops" rule.
        for gate_name in all_registered_gates():
            spec = get_gate_spec(gate_name)
            self.assertIn(spec.qir_callee(), self.output,
                          msg=f"QIR callee for {gate_name} not emitted")

    def test_no_unsupported_marker_for_registered_gates(self):
        self.assertNotIn("Unsupported gate", self.output)

    def test_module_header_present(self):
        self.assertIn("; ModuleID = 'EigenQIRModule'", self.output)
        self.assertIn("define void @main() #0 {", self.output)


class TestBraketBackend(unittest.TestCase):
    def setUp(self):
        self.graph = _build_graph_with_all_gates()
        self.output = BraketBackend().export(self.graph)

    def test_emits_circuit_class_instantiation(self):
        self.assertIn("from braket.circuits import Circuit", self.output)
        self.assertIn("device_circuit = Circuit()", self.output)

    def test_swap_emitted(self):
        self.assertIn("device_circuit.swap(0, 1)", self.output)

    def test_all_registered_gates_appear(self):
        for gate_name in all_registered_gates():
            spec = get_gate_spec(gate_name)
            self.assertIn(spec.braket_method, self.output,
                          msg=f"Braket method {spec.braket_method} for {gate_name} missing")

    def test_rotation_gates_emit_args(self):
        # RX(0.1) on qubit 0 becomes .rx(0, 0.1)
        self.assertIn("device_circuit.rx(0, 0.1)", self.output)


class TestIonQBackend(unittest.TestCase):
    def setUp(self):
        self.graph = _build_graph_with_all_gates()
        raw = IonQBackend().export(self.graph)
        self.output = raw
        self.doc = json.loads(raw)

    def test_emits_valid_json(self):
        self.assertIsInstance(self.doc, dict)
        self.assertEqual(self.doc["qubits"], 4)
        self.assertIsInstance(self.doc["circuit"], list)

    def test_swap_is_present(self):
        # Audit's "no silent drops" rule
        gates_emitted = {entry["gate"] for entry in self.doc["circuit"]}
        self.assertIn("swap", gates_emitted)

    def test_all_registered_gates_appear_in_circuit(self):
        gates_emitted = {entry["gate"] for entry in self.doc["circuit"]}
        for gate_name in all_registered_gates():
            spec = get_gate_spec(gate_name)
            self.assertIn(spec.ionq_gate, gates_emitted,
                          msg=f"ionq_gate {spec.ionq_gate} for {gate_name} missing")

    def test_rotation_gates_carry_phase_field(self):
        rx_entries = [e for e in self.doc["circuit"] if e["gate"] == "rx"]
        self.assertTrue(rx_entries)
        self.assertIn("phase", rx_entries[0])


if __name__ == '__main__':
    unittest.main()
