"""Tests for Eigen 2.3 Helios new features:
- GPU Engine routing
- Tensor Compiler
- Parallel blocks (lexer/parser/bytecode/VM)
- Verify command
- MLIR dialect
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.frontend.lexer import Lexer, TokenType
from src.frontend.parser import Parser
from src.frontend.ast import ParallelBlockNode, TaskStatementNode
from src.backend.bytecode import Opcode


# ========== GPU Engine Tests ==========

class TestGPUEngine:
    def test_gpu_engine_creation(self):
        """GPUEngine should be constructable."""
        from src.backend.gpu.gpu_engine import GPUEngine
        engine = GPUEngine(platform_name='auto')
        assert engine.platform in ('none', 'cuda', 'rocm', 'metal')
    
    def test_gpu_engine_init_state(self):
        """GPUEngine should initialize |0> state."""
        from src.backend.gpu.gpu_engine import GPUEngine
        engine = GPUEngine(platform_name='auto')
        engine.initialize_state(1)
        state = engine.get_state()
        assert len(state) == 2
        assert abs(state[0] - 1.0) < 1e-10
        assert abs(state[1]) < 1e-10
    
    def test_gpu_engine_apply_hadamard(self):
        """GPUEngine should apply a Hadamard gate correctly."""
        from src.backend.gpu.gpu_engine import GPUEngine
        engine = GPUEngine(platform_name='auto')
        engine.initialize_state(1)
        inv = 1.0 / math.sqrt(2.0)
        H = [[inv, inv], [inv, -inv]]
        engine.apply_gate([0], H)
        state = engine.get_state()
        assert abs(abs(state[0]) - inv) < 1e-10
        assert abs(abs(state[1]) - inv) < 1e-10

    def test_gpu_engine_set_state(self):
        """GPUEngine set_state should overwrite the state vector."""
        from src.backend.gpu.gpu_engine import GPUEngine
        engine = GPUEngine(platform_name='auto')
        engine.initialize_state(1)
        engine.set_state([0.0+0j, 1.0+0j])
        state = engine.get_state()
        assert abs(state[0]) < 1e-10
        assert abs(state[1] - 1.0) < 1e-10

    def test_gpu_engine_two_qubit_state(self):
        """GPUEngine should handle 2-qubit state initialization."""
        from src.backend.gpu.gpu_engine import GPUEngine
        engine = GPUEngine(platform_name='auto')
        engine.initialize_state(2)
        state = engine.get_state()
        assert len(state) == 4
        assert abs(state[0] - 1.0) < 1e-10

    def test_simulator_gpu_none_default(self):
        """Simulator with gpu_platform='none' should behave normally."""
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(gpu_platform='none')
        assert sim.gpu_engine is None
        sim.allocate_qubit('q0')
        sim.H('q0')
        state = sim.get_state_vector()
        inv = 1.0 / math.sqrt(2.0)
        assert abs(abs(state[0]) - inv) < 1e-10
        assert abs(abs(state[1]) - inv) < 1e-10

    def test_simulator_gpu_param_passthrough(self):
        """Simulator constructor should accept gpu_platform parameter."""
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(gpu_platform='none')
        assert sim.gpu_platform == 'none'
        assert sim.gpu_engine is None

    def test_simulator_dense_cnot(self):
        """Dense CNOT should produce Bell state."""
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(gpu_platform='none')
        sim.allocate_qubit('q0')
        sim.allocate_qubit('q1')
        sim.H('q0')
        sim.CNOT('q0', 'q1')
        amps = sim.get_amplitudes_dict()
        assert len(amps) == 2
        for state, amp in amps.items():
            assert abs(abs(amp)**2 - 0.5) < 1e-10

    def test_simulator_dense_measure(self):
        """Dense measurement should return 0 or 1."""
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(gpu_platform='none')
        sim.allocate_qubit('q0')
        result = sim.measure('q0')
        assert result in (0, 1)

    def test_simulator_dense_cz(self):
        """Dense CZ should produce valid state."""
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(gpu_platform='none')
        sim.allocate_qubit('q0')
        sim.allocate_qubit('q1')
        sim.H('q0')
        sim.H('q1')
        sim.CZ('q0', 'q1')
        state = sim.get_state_vector()
        assert len(state) == 4

    def test_simulator_dense_swap(self):
        """Dense SWAP should swap qubit states."""
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(gpu_platform='none')
        sim.allocate_qubit('q0')
        sim.allocate_qubit('q1')
        sim.X('q0')
        sim.SWAP('q0', 'q1')
        state = sim.get_state_vector()
        assert abs(abs(state[2]) - 1.0) < 1e-10


# ========== Tensor Compiler Tests ==========

class TestTensorCompiler:
    def test_tensor_graph_creation(self):
        from src.ir.tensor_compiler import TensorGraph
        tg = TensorGraph()
        n1 = tg.add_node("test1", indices=["i1", "i2"])
        n2 = tg.add_node("test2", indices=["i2", "i3"])
        tg.add_edge(n1.id, n2.id, "i2")
        summary = tg.summary()
        assert summary['num_tensors'] == 2
        assert summary['num_contractions'] == 1

    def test_greedy_contraction_optimizer(self):
        from src.ir.tensor_compiler import TensorGraph, GreedyContractionOptimizer
        tg = TensorGraph()
        n1 = tg.add_node("a", indices=["i1", "i2"])
        n2 = tg.add_node("b", indices=["i2", "i3"])
        n3 = tg.add_node("c", indices=["i3", "i4"])
        tg.add_edge(n1.id, n2.id, "i2")
        tg.add_edge(n2.id, n3.id, "i3")
        opt = GreedyContractionOptimizer()
        order = opt.find_contraction_order(tg)
        assert len(order) == 2

    def test_tensor_circuit_compiler_init(self):
        from src.ir.tensor_compiler import TensorCircuitCompiler
        tcc = TensorCircuitCompiler()
        assert tcc is not None

    def test_tensor_node_rank(self):
        from src.ir.tensor_compiler import TensorNode
        tn = TensorNode(0, "test", indices=["a", "b", "c"])
        assert tn.rank() == 3

    def test_contraction_edge_repr(self):
        from src.ir.tensor_compiler import ContractionEdge
        edge = ContractionEdge(0, 1, "shared")
        assert "0" in repr(edge)
        assert "1" in repr(edge)
        assert "shared" in repr(edge)

    def test_tensor_graph_new_index(self):
        from src.ir.tensor_compiler import TensorGraph
        tg = TensorGraph()
        idx1 = tg.new_index("q")
        idx2 = tg.new_index("q")
        assert idx1 != idx2

    def test_empty_contraction(self):
        from src.ir.tensor_compiler import TensorGraph, GreedyContractionOptimizer
        tg = TensorGraph()
        tg.add_node("lone", indices=["i1"])
        opt = GreedyContractionOptimizer()
        order = opt.find_contraction_order(tg)
        assert len(order) == 0

    def test_tensor_graph_summary(self):
        from src.ir.tensor_compiler import TensorGraph
        tg = TensorGraph()
        tg.add_node("a", indices=["x", "y"])
        tg.add_node("b", indices=["y", "z", "w"])
        summary = tg.summary()
        assert summary['total_rank'] == 5


# ========== Parallel Block Tests ==========

class TestParallelBlocks:
    def test_lexer_parallel_token(self):
        code = "parallel { }"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        types = [t.type for t in tokens]
        assert TokenType.PARALLEL in types

    def test_lexer_task_token(self):
        code = "task foo()"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        types = [t.type for t in tokens]
        assert TokenType.TASK in types

    def test_parser_parallel_block(self):
        code = """eigen 2.3

qfunc doSomething() {
    qubit q
    H q
}
parallel {
    task doSomething()
}
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        found = False
        for node in ast.body:
            if isinstance(node, ParallelBlockNode):
                found = True
                assert len(node.tasks) >= 1
        assert found, "ParallelBlockNode not found in AST"

    def test_spawn_join_opcodes_exist(self):
        assert hasattr(Opcode, 'SPAWN')
        assert hasattr(Opcode, 'JOIN')
        assert Opcode.SPAWN == "SPAWN"
        assert Opcode.JOIN == "JOIN"

    def test_ebc_compiler_parallel_block(self):
        """EBC compiler should emit SPAWN and JOIN for parallel blocks."""
        from src.backend.ebc_compiler import EBCCompiler
        
        code = """eigen 2.3

func doWork() -> int {
    return 1
}

parallel {
    task doWork()
}
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        compiler = EBCCompiler()
        instructions = compiler.compile_ast(ast)
        opcodes = [i.opcode for i in instructions]
        assert Opcode.SPAWN in opcodes
        assert Opcode.JOIN in opcodes


# ========== VM GPU Parameter Tests ==========

class TestVMGPUParam:
    def test_vm_accepts_gpu_platform_none(self):
        from src.backend.vm import EigenVM
        vm = EigenVM(gpu_platform='none')
        assert vm.simulator.gpu_engine is None

    def test_vm_constructor_default(self):
        from src.backend.vm import EigenVM
        vm = EigenVM()
        assert vm.simulator.gpu_engine is None

    def test_runtime_accepts_gpu_platform_none(self):
        from src.runtime import EigenRuntime
        rt = EigenRuntime(gpu_platform='none')
        assert rt.simulator.gpu_engine is None

    def test_runtime_constructor_default(self):
        from src.runtime import EigenRuntime
        rt = EigenRuntime()
        assert rt.simulator.gpu_engine is None


# ========== MLIR Dialect Tests ==========

class TestMLIRDialect:
    def test_mlir_module_creation(self):
        from src.ir.mlir_dialect import MLIRModule, MLIRFunction
        module = MLIRModule()
        func = MLIRFunction("main")
        module.add_function(func)
        assert len(module.functions) == 1
        assert module.functions[0].name == "main"

    def test_mlir_op_creation(self):
        from src.ir.mlir_dialect import MLIROp, MLIRValue
        val = MLIRValue("q0", "qubit")
        op = MLIROp("quantum.alloc", operands=[], results=[val])
        assert op.op_name == "quantum.alloc"
        assert len(op.results) == 1
        assert op.results[0].name == "q0"

    def test_mlir_value_repr(self):
        from src.ir.mlir_dialect import MLIRValue
        val = MLIRValue("q0", "qubit")
        r = repr(val)
        assert "q0" in r
        assert "qubit" in r

    def test_ast_to_mlir_converter(self):
        from src.ir.mlir_dialect import ASTToMLIRConverter
        
        code = "eigen 2.3\nqubit q\nH q\n"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        converter = ASTToMLIRConverter()
        mlir_module = converter.convert(ast)
        assert mlir_module is not None
        assert len(mlir_module.functions) > 0

    def test_mlir_to_eqir_roundtrip(self):
        """AST -> MLIR -> EQIR should produce valid EQIR graph."""
        from src.ir.mlir_dialect import ASTToMLIRConverter, MLIRToEQIRConverter
        
        code = "eigen 2.3\nqubit q0\nqubit q1\nH q0\nCNOT q0, q1\n"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        ast_to_mlir = ASTToMLIRConverter()
        mlir_module = ast_to_mlir.convert(ast)
        
        mlir_to_eqir = MLIRToEQIRConverter()
        eqir_graph = mlir_to_eqir.convert(mlir_module)
        
        assert eqir_graph is not None
        assert len(eqir_graph.nodes) > 0

    def test_mlir_block_operations(self):
        from src.ir.mlir_dialect import MLIRBlock, MLIROp
        block = MLIRBlock("entry")
        op = MLIROp("arith.add")
        block.add_operation(op)
        assert len(block.operations) == 1


# ========== Verify Command Tests ==========

class TestVerifyCommand:
    def test_verify_command_registered(self):
        from src.cli import COMMAND_REGISTRY
        assert "verify" in COMMAND_REGISTRY
        assert "verify-equiv" in COMMAND_REGISTRY

    def test_print_results_helper_pass(self):
        from src.commands.verify import _print_results
        import io
        from contextlib import redirect_stdout
        
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_results([], [], ["Parse: OK"])
        output = buf.getvalue()
        assert "VERIFICATION PASSED" in output

    def test_print_results_helper_fail(self):
        from src.commands.verify import _print_results
        import io
        from contextlib import redirect_stdout
        
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_results(["Type Error: x"], ["Warning: y"], ["Parse: OK"])
        output = buf.getvalue()
        assert "VERIFICATION FAILED" in output
        assert "1 error" in output


# ========== Integration Tests ==========

class TestIntegration:
    def test_full_pipeline_vm(self):
        """Full compile+execute pipeline via VM should work."""
        from src.compiler import compile_to_eqir
        from src.backend.ebc_compiler import EBCCompiler
        from src.backend.vm import EigenVM
        
        code = "eigen 2.3\nqubit q0\nqubit q1\ncbit c0\ncbit c1\nH q0\nCNOT q0, q1\nmeasure q0 -> c0\nmeasure q1 -> c1\n"
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.eig', delete=False, dir='.') as f:
            f.write(code)
            tmppath = f.name
        try:
            graph, ast_node = compile_to_eqir(tmppath, '.')
            compiler = EBCCompiler()
            instructions = compiler.compile_eqir(graph)
            vm = EigenVM(gpu_platform='none')
            vm.execute(instructions)
        finally:
            os.unlink(tmppath)

    def test_full_pipeline_runtime(self):
        """Full compile+execute pipeline via Runtime should work."""
        from src.compiler import compile_to_eqir
        from src.runtime import EigenRuntime
        
        code = "eigen 2.3\nqubit q0\ncbit c0\nH q0\nmeasure q0 -> c0\n"
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.eig', delete=False, dir='.') as f:
            f.write(code)
            tmppath = f.name
        try:
            graph, ast_node = compile_to_eqir(tmppath, '.')
            runtime = EigenRuntime(gpu_platform='none')
            runtime.execute(graph)
        finally:
            os.unlink(tmppath)

    def test_compile_to_eqir_via_mlir(self):
        """compile_to_eqir should use the MLIR pipeline and produce valid EQIR."""
        from src.compiler import compile_to_eqir
        
        code = "eigen 2.3\nqubit q0\nH q0\n"
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.eig', delete=False, dir='.') as f:
            f.write(code)
            tmppath = f.name
        try:
            graph, ast_node = compile_to_eqir(tmppath, '.')
            assert graph is not None
            assert len(graph.nodes) > 0
        finally:
            os.unlink(tmppath)
