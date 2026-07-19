import unittest
import os
import tempfile
from src.backend.llvm_compiler import LLVMCompiler
from src.backend.scheduler import TaskScheduler
from src.ir.ssa.cfg import BasicBlock
from src.backend.bytecode import Instruction, Opcode
from src.compiler import load_from_cache, save_to_cache

class TestWeakSpotsFixes(unittest.TestCase):
    def test_llvm_compiler_simple(self):
        blocks = []
        b0 = BasicBlock(0)
        b0.instructions = [
            Instruction(Opcode.LOAD_CONST, 42),
            Instruction(Opcode.STORE_VAR, "x"),
            Instruction(Opcode.Q_ALLOC, "q0"),
            Instruction(Opcode.Q_GATE, ("H", ["q0"])),
            Instruction(Opcode.HALT)
        ]
        blocks.append(b0)
        
        compiler = LLVMCompiler()
        llvm_ir = compiler.compile_ssa(blocks)
        
        self.assertIn("; ModuleID = 'EigenLLVMModule'", llvm_ir)
        self.assertIn("%q0 = call %Qubit* @__quantum__rt__qubit_allocate()", llvm_ir)
        self.assertIn("call void @__quantum__qis__h__body(%Qubit* %q0)", llvm_ir)
        self.assertIn("ret void", llvm_ir)

    def test_task_scheduler_dag(self):
        scheduler = TaskScheduler(max_workers=2)
        
        def fn_a():
            return 10
            
        def fn_b(val):
            return val + 5
            
        def fn_c(val):
            return val * 2
            
        scheduler.add_task("A", fn_a)
        scheduler.add_task("B", fn_b, args=(10,), dependencies=["A"])
        scheduler.add_task("C", fn_c, args=(10,), dependencies=["A"])
        
        results = scheduler.run_scheduler()
        self.assertEqual(results["A"], 10)
        self.assertEqual(results["B"], 15)
        self.assertEqual(results["C"], 20)

    def test_incremental_caching(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.eig")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("eigen 1.0\nlet x: int = 42\n")
                
            dummy_ast = {"dummy": "ast_node"}
            save_to_cache(file_path, temp_dir, "ast", dummy_ast)
            
            loaded = load_from_cache(file_path, temp_dir, "ast")
            self.assertEqual(loaded, dummy_ast)

    def test_crash_report_creation(self):
        from src.crash_report import write_crash_report
        class DummyFrame:
            def __init__(self, func_name, current_line, locals_dict):
                self.func_name = func_name
                self.current_line = current_line
                self.locals = locals_dict
        
        stack = [DummyFrame("foo", 10, {"a": 1}), DummyFrame("bar", 20, {"b": 2})]
        
        import glob
        for f in glob.glob("crash-*.log"):
            os.remove(f)
            
        try:
            write_crash_report(ValueError("Test error"), stack, 42, "ADD", {"global_var": 42})
            crash_files = glob.glob("crash-*.log")
            self.assertTrue(len(crash_files) > 0)
            with open(crash_files[0], "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("EIGEN VM CRASH REPORT", content)
            self.assertIn("Test error", content)
            self.assertIn("foo", content)
            self.assertIn("bar", content)
        finally:
            for f in glob.glob("crash-*.log"):
                os.remove(f)

    def test_console_printing_no_directives(self):
        # Verify that BUG-002 fixes print formatting to not have directive prefixes
        from src.backend.vm import EigenVM
        from src.backend.bytecode import Instruction, Opcode
        import io
        import src.backend.vm as vm_module
        
        old_native = vm_module.native
        vm_module.native = None
        try:
            vm = EigenVM()
            new_stdout = io.StringIO()
            vm.output_stream = new_stdout
            
            vm.execute([
                Instruction(Opcode.LOAD_CONST, "hello world"),
                Instruction(Opcode.PRINT),
                Instruction(Opcode.HALT)
            ])
                
            output = new_stdout.getvalue()
            self.assertEqual(output.strip(), "hello world")
        finally:
            vm_module.native = old_native

    def test_simulator_memory_limit(self):
        from src.simulator import PythonDenseStatevector
        sv = PythonDenseStatevector()
        
        # Mock _state to have 2**25 elements
        import numpy as np
        sv._state = np.zeros(1 << 25, dtype=complex)
        
        # Now allocate_qubit should raise MemoryError
        with self.assertRaises(MemoryError):
            sv.allocate_qubit()
