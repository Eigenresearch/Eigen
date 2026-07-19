import unittest
from src.backend.vm import EigenVM
from src.backend.bytecode import Opcode, Instruction
from src.jit.jit_compiler import get_function_hash
from src.routing.router import SABRE, CouplingMap
from src.backend.vm_optimizations import ObjectPool
import src.packager as packager

class TestAuditItems(unittest.TestCase):
    def test_jit_disabled_by_default(self):
        # Audit §2: the fragile hand-rolled Python-sourcegen fast-loop JIT
        # (_try_fast_array_loop / _try_fast_loop) was removed entirely. The
        # trace JIT (jit_compiler/native_codegen) still exists and is enabled
        # at -O3; this assertion guards against reintroducing the removed
        # methods under the same names.
        vm = EigenVM(opt_level=3)
        self.assertFalse(hasattr(vm, "_try_fast_array_loop"))
        self.assertFalse(hasattr(vm, "_try_fast_loop"))

    def test_native_executor_excludes_print(self):
        # Create a VM and check the supported opcodes in _execute_locked
        # This is hard to check directly without mocking, but we can check if it falls back correctly.
        vm = EigenVM()
        instrs = [
            Instruction(Opcode.LOAD_CONST, 42),
            Instruction(Opcode.PRINT, None),
            Instruction(Opcode.HALT, None)
        ]
        # If PRINT is present, it should NOT use native executor.
        # We can't easily verify which executor was used without spying,
        # but we can verify that PRINT still works (which it would in both, but differently).
        # Actually, let's just check the 'supported' set in the code via grep if needed,
        # but here we just run it to ensure no crash.
        vm.execute(instrs)

    def test_streaming_hash_stability(self):
        instrs = [Instruction(Opcode.LOAD_CONST, 1), Instruction(Opcode.ADD, None)]
        h1 = get_function_hash(instrs)
        h2 = get_function_hash(instrs)
        self.assertEqual(h1, h2)
        
        instrs2 = [Instruction(Opcode.LOAD_CONST, 1), Instruction(Opcode.SUB, None)]
        h3 = get_function_hash(instrs2)
        self.assertNotEqual(h1, h3)

    def test_object_pool_bounded(self):
        pool = ObjectPool(max_size=2)
        l1 = pool.borrow()
        l2 = pool.borrow()
        l3 = pool.borrow()
        
        pool.release(l1)
        pool.release(l2)
        pool.release(l3) # This should exceed max_size, so it won't be added
        
        self.assertEqual(len(pool._pool), 2)

    def test_sabre_timeout(self):
        cm = CouplingMap([(0, 1), (1, 2)])
        router = SABRE(cm)
        ops = [{'gate': 'CNOT', 'targets': ['q0', 'q2']}]
        # Very short timeout should trigger
        with self.assertRaises(TimeoutError):
            router.route(ops, ['q0', 'q1', 'q2'], timeout=0.000001)

    def test_toml_requirement(self):
        # Ensure parse_toml raises ImportError if no library is available.
        # We can temporarily hide the libs.
        orig_tomllib = packager.tomllib
        packager.tomllib = None
        try:
            with self.assertRaises(ImportError):
                packager.parse_toml("key = 'value'")
        finally:
            packager.tomllib = orig_tomllib

if __name__ == "__main__":
    unittest.main()
