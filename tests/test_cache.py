import unittest
import os
import shutil
import tempfile
from src.main import get_project_hash, load_from_cache, save_to_cache
from src.bytecode import Instruction

class TestCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = os.path.join(self.temp_dir, "test.eig")
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("eigen 1.0\nlet x: int = 42\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_cache_save_and_load(self):
        # Create dummy instructions
        instructions = [
            Instruction("LOAD_CONST", 42),
            Instruction("STORE_VAR", "x")
        ]
        
        # Initially, there is no cache
        loaded = load_from_cache(self.file_path, self.temp_dir, "ebc")
        self.assertIsNone(loaded)
        
        # Save to cache
        save_to_cache(self.file_path, self.temp_dir, "ebc", instructions)
        
        # Load from cache
        loaded = load_from_cache(self.file_path, self.temp_dir, "ebc")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].opcode, "LOAD_CONST")
        self.assertEqual(loaded[0].arg, 42)
        self.assertEqual(loaded[1].opcode, "STORE_VAR")
        self.assertEqual(loaded[1].arg, "x")

if __name__ == "__main__":
    unittest.main()
