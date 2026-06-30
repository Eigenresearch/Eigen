import unittest
import os
import subprocess

class TestGenericsMonomorphization(unittest.TestCase):
    def test_generics_compilation_and_execution(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(test_dir)
        python_exe = os.path.join(workspace_root, ".venv", "Scripts", "python.exe")
        main_py = os.path.join(workspace_root, "src", "main.py")

        source_file = os.path.join(test_dir, "temp_generics_test.eig")

        # Define a generic max function and call it with int and float parameters
        with open(source_file, "w", encoding="utf-8") as f:
            f.write("""eigen 2.5
            
            func max<T>(a: T, b: T) -> T {
                if a > b {
                    return a
                }
                return b
            }
            
            let x: int = max(5, 10)
            let y: float = max(0.5, 0.9)
            
            print x
            print y
            """)

        try:
            # Clean cache first to ensure type checking runs fresh
            cache_db = os.path.join(workspace_root, ".eigen_cache")
            if os.path.exists(cache_db):
                import shutil
                shutil.rmtree(cache_db, ignore_errors=True)

            result = subprocess.run([
                python_exe, main_py, "run", source_file, "--vm"
            ], capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, f"Generics execution failed: {result.stderr}")
            self.assertIn("10", result.stdout)
            self.assertIn("0.9", result.stdout)
        finally:
            if os.path.exists(source_file):
                os.remove(source_file)

if __name__ == '__main__':
    unittest.main()
