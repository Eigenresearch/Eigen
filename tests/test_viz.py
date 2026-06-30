import unittest
import os
import subprocess

class TestVizCommand(unittest.TestCase):
    def test_viz_generation(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(test_dir)
        python_exe = os.path.join(workspace_root, ".venv", "Scripts", "python.exe")
        main_py = os.path.join(workspace_root, "src", "main.py")

        source_file = os.path.join(test_dir, "temp_viz_test.eig")
        output_svg = os.path.join(test_dir, "temp_viz_test.svg")

        with open(source_file, "w", encoding="utf-8") as f:
            f.write("""eigen 2.5
            qubit q0
            qubit q1
            H q0
            CNOT q0, q1
            """)

        try:
            result = subprocess.run([
                python_exe, main_py, "viz", source_file, "-o", output_svg
            ], capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, f"viz command failed: {result.stderr}")
            self.assertTrue(os.path.exists(output_svg))
            with open(output_svg, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<svg", content)
            self.assertIn("q0", content)
            self.assertIn("q1", content)
            self.assertIn("H", content)
            self.assertIn("cnot-target-outer", content)
        finally:
            if os.path.exists(source_file):
                os.remove(source_file)
            if os.path.exists(output_svg):
                os.remove(output_svg)

if __name__ == '__main__':
    unittest.main()
