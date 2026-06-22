import unittest
import os
import sys
import subprocess
import shutil
import tempfile

class TestCLICommands(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.main_path = os.path.join(self.workspace_root, "src", "main.py")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def run_cli(self, args, cwd=None):
        if cwd is None:
            cwd = self.temp_dir
        cmd = [sys.executable, self.main_path] + args
        # Add workspace root to PYTHONPATH so imports resolve correctly
        env = os.environ.copy()
        env["PYTHONPATH"] = self.workspace_root + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)

    def test_init_command(self):
        res = self.run_cli(["init", "test_pkg"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("Initialized new Eigen package", res.stdout)
        self.assertIn("Created template entrypoint", res.stdout)
        
        toml_path = os.path.join(self.temp_dir, "eigen.toml")
        main_eig_path = os.path.join(self.temp_dir, "src", "main.eig")
        
        self.assertTrue(os.path.isfile(toml_path))
        self.assertTrue(os.path.isfile(main_eig_path))
        
        with open(toml_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn('name = "test_pkg"', content)

    def test_build_and_exec_command(self):
        # 1. Initialize a package first
        self.run_cli(["init", "my_project"])
        main_eig_path = os.path.join(self.temp_dir, "src", "main.eig")
        
        # 2. Build the main.eig file
        res = self.run_cli(["build", main_eig_path])
        self.assertEqual(res.returncode, 0)
        self.assertIn("Compiling", res.stdout)
        self.assertIn("Compilation successful", res.stdout)
        
        ebc_path = os.path.join(self.temp_dir, "src", "main.ebc")
        self.assertTrue(os.path.isfile(ebc_path))
        
        # 3. Exec the compiled .ebc file
        res_exec = self.run_cli(["exec", ebc_path])
        self.assertEqual(res_exec.returncode, 0)
        self.assertIn("Hello from Eigen!", res_exec.stdout)

    def test_bench_command(self):
        # We need benchmarks/ directory inside target cwd
        # Let's copy the project's benchmarks directory to our temp dir
        src_bench_dir = os.path.join(self.workspace_root, "benchmarks")
        if os.path.isdir(src_bench_dir):
            shutil.copytree(src_bench_dir, os.path.join(self.temp_dir, "benchmarks"))
            res = self.run_cli(["bench"])
            self.assertEqual(res.returncode, 0)
            self.assertIn("EIGEN BENCHMARK SUITE", res.stdout)
            self.assertIn("factorial.eig", res.stdout)
            self.assertIn("fibonacci.eig", res.stdout)

if __name__ == "__main__":
    unittest.main()
