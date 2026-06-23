import sys
import os
import unittest
from src.cli import register_command

@register_command("test")
def test_command(args, workspace_root):
    print("Running Eigen Test Suite...")
    suite = unittest.defaultTestLoader.discover(os.path.join(os.path.dirname(__file__), "../../tests"))
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
