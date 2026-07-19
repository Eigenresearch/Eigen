"""§9.1 — Mutation testing configuration and runner.

Provides a configuration file for mutmut and a helper to run
mutation testing on a subset of source modules.

Mutation testing verifies the quality of our test suite by
injecting small mutations into the source code and checking
whether the tests catch them (mutations should be "killed"
by failing tests). Survivors indicate gaps in test coverage.
"""
from __future__ import annotations

import dataclasses

MUTMUT_CONFIG = {
    "paths_to_mutate": [
        "src/backend/bytecode.py",
        "src/backend/vm.py",
        "src/frontend/lexer.py",
        "src/frontend/parser.py",
        "src/ir/optimizer.py",
        "src/ir/ir_graph.py",
        "src/sparse_simulator.py",
        "src/tensor_network/mps.py",
        "src/packager.py",
        "src/ffi.py",
        "src/project_scalability.py",
        "src/compilation_research.py",
        "src/quantum_tomography.py",
    ],
    "tests_dir": "tests/",
    "runner": "python -m pytest -x -q --timeout=30",
    "max_modules": 5,
}


def write_mutmut_config(path: str = "setup.cfg"):
    """Write mutmut configuration to setup.cfg."""
    config_lines = ["[mutmut]"]
    paths = ":".join(MUTMUT_CONFIG["paths_to_mutate"])
    config_lines.append(f"paths_to_mutate = {paths}")
    config_lines.append(f"tests_dir = {MUTMUT_CONFIG['tests_dir']}")
    config_lines.append(f"runner = {MUTMUT_CONFIG['runner']}")
    config = "\n".join(config_lines) + "\n"

    with open(path, "w") as f:
        f.write(config)
    return config


@dataclasses.dataclass
class MutationTestResult:
    """Result of a mutation testing run."""
    total_mutants: int
    killed: int
    survived: int
    timeout: int
    skipped: int
    mutation_score: float  # killed / (killed + survived)

    @property
    def quality_grade(self) -> str:
        if self.mutation_score >= 0.9:
            return "A"
        elif self.mutation_score >= 0.8:
            return "B"
        elif self.mutation_score >= 0.7:
            return "C"
        elif self.mutation_score >= 0.6:
            return "D"
        return "F"


def parse_mutmut_results(output: str) -> MutationTestResult:
    """Parse mutmut output text into a MutationTestResult."""
    import re
    killed = 0
    survived = 0
    timeout = 0
    skipped = 0

    # Try to parse "label: N" format
    for label, _var in [("killed", "killed"), ("survived", "survived"),
                         ("timeout", "timeout"), ("skipped", "skipped")]:
        m = re.search(rf'{label}[:\s]+(\d+)', output, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if label == "killed":
                killed = val
            elif label == "survived":
                survived = val
            elif label == "timeout":
                timeout = val
            elif label == "skipped":
                skipped = val

    # Fallback: count occurrences if no numbers found
    if killed == 0 and "killed" in output.lower():
        killed = output.lower().count("killed")
    if survived == 0 and "survived" in output.lower():
        survived = output.lower().count("survived")

    total = killed + survived + timeout + skipped
    if killed + survived > 0:
        score = killed / (killed + survived)
    else:
        score = 0.0
    return MutationTestResult(
        total_mutants=total,
        killed=killed,
        survived=survived,
        timeout=timeout,
        skipped=skipped,
        mutation_score=score,
    )


def run_mutation_testing(modules: list[str] | None = None) -> str:
    """Run mutation testing on the specified modules.

    This is a surface function — actual execution requires
    mutmut to be installed and the test suite to be green.
    Returns the mutmut stdout output.
    """
    import subprocess
    import sys

    paths = modules or MUTMUT_CONFIG["paths_to_mutate"][:2]
    cmd = [sys.executable, "-m", "mutmut", "run",
           "--paths-to-mutate", ":".join(paths)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=600)
        return result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "mutmut not available or timed out"
