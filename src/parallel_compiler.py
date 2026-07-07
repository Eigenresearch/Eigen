"""§8.2 — Parallel compilation: compile multiple modules
simultaneously using thread/process pools.

When multiple modules need to be compiled, they can be compiled
in parallel since module compilation is independent (unless
there are cross-module dependencies that must be resolved first).
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import typing


@dataclasses.dataclass
class CompilationTask:
    """A single module compilation task."""
    module_name: str
    source_path: str
    dependencies: list[str] = dataclasses.field(default_factory=list)
    status: str = "pending"  # pending → compiling → done / error
    result: typing.Any = None
    error: str | None = None


@dataclasses.dataclass
class CompilationResult:
    """Result of a parallel compilation run."""
    tasks: list[CompilationTask]
    total_duration_s: float
    parallelism: int

    @property
    def succeeded(self) -> int:
        return sum(1 for t in self.tasks if t.status == "done")

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tasks if t.status == "error")


def topological_compile_order(tasks: list[CompilationTask]) -> list[str]:
    """Determine the order in which modules can be compiled,
    respecting dependencies. Returns a list of module names
    in compilation order.
    """
    task_map = {t.module_name: t for t in tasks}
    visited = set()
    order = []

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        task = task_map.get(name)
        if task:
            for dep in task.dependencies:
                if dep in task_map:
                    visit(dep)
            order.append(name)

    for t in tasks:
        visit(t.module_name)
    return order


def compile_in_parallel(tasks: list[CompilationTask],
                          compile_fn: typing.Callable[[CompilationTask],
                                                       typing.Any],
                          max_workers: int = 4,
                          timeout_s: float = 300.0
                          ) -> CompilationResult:
    """Compile multiple modules in parallel.

    Respects dependencies: a module is only compiled after all its
    dependencies are done. Uses ThreadPoolExecutor for I/O-bound
    compilation or ProcessPoolExecutor for CPU-bound work.

    Args:
        tasks: List of compilation tasks with dependencies.
        compile_fn: Function that takes a CompilationTask and returns
                    the compiled artifact.
        max_workers: Maximum number of parallel workers.
        timeout_s: Total timeout for the compilation run.

    Returns:
        CompilationResult with per-task status.
    """
    import time
    t0 = time.monotonic()

    order = topological_compile_order(tasks)
    task_map = {t.module_name: t for t in tasks}

    # Group tasks into waves that can run in parallel
    done = set()
    remaining = set(t.module_name for t in tasks)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        while remaining:
            # Find all tasks whose dependencies are done
            ready = [name for name in order
                       if name in remaining
                       and all(d in done for d in task_map[name].dependencies
                                if d in task_map)]
            if not ready:
                # Circular dependency or all remaining have
                # unresolvable deps — mark as error
                for name in remaining:
                    task_map[name].status = "error"
                    task_map[name].error = "Unresolved dependencies"
                break

            # Submit ready tasks in parallel
            futures = {}
            for name in ready:
                task = task_map[name]
                task.status = "compiling"
                future = executor.submit(compile_fn, task)
                futures[future] = name

            # Wait for this wave to complete
            for future in concurrent.futures.as_completed(
                    futures, timeout=timeout_s):
                name = futures[future]
                task = task_map[name]
                try:
                    task.result = future.result()
                    task.status = "done"
                except Exception as e:
                    task.status = "error"
                    task.error = str(e)
                done.add(name)
                remaining.discard(name)

    elapsed = time.monotonic() - t0
    return CompilationResult(
        tasks=tasks,
        total_duration_s=elapsed,
        parallelism=max_workers,
    )
