"""Cooperative task lifecycle for Eigen VM async bytecode.

Tasks are executed by the VM dispatch loop, never by OS worker threads.  The
scheduler owns task state and validation; the VM owns instruction execution.
"""
from __future__ import annotations

import dataclasses
import enum
import itertools
from typing import Any


class TaskState(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclasses.dataclass
class VMTask:
    task_id: int
    target: int
    func_name: str
    args: tuple[Any, ...]
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: BaseException | str | None = None
    yield_count: int = 0

    @property
    def done(self) -> bool:
        return self.state in (TaskState.COMPLETED, TaskState.FAILED)


class CooperativeTaskScheduler:
    """Validates and records lifecycle transitions for VM tasks."""

    def __init__(self):
        self._ids = itertools.count(1)
        self._tasks: dict[int, VMTask] = {}

    def reset(self) -> None:
        self._ids = itertools.count(1)
        self._tasks.clear()

    def create(self, target: int, func_name: str, args: list[Any]) -> VMTask:
        if not isinstance(target, int) or target < 0:
            raise ValueError(f"Invalid async function target: {target!r}")
        task = VMTask(next(self._ids), target, func_name, tuple(args))
        self._tasks[task.task_id] = task
        return task

    def start(self, task: VMTask) -> None:
        self._require(task, TaskState.PENDING)
        task.state = TaskState.RUNNING

    def yield_task(self, task: VMTask) -> None:
        if task.state not in (TaskState.RUNNING, TaskState.SUSPENDED):
            raise RuntimeError(
                f"Task {task.task_id} cannot yield from {task.state.value}"
            )
        task.yield_count += 1
        task.state = TaskState.RUNNING

    def complete(self, task: VMTask, result: Any) -> None:
        if task.done:
            raise RuntimeError(f"Task {task.task_id} is already finished")
        task.result = result
        task.state = TaskState.COMPLETED

    def fail(self, task: VMTask, error: BaseException | str) -> None:
        if task.done:
            return
        task.error = error
        task.state = TaskState.FAILED

    def result(self, task: VMTask) -> Any:
        if task.state == TaskState.FAILED:
            if isinstance(task.error, BaseException):
                raise task.error
            raise RuntimeError(f"AsyncTaskError: {task.error}")
        if task.state != TaskState.COMPLETED:
            raise RuntimeError(
                f"Task {task.task_id} is not complete ({task.state.value})"
            )
        return task.result

    def get(self, task_id: int) -> VMTask:
        return self._tasks[task_id]

    def _require(self, task: VMTask, expected: TaskState) -> None:
        if self._tasks.get(task.task_id) is not task:
            raise RuntimeError(f"Task {task.task_id} belongs to another scheduler")
        if task.state != expected:
            raise RuntimeError(
                f"Task {task.task_id} expected {expected.value}, "
                f"got {task.state.value}"
            )
