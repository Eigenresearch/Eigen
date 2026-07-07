"""§3.1 — Async/Await surface-level envelope.

Roadmap checkbox:

    - [ ] Async/Await — нативная асинхронность

Eigen is single-threaded with a cooperative multi-tasking VM:
no real OS threads for quantum simulation (the qubit state vector
isn't thread-safe). The §3.1 "async/await" checkbox is therefore
an API-level envelope:

  * `AsyncTask(coroutine)` — wraps a Python coroutine or generator
    representing an `async func` body. The task is **executed
    synchronously to completion** in this envelope — we do not
    use an event loop. The envelope's purpose is:

    - to give callers a stable `await`-shaped API,
    - to permit cooperative `yield` points (via `AsyncTask.yield_(value)`)
      so callers can compose coroutines that *appear* to wait,
    - to make `await` deterministic for the audit-equivalence
      tests under §1.2 (determinism).

  * `await_(task)` — synchronously drives the wrapped coroutine to
    its next yield or completion and returns either the yielded
    value or the final result.

  * `AsyncError` — raised when an `async func` raises. The original
    exception is preserved on `.cause`.

API summary
-----------
  * `AsyncTask(fn, *args, **kwargs)` — invokes `fn` (the coroutine
    factory) immediately and stores the resulting generator. `fn`
    must return a generator (typically `async def`/`yield`-based).
  * `AsyncTask.is_done` — bool after `await_` returns the StopIteration
    value.
  * `AsyncTask.result` — the final return value (or re-raises if an
    exception propagated).
  * `await_(task)` — drives the task forward one step or completes
    it. If the task is already complete, returns its result without
    further execution.

Limitations compared to real async/await:
  - No true concurrency; `await_` blocks.
  - No event loop / I/O multiplexing.
  - No cancellation tokens.
  - This is the §3.1 envelope: the *surface* is what the runtime
    exposes to user programs; the *implementation* can be enriched
    in Phase E (Hybrid Orchestration, §6.3) when we wire I/O
    backends in.
"""
from __future__ import annotations

import dataclasses
import typing


class AsyncError(Exception):
    """Wraps an exception propagated out of an `async func` body."""

    def __init__(self, message: str, cause: BaseException):
        super().__init__(message)
        self.cause = cause


class AsyncStateError(Exception):
    """Raised when `await_` is called on a task that has already
    ended with an error or already returned a value."""
    pass


@dataclasses.dataclass
class AsyncTask:
    """A cooperatively-scheduled, synchronously-executed task. The
    underlying `generator` must support the generator protocol
    (`send`, `throw`)."""
    generator: typing.Generator
    _done: bool = dataclasses.field(default=False, repr=False)
    _result: typing.Any = dataclasses.field(default=None, repr=False)
    _error: typing.Optional[BaseException] = dataclasses.field(default=None, repr=False)

    @classmethod
    def start(cls, fn: typing.Callable, *args, **kwargs) -> "AsyncTask":
        """Construct an `AsyncTask` from a callable that returns a
        generator."""
        gen = fn(*args, **kwargs)
        if not hasattr(gen, "send"):
            raise AsyncError(
                "AsyncTask body must return a generator (use `yield` expressions)",
                cause=TypeError(type(gen).__name__))
        return cls(generator=gen)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def result(self) -> typing.Any:
        if self._error is not None:
            raise self._error
        return self._result

    def step(self, value: typing.Any = None) -> typing.Any:
        """Drive the task forward by one yield. Returns whatever the
        generator yields, or `self._result` (the final return value)
        if it completes. Raises `AsyncError` if the generator raises
        an exception."""
        if self._done:
            if self._error is not None:
                raise AsyncError(str(self._error), cause=self._error) \
                    from self._error
            return self._result
        try:
            return self.generator.send(value)
        except StopIteration as stop:
            self._done = True
            self._result = stop.value
            return self._result
        except AsyncError:
            raise
        except Exception as e:
            self._done = True
            self._error = e
            raise AsyncError(
                f"Async task raised {type(e).__name__}: {e}", cause=e) from e

    def run_to_completion(self) -> typing.Any:
        """Drive the task to completion, ignoring intermediate yields.
        Returns the final result. Equivalent to `await task` in
        languages without explicit `yield`."""
        if self._done:
            return self.result
        last_yield = None
        while not self._done:
            last_yield = self.step(last_yield)
        return self._result


def await_(task: AsyncTask) -> typing.Any:
    """User-facing `await`. Alias for `task.step(None)`.

    On the first `await` call, the generator advances to its first
    yield (or to completion if it had none).

    In our synchronous envelope, repeated `await_` calls return
    successive yield values until the generator completes; the
    final `await_` returns the generator's return value (which
    becomes the task's `.result`)."""
    if not isinstance(task, AsyncTask):
        raise AsyncError(f"await_ requires AsyncTask, got {type(task).__name__}",
                         cause=TypeError(type(task).__name__))
    return task.step(None)


def yield_(value: typing.Any = None) -> typing.Any:
    """Inside an async body, suspend execution and yield `value` to
    the caller. Resumes when the caller's `await_` returns the next
    yielded value or the generator's return value (None if no more
    yields)."""
    # Use a plain yield — this function must run inside a generator.
    return (yield value)


__all__ = [
    "AsyncTask",
    "AsyncError",
    "AsyncStateError",
    "await_",
    "yield_",
]
