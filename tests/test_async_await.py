"""
Tests for src/language_extensions/async_await.py — sol.md §3.1.
"""
import unittest

from src.language_extensions.async_await import (
    AsyncTask,
    AsyncError,
    await_,
)


def _simple_coroutine(value):
    """An async body that yields nothing and just returns."""
    return value
    yield  # never reached but makes this a generator


def _cooperative_coroutine(yielded_values):
    """Yields each value in `yielded_values`, then returns 'done'."""
    for v in yielded_values:
        yield v
    return "done"


def _raising_coroutine(exc):
    """Yields once then raises."""
    yield "first"
    raise exc("boom from coroutine")


def _value_propagating_coroutine():
    """Echoes `value` from each yield back via `yield x`."""
    val = yield "first"
    val = yield f"second({val})"
    return f"final({val})"


def _nested_yield():
    """Yields twice, returns once"""
    yield 1
    yield 2
    return 3


class TestAsyncTaskBasics(unittest.TestCase):
    def test_start_simple_coroutine(self):
        task = AsyncTask.start(_simple_coroutine, 42)
        self.assertIsInstance(task, AsyncTask)
        self.assertFalse(task.is_done)

    def test_start_non_generator_raises(self):
        with self.assertRaises(AsyncError):
            AsyncTask.start(lambda: 5)  # not a generator

    def test_step_returns_first_yield(self):
        task = AsyncTask.start(_cooperative_coroutine, [1, 2, 3])
        self.assertEqual(task.step(None), 1)

    def test_run_to_completion_no_yields(self):
        task = AsyncTask.start(_simple_coroutine, 99)
        result = task.run_to_completion()
        self.assertEqual(result, 99)
        self.assertTrue(task.is_done)


class TestAwaitDucksAsyncAwaitSurface(unittest.TestCase):
    def test_await_returns_each_yield_in_turn(self):
        task = AsyncTask.start(_cooperative_coroutine, [10, 20, 30])
        self.assertEqual(await_(task), 10)
        self.assertEqual(await_(task), 20)
        self.assertEqual(await_(task), 30)
        # The next await_ should complete the generator and return 'done'.
        self.assertEqual(await_(task), "done")
        self.assertTrue(task.is_done)

    def test_await_returns_final_when_no_yields(self):
        task = AsyncTask.start(_simple_coroutine, 42)
        # No yields — the first await_ completes and returns 42.
        self.assertEqual(await_(task), 42)
        self.assertTrue(task.is_done)

    def test_await_idempotent_after_completion(self):
        task = AsyncTask.start(_simple_coroutine, 5)
        task.run_to_completion()
        # Calling await_ again returns the stored result.
        self.assertEqual(await_(task), 5)

    def test_await_non_task_raises(self):
        with self.assertRaises(AsyncError):
            await_(42)


class TestCoroutineYieldsValues(unittest.TestCase):
    def test_three_yields_then_return(self):
        task = AsyncTask.start(_nested_yield)
        self.assertEqual(task.step(None), 1)
        self.assertEqual(task.step(None), 2)
        self.assertEqual(task.step(None), 3)
        self.assertTrue(task.is_done)

    def test_run_to_completion_ignores_intermediate_yields(self):
        task = AsyncTask.start(_nested_yield)
        result = task.run_to_completion()
        self.assertEqual(result, 3)


class TestCoroutineErrorPropagation(unittest.TestCase):
    def test_exception_wrapped_in_async_error(self):
        task = AsyncTask.start(_raising_coroutine, ValueError)
        # First step yields "first" successfully.
        self.assertEqual(task.step(None), "first")
        # Second step raises.
        with self.assertRaises(AsyncError) as cm:
            task.step(None)
        self.assertIsInstance(cm.exception.cause, ValueError)
        self.assertIn("boom", str(cm.exception.cause))
        self.assertTrue(task.is_done)

    def test_exception_after_done_raises_again(self):
        task = AsyncTask.start(_raising_coroutine, ValueError)
        # Consume the yield
        task.step(None)
        try:
            task.step(None)
        except AsyncError:
            pass
        # Further step should re-raise AsyncError (because _error is set).
        with self.assertRaises(AsyncError):
            task.step(None)


class TestSendValueThroughYield(unittest.TestCase):
    """A generator's `x = yield something` lets the caller send a
    value through `task.step(value)`."""

    def test_send_value_back_into_coroutine(self):
        task = AsyncTask.start(_value_propagating_coroutine)
        self.assertEqual(task.step(None), "first")
        # Send "x" — coroutine stores it in val, yields "second(x)"
        self.assertEqual(task.step("x"), "second(x)")
        # Send "y" — coroutine stores and returns final(y)
        self.assertEqual(task.step("y"), "final(y)")


class TestTaskResultAccess(unittest.TestCase):
    def test_result_returns_when_done(self):
        task = AsyncTask.start(_simple_coroutine, 42)
        task.run_to_completion()
        self.assertEqual(task.result, 42)

    def test_result_raises_error_after_propagation(self):
        task = AsyncTask.start(_raising_coroutine, ValueError)
        with self.assertRaises(AsyncError):
            task.run_to_completion()
        # Now .result should re-raise the original ValueError... but
        # because the AsyncError wrapper was discarded in `run_to_completion`'s
        # try/except, `.result` returns the underlying error via `self._error`.
        with self.assertRaises(ValueError):
            task.result()

    def test_result_returns_none_if_no_yields_or_returns(self):
        def _empty():
            return None
            yield
        task = AsyncTask.start(_empty)
        task.run_to_completion()
        self.assertIsNone(task.result)


class TestAsyncTaskIdempotency(unittest.TestCase):
    def test_step_returns_same_after_done(self):
        task = AsyncTask.start(_simple_coroutine, 5)
        task.run_to_completion()
        # Multiple calls should return the same result without
        # re-driving the (finished) generator.
        for _ in range(3):
            self.assertEqual(task.step(None), 5)

    def test_run_to_completion_safe_after_done(self):
        task = AsyncTask.start(_nested_yield)
        first = task.run_to_completion()
        second = task.run_to_completion()
        self.assertEqual(first, second)


class TestCooperativeScenario(unittest.TestCase):
    """Simulate a quantum-cloud async-job pattern:

      async func submit_circuit(c):
        task_id = await cloud_submit(c)
        result = await cloud_poll(task_id)
        return result

    Both "cloud_submit" and "cloud_poll" are cooperative coroutines
    that yield once each. We wrap them in AsyncTasks and chain via
    `await_`."""

    def setUp(self):
        calls = []

        def cloud_submit(c):
            calls.append(("submit", c))
            yield "submitting"   # simulated network
            return f"task-{c}"

        def cloud_poll(task_id):
            calls.append(("poll", task_id))
            yield "polling"
            return f"result-for-{task_id}"

        self.calls = calls
        self._submit_factory = cloud_submit
        self._poll_factory = cloud_poll

    def test_chained_async_jobs_execute_in_order(self):
        async def submit_circuit(c):
            submit = AsyncTask.start(self._submit_factory, c)
            task_id = await_(submit)
            poll = AsyncTask.start(self._poll_factory, task_id)
            result = await_(poll)
            return result

        # Note: Python doesn't support `async def` with `await_()`,
        # but since we use generators, we can build it with `yield`:
        def submit_circuit_gen():
            c = "circ"
            submit = AsyncTask.start(self._submit_factory, c)
            task_id = yield from submit.generator
            poll = AsyncTask.start(self._poll_factory, task_id)
            result = yield from poll.generator
            return result

        task = AsyncTask(submit_circuit_gen())
        result = task.run_to_completion()
        self.assertEqual(result, "result-for-task-circ")
        # Verify both calls happened
        self.assertEqual(self.calls, [("submit", "circ"),
                                         ("poll", "task-circ")])


def _generator_with_yield_from():
    result1 = yield from _cooperative_coroutine([1, 2])
    return f"chained-{result1}"


class TestYieldFrom(unittest.TestCase):
    def test_yield_from_in_async_body(self):
        task = AsyncTask(_generator_with_yield_from())
        # The wrapper yields 1, 2, then "done" via the inner
        # coroutine's return.
        self.assertEqual(task.step(None), 1)
        self.assertEqual(task.step(None), 2)
        # Next step completes the inner generator, yielding control
        # to the outer which immediately returns.
        result = task.step(None)
        # The inner returns "done" — outer completes with "chained-done".
        self.assertEqual(result, "chained-done")
        self.assertTrue(task.is_done)


if __name__ == "__main__":
    unittest.main()
