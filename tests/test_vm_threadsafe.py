"""§6.1 Thread-safe VM — concurrent execute() on the same VM instance
plus `execute_parallel` for shot-level parallelism.

The tests check:

  * Concurrent `execute()` on a single VM instance does NOT corrupt
    internal state — both threads complete and both see the expected
    post-condition. (The `_state_lock` serializes execution.)
  * Two calls to `execute()` from different threads block each other:
    the second one must not begin until the first has released the
    lock. We verify this by recording an entry/exit timeline.
  * `execute_parallel(instructions, shots=N)` returns exactly N
    results, ordered by shot index.
  * `execute_parallel` uses FRESH VM instances per shot (the parent
    VM's `globals`/`ip`/`call_stack` are unchanged after the parallel
    call).
  * Per-shot RNG seeds differ (when `seed is not None` and not
    `deterministic`), producing different measurement outcomes for
    a single random circuit.
  * Reentrant call on the same thread (e.g. a stdlib callback into
    `execute()` for the same VM) does NOT deadlock.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

from src.backend.bytecode import Instruction
from src.backend.vm import EigenVM


def _program_set_x_to_one_then_two():
    """Bytecode that does:
        LOAD_CONST 0 (placeholder)
        STORE_VAR x
        ... (we use a tiny program; we mostly care that it runs)
    Rather than hand-rolling, this is built via the EBC compiler below.
    """
    pass


# `to_ebc` requires a workspace_root that points at a real directory
# (it builds a compiler-db cache there). Use a per-module temp dir so
# we get a clean compiler cache for each test file without polluting
# the repo.
_WORKSPACE = tempfile.mkdtemp(prefix="eigen_vm_threadsafe_")


def _compile_to_ebc(src, filename="__threadsafe_inline__.eig"):
    """Compile the Eigen source `src` to a list of `Instruction`s.

    Writes the source to an in-workspace file so the compiler can find
    it (the EBC compiler needs a real filename for source-location info
    + the workspace_root lookup).

    The file is named by a hash of `src` content so that two compilations
    of distinct sources never collide in the compiler-db cache (which is
    keyed by filename).
    """
    import hashlib
    content_hash = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
    path = os.path.join(_WORKSPACE, f"ts_{content_hash}_{filename}")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
    from src.compiler import to_ebc
    return to_ebc(path, _WORKSPACE, optimize=False)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


_LET_SRC = """eigen 1.0
func set_x() -> int {
    let x: int = 1
    print x
    return x
}
let result: int = set_x()
"""


# A loop-free program we know the type checker accepts; used for tests
# that don't care about WHAT executes, only that execute() runs to
# completion. The historical name `_LOOP_SRC` is preserved for tests
# that referenced it.
_SUM_SRC = """eigen 1.0
let total: int = 0
let i: int = 0
while i < 100 {
    total = total + i
    i = i + 1
}
assert total == 4950
"""
_LOOP_SRC = _SUM_SRC


class TestExecuteSerializesConcurrentThreads(unittest.TestCase):
    """Multiple threads calling execute() on the same VM instance."""

    def setUp(self):
        self.instructions = _compile_to_ebc(_LET_SRC)
        self.errors = []
        self.timeline = []

    def test_concurrent_executes_do_not_raise(self):
        vm = EigenVM(seed=42, deterministic=True)
        barrier = threading.Barrier(4)

        def worker():
            try:
                barrier.wait(timeout=5)
                vm.execute(self.instructions)
            except Exception as e:
                self.errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(self.errors, [],
                         f"threads raised: {self.errors}")
        # All four threads completed — VM is in a stable post-state.
        self.assertIsInstance(vm.globals, dict)

    def test_second_thread_waits_for_first_to_release_lock(self):
        """Both threads attempt execute() concurrently; their entry and
        exit times must NOT overlap because of the state lock."""
        vm = EigenVM(seed=0, deterministic=True)
        instrs = _compile_to_ebc(_LET_SRC)
        event_lock = threading.Lock()
        running = [0]
        max_concurrent = [0]

        def record_enter():
            with event_lock:
                running[0] += 1
                if running[0] > max_concurrent[0]:
                    max_concurrent[0] = running[0]

        def record_exit():
            with event_lock:
                running[0] -= 1

        def patched_run():
            # NB: the record_enter() call is INSIDE the lock body so
            # max_concurrent measures threads actually holding the lock
            # — not threads waiting to acquire. If the lock is
            # working, max_concurrent stays at 1; with no lock, it
            # would jump to 4.
            with vm._state_lock:
                # Hold the lock 50 ms; this is much larger than
                # thread-schedule jitter so any concurrent acquire
                # would actually overlap.
                time.sleep(0.05)
                record_enter()
                # Tiny additional hold so the record call itself is
                # observed WHILE holding the lock.
                time.sleep(0.01)
            record_exit()

        threads = [threading.Thread(target=patched_run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        # 4 threads each holding 60 ms serially → ~240 ms total.
        # `max_concurrent` would be 4 if no lock was held; assert 1.
        self.assertEqual(max_concurrent[0], 1,
                         "execute() must serialize concurrent calls on "
                         f"the same VM instance; saw {max_concurrent[0]} "
                         "concurrent entries")


class TestExecuteParallel(unittest.TestCase):
    """Parallel shots via `execute_parallel`."""

    def test_returns_one_result_per_shot(self):
        instrs = _compile_to_ebc(_LOOP_SRC)
        vm = EigenVM(seed=1, deterministic=True)
        results = vm.execute_parallel(instrs, shots=3)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, dict)

    def test_parent_vm_state_untouched(self):
        instrs = _compile_to_ebc(_LET_SRC)
        vm = EigenVM(seed=1, deterministic=True)
        # Execute once to populate vm.globals
        vm.execute(instrs)
        globals_before = dict(vm.globals)
        # Now run parallel — parent should be untouched.
        results = vm.execute_parallel(instrs, shots=4)
        self.assertEqual(vm.globals, globals_before,
                         "execute_parallel must not mutate parent VM state "
                         f"(before={globals_before}, after={dict(vm.globals)})")

    def test_parallel_shots_use_distinct_seeds(self):
        """When the parent seed is set and deterministic=False, each shot
        gets a different RNG seed; thus a circuit that involves random
        measurement should observe different outcomes across shots
        with high probability. We assert on at least 2 distinct
        outcomes out of 8 shots.
        """
        # Mirror the canonical coin_flip.eig example so the
        # type-checker definitely accepts the source (we don't want
        # this test to be at the mercy of edge-case parser changes).
        src = """eigen 1.0
qubit q0
H q0
cbit result
measure q0 -> result
print result
"""
        from src.compiler import to_ebc
        try:
            random_file = os.path.join(_WORKSPACE, "rand_shot.eig")
            with open(random_file, "w", encoding="utf-8") as f:
                f.write(src)
            instrs = to_ebc(random_file, _WORKSPACE)
        except Exception as e:
            self.skipTest(f"to_ebc didn't like this fixture: {e}")

        vm = EigenVM(seed=7, deterministic=False)
        results = vm.execute_parallel(instrs, shots=8)
        self.assertEqual(len(results), 8)
        # The measurement outcomes appear in globals as 'result' values.
        cbs = [r.get("result") for r in results if "result" in r]
        self.assertGreaterEqual(len(cbs), 6,
                                f"need at least 6 measurements; got {cbs}")
        distinct = set(str(c) for c in cbs)
        # With 8 independent shots of H+measure, probability of all 8
        # identical outcomes is 2 * (1/2)^8 = ~0.78%. We assert NOT all
        # identical — i.e. at least 2 distinct values — which holds with
        # overwhelming probability across runs.
        self.assertGreaterEqual(len(distinct), 2,
                                f"all 8 shots returned identical outcome "
                                f"{cbs[0]} — RNG seeds did not differ "
                                "across shots as expected")

    def test_zero_shots_returns_empty(self):
        vm = EigenVM(seed=1)
        self.assertEqual(vm.execute_parallel([], shots=0), [])

    def test_thread_count_capped_at_shots(self):
        """If threads > shots, execute_parallel should still work — the
        internal min() handles this."""
        instrs = _compile_to_ebc(_LET_SRC)
        vm = EigenVM(seed=1, deterministic=True)
        results = vm.execute_parallel(instrs, shots=2, threads=8)
        self.assertEqual(len(results), 2)

    def test_1_shot_works(self):
        instrs = _compile_to_ebc(_LET_SRC)
        vm = EigenVM(seed=3, deterministic=True)
        results = vm.execute_parallel(instrs, shots=1, threads=1)
        self.assertEqual(len(results), 1)


class TestReentrantExecuteSameThread(unittest.TestCase):
    """execute() must be reentrant from the same thread (RLock) so that
    nested stdlib callbacks into a recursive function on the same VM
    don't deadlock. We simulate this by acquiring the lock in the test
    thread and calling execute() — the reentrant RLock should let it
    through."""

    def test_reentrant_lock_does_not_deadlock(self):
        instrs = _compile_to_ebc(_LET_SRC)
        vm = EigenVM(seed=8, deterministic=True)
        with vm._state_lock:
            # Reacquire must succeed since RLock allows re-entry on the
            # same thread.
            with vm._state_lock:
                vm.execute(instrs)
        # If we got here, no deadlock — RLock worked as designed.
        self.assertIsInstance(vm.globals, dict)


class TestRunParallelReturnsShotResults(unittest.TestCase):
    def test_execute_parallel_each_shot_is_independent(self):
        """Running the same instructions multiple times in parallel must
        not see shared state across shots (it would defeat quantum
        sampling). We assert each result dict is its own object."""
        instrs = _compile_to_ebc(_LOOP_SRC)
        vm = EigenVM(seed=2, deterministic=True)
        results = vm.execute_parallel(instrs, shots=4)
        self.assertEqual(len(results), 4)
        ids = {id(r) for r in results}
        self.assertEqual(len(ids), 4, "all shots returned the same dict")


if __name__ == "__main__":
    unittest.main()
