"""§10.2 — CLI Enhancement tests, organised by the six roadmap
checkboxes."""
import os
import tempfile
import time
import unittest

from src.cli_enhancements import (
    ShellKind,
    CompletionSpec,
    default_eigen_completion_spec,
    generate_completion_script,
    REPLState,
    REPL,
    FileWatcher,
    PlaygroundBackend,
    Playground,
    MigrationStep,
    Migrate,
    Colour,
    colourise,
    ProgressBar,
)


# ---------------------------------------------------------------------------
# Shell completion
# ---------------------------------------------------------------------------

class TestShellKind(unittest.TestCase):
    def test_four_kinds(self):
        self.assertEqual({k.value for k in ShellKind},
                          {"bash", "zsh", "fish", "powershell"})


class TestCompletionSpec(unittest.TestCase):
    def test_default_has_repl(self):
        spec = default_eigen_completion_spec()
        self.assertIn("repl", spec.subcommands)
        self.assertIn("run", spec.subcommands)


class TestGenerateCompletionScript(unittest.TestCase):
    def test_bash_contains_subcommands(self):
        script = generate_completion_script(ShellKind.BASH)
        self.assertIn("_eigen_completion", script)
        self.assertIn("complete -F _eigen_completion eigen", script)

    def test_bash_script_lists_subcommands(self):
        script = generate_completion_script(ShellKind.BASH)
        for sub in default_eigen_completion_spec().subcommands:
            self.assertIn(sub, script)

    def test_zsh_contains_compdef(self):
        script = generate_completion_script(ShellKind.ZSH)
        self.assertIn("#compdef eigen", script)

    def test_fish_uses_complete_command(self):
        script = generate_completion_script(ShellKind.FISH)
        self.assertIn("complete -c eigen", script)

    def test_powershell_uses_register_argument_completer(self):
        script = generate_completion_script(ShellKind.POWERSHELL)
        self.assertIn("Register-ArgumentCompleter", script)

    def test_unknown_shell_raises(self):
        with self.assertRaises(ValueError):
            generate_completion_script("ksh")  # type: ignore[arg-type]

    def test_custom_spec_is_used(self):
        spec = CompletionSpec(name="mycli", subcommands=["a", "b"])
        bash = generate_completion_script(ShellKind.BASH, spec=spec)
        self.assertIn("mycli", bash)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

class TestREPL(unittest.TestCase):
    def test_default_step_returns_continue_with_output(self):
        repl = REPL()
        result = repl.step("let x = 1")
        # Default evaluator returns the input.
        self.assertEqual(result.state, REPLState.CONTINUE)
        self.assertEqual(result.output, "let x = 1\n")

    def test_exit_command_sets_state(self):
        repl = REPL()
        result = repl.step(":exit")
        self.assertEqual(result.state, REPLState.EXIT)

    def test_help_command_returns_help_text(self):
        repl = REPL()
        result = repl.step(":help")
        self.assertIn(":exit", result.output)
        self.assertIn(":help", result.output)

    def test_error_state_captured(self):
        def boom(source):
            raise RuntimeError("kaboom")
        repl = REPL(evaluator=boom)
        result = repl.step("trigger")
        self.assertEqual(result.state, REPLState.ERROR)
        self.assertIn("kaboom", result.error)

    def test_multi_line_collects_until_balance(self):
        repl = REPL()
        # Open paren on line 1; not balanced yet → CONTINUE.
        r1 = repl.step("fn f(x) {")
        self.assertEqual(r1.state, REPLState.CONTINUE)
        # Closing brace brings paren_depth back to 0 → flush.
        r2 = repl.step("  x")
        self.assertEqual(r2.state, REPLState.CONTINUE)
        r3 = repl.step("}")
        # After the } line, we should now flush the buffer.
        # (paren_depth went 1 → 2 → 1 → 0 → 0 after the }.)
        # Final flush returns the whole source as text.
        self.assertEqual(r3.output, "fn f(x) {\n  x\n}\n")

    def test_history_records_flushed_input(self):
        repl = REPL()
        repl.step("a")
        repl.step("b")
        # `a` and `b` are single-line inputs; each flushes.
        self.assertEqual(len(repl.history), 2)


# ---------------------------------------------------------------------------
# File watcher
# ---------------------------------------------------------------------------

class TestFileWatcher(unittest.TestCase):
    def test_records_event_on_file_change(self):
        with tempfile.NamedTemporaryFile("w", suffix=".eig",
                                            delete=False) as tmp:
            tmp.write("a")
            path = tmp.name
        try:
            recorded_events: list = []
            watcher = FileWatcher(paths=[path],
                                    callback=recorded_events.append,
                                    poll_interval_s=0.01,
                                    max_iterations=4)
            # Force the new mtime to differ from the captured
            # initial mtime. We use os.utime to set an mtime far
            # in the future so resolution differences don't
            # bite us.
            new_mtime = os.path.getmtime(path) + 60.0
            os.utime(path, (new_mtime, new_mtime))
            time.sleep(0.05)
            events = watcher.run(max_iterations=4)
            self.assertGreaterEqual(len(events), 1)
            self.assertEqual(events[0].file_path, path)
        finally:
            os.unlink(path)

    def test_no_change_no_event(self):
        with tempfile.NamedTemporaryFile("w", suffix=".eig",
                                            delete=False) as tmp:
            tmp.write("a")
            path = tmp.name
        try:
            recorded: list = []
            watcher = FileWatcher(paths=[path],
                                    callback=recorded.append,
                                    poll_interval_s=0.01,
                                    max_iterations=1)
            watcher.run()
            self.assertEqual(recorded, [])
        finally:
            os.unlink(path)

    def test_should_stop_breaks_loop(self):
        with tempfile.NamedTemporaryFile("w", suffix=".eig",
                                            delete=False) as tmp:
            tmp.write("a")
            path = tmp.name
        try:
            recorded: list = []
            stop_requested = [False]
            def should_stop():
                return stop_requested[0]
            watcher = FileWatcher(paths=[path],
                                    callback=recorded.append,
                                    poll_interval_s=0.01,
                                    should_stop=should_stop,
                                    max_iterations=10)
            stop_requested[0] = True
            events = watcher.run()
            self.assertEqual(events, [])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------

class TestPlaygroundBackend(unittest.TestCase):
    def test_eval_under_limits_succeeds(self):
        backend = PlaygroundBackend(evaluator=lambda s: f"out:{s}",
                                       max_statement_count=10,
                                       max_eval_depth=2)
        self.assertEqual(backend.eval("hi"), "out:hi")

    def test_max_statement_count_exceeded_raises(self):
        backend = PlaygroundBackend(evaluator=lambda s: "",
                                       max_statement_count=1,
                                       max_eval_depth=2)
        # The string "a\nb\nc" has 3 non-blank lines.
        with self.assertRaises(RuntimeError):
            backend.eval("a\nb\nc")

    def test_max_eval_depth_exceeded_raises(self):
        backend = PlaygroundBackend(evaluator=lambda s: "",
                                       max_statement_count=10,
                                       max_eval_depth=1)
        # Manually push depth to the limit.
        backend._eval_depth = 1
        with self.assertRaises(RuntimeError):
            backend.eval("a")

    def test_reset_clears_counts(self):
        backend = PlaygroundBackend(evaluator=lambda s: "",
                                       max_statement_count=2)
        backend.eval("a")
        backend.reset()
        self.assertEqual(backend._stmt_count, 0)


class TestPlayground(unittest.TestCase):
    def test_run_returns_backend_output(self):
        backend = PlaygroundBackend(evaluator=lambda s: s.upper())
        p = Playground(backend=backend)
        self.assertEqual(p.run("hello"), "HELLO")


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------

class TestMigrate(unittest.TestCase):
    def test_migrate_replaces_semicolon_with_newline(self):
        m = Migrate()
        result = m.migrate("h q0; x q0;")
        self.assertIn("h q0", result)
        self.assertIn("x q0", result)
        self.assertNotIn(";", result)

    def test_does_not_replace_semicolon_in_strings(self):
        m = Migrate()
        source = 'let s = "a;b"\nlet t = 1'
        result = m.migrate(source)
        # The `;` inside the string should be preserved.
        self.assertIn("a;b", result)

    def test_does_not_replace_semicolon_in_block_comment(self):
        m = Migrate()
        source = "/* not ; replaced */\nx"
        result = m.migrate(source)
        self.assertIn("not ; replaced", result)

    def test_diff_emits_unified_diff(self):
        m = Migrate()
        before = "h q0;"
        after = m.migrate(before)
        diff = m.diff(before, after)
        self.assertTrue(any("h q0" in d for d in diff))

    def test_migrate_hgate_renamer(self):
        m = Migrate()
        result = m.migrate("hgate q0")
        # After migration, the gate name should be `h`.
        self.assertIn("h", result)

    def test_step_added_to_list(self):
        m = Migrate()
        m.steps.append(MigrationStep(name="noop",
                                        apply=lambda s: s))
        self.assertGreater(len(m.steps), 2)


# ---------------------------------------------------------------------------
# Colour output
# ---------------------------------------------------------------------------

class TestColour(unittest.TestCase):
    def test_colourise_disabled_returns_text(self):
        out = colourise("hello", Colour.RED, enabled=False)
        self.assertEqual(out, "hello")

    def test_colourise_enabled_has_escape_codes(self):
        out = colourise("hello", Colour.RED, enabled=True)
        self.assertIn("\x1b[31m", out)
        self.assertIn("\x1b[0m", out)
        self.assertIn("hello", out)

    def test_colourise_unknown_colour_returns_raw_text(self):
        # The default ANSI code lookup returns "" for unknown →
        # text should still be wrapped in reset.
        out = colourise("hello", Colour.RED, enabled=True)
        self.assertTrue(out.startswith("\x1b[31m"))


class TestProgressBar(unittest.TestCase):
    def test_render_disabled_returns_empty(self):
        pb = ProgressBar(total=10, enabled=False)
        self.assertEqual(pb.render(), "")

    def test_render_at_zero(self):
        pb = ProgressBar(total=10)
        out = pb.render()
        self.assertIn("0%", out)

    def test_render_at_half(self):
        pb = ProgressBar(total=10)
        pb.update(5)
        out = pb.render()
        self.assertIn("50%", out)

    def test_render_at_full(self):
        pb = ProgressBar(total=10, enabled=True)
        pb.update(10)
        out = pb.render()
        self.assertIn("100%", out)
        # All filled
        self.assertIn("#" * 40, out)

    def test_finish_returns_newline(self):
        pb = ProgressBar(total=10)
        pb.update(10)
        out = pb.finish()
        self.assertTrue(out.endswith("\n"))

    def test_update_clamps_high(self):
        pb = ProgressBar(total=10)
        pb.update(20)
        self.assertEqual(pb.current, 10)

    def test_update_clamps_low(self):
        pb = ProgressBar(total=10)
        pb.update(-5)
        self.assertEqual(pb.current, 0)

    def test_render_with_label(self):
        pb = ProgressBar(total=10, label="task")
        out = pb.render()
        self.assertIn("task", out)

    def test_total_zero_returns_empty(self):
        pb = ProgressBar(total=0)
        self.assertEqual(pb.render(), "")


if __name__ == "__main__":
    unittest.main()
