"""§10.3 — Documentation tests."""
import unittest

from src.documentation_pipeline import (
    TutorialStep,
    InteractiveTutorial,
    get_started_tutorial,
    APIEntry,
    extract_api_entries,
    render_api_reference,
    Recipe,
    Cookbook,
    default_cookbook,
    VideoTutorial,
    VideoTutorialCatalogue,
    default_video_catalogue,
    BrowserPlaygroundBuilder,
)


# ---------------------------------------------------------------------------
# Tutorial
# ---------------------------------------------------------------------------

class TestTutorialStep(unittest.TestCase):
    def test_fields_preserved(self):
        s = TutorialStep(title="t", description="d",
                            code="c", expected_output="o",
                            explanation="e")
        self.assertEqual(s.title, "t")
        self.assertEqual(s.code, "c")


class TestInteractiveTutorial(unittest.TestCase):
    def setUp(self):
        self.t = InteractiveTutorial(title="t", steps=[
            TutorialStep(title="Step 1", description="d1"),
            TutorialStep(title="Step 2", description="d2"),
            TutorialStep(title="Step 3", description="d3"),
        ])

    def test_initial_step_is_step_1(self):
        self.assertEqual(self.t.current_step().title, "Step 1")

    def test_next_step_advances_index(self):
        self.assertEqual(self.t.next_step().title, "Step 2")
        self.assertEqual(self.t.current_step().title, "Step 2")
        self.assertEqual(self.t.next_step().title, "Step 3")
        self.assertIsNone(self.t.next_step())  # end of tutorial

    def test_previous_step_decrements(self):
        self.t.next_step()
        self.t.next_step()
        self.assertEqual(self.t.previous_step().title, "Step 2")

    def test_previous_step_at_start_returns_none(self):
        self.assertIsNone(self.t.previous_step())

    def test_reset_returns_to_step_1(self):
        self.t.next_step()
        self.t.reset()
        self.assertEqual(self.t.current_step().title, "Step 1")

    def test_total_steps(self):
        self.assertEqual(self.t.total_steps(), 3)


class TestGetStartedTutorial(unittest.TestCase):
    def test_returns_interactive_tutorial(self):
        t = get_started_tutorial()
        self.assertIsInstance(t, InteractiveTutorial)
        self.assertEqual(t.title, "Get Started with Eigen")

    def test_has_at_least_five_steps(self):
        t = get_started_tutorial()
        self.assertGreaterEqual(t.total_steps(), 5)

    def test_first_step_has_code(self):
        t = get_started_tutorial()
        self.assertTrue(t.current_step().code)


# ---------------------------------------------------------------------------
# API reference generator
# ---------------------------------------------------------------------------

class TestAPIEntry(unittest.TestCase):
    def test_default_optional_fields(self):
        e = APIEntry(name="x", kind="function")
        self.assertEqual(e.parameters, [])
        self.assertEqual(e.examples, [])


class TestExtractApiEntries(unittest.TestCase):
    def test_extract_from_simple_module(self):
        # Define a small module-like object.
        class FakeMod:
            __name__ = "fakemod"
            def public_fn(self, a, b):
                """Public function."""
                return a + b
            def _private_fn(self):
                return None
            class PublicClass:
                """A public class."""
                pass
            class _PrivateClass:
                pass
        out = extract_api_entries(FakeMod)
        names = {e.name for e in out}
        self.assertIn("public_fn", names)
        self.assertIn("PublicClass", names)
        self.assertNotIn("_private_fn", names)
        self.assertNotIn("_PrivateClass", names)


class TestRenderApiReference(unittest.TestCase):
    def test_renders_markdown_header(self):
        entries = [
            APIEntry(name="foo", kind="function",
                        module_path="m", docstring="Docs.",
                        signature="()"),
            APIEntry(name="Bar", kind="class",
                        module_path="m", docstring="A class."),
        ]
        out = render_api_reference(entries)
        self.assertIn("# API Reference", out)
        self.assertIn("## `m`", out)
        self.assertIn("`foo`", out)
        self.assertIn("`Bar`", out)
        # Docstrings appear in output
        self.assertIn("Docs.", out)
        self.assertIn("A class.", out)


# ---------------------------------------------------------------------------
# Cookbook
# ---------------------------------------------------------------------------

class TestRecipe(unittest.TestCase):
    def test_default_tags(self):
        r = Recipe(title="t", problem="p", solution="s")
        self.assertEqual(r.tags, [])


class TestCookbook(unittest.TestCase):
    def test_add_recipe(self):
        c = Cookbook()
        c.add(Recipe(title="r", problem="p", solution="s", tags=["a"]))
        self.assertEqual(len(c.recipes), 1)

    def test_by_tag_filters_correctly(self):
        c = Cookbook()
        c.add(Recipe(title="r1", problem="p", solution="s", tags=["a"]))
        c.add(Recipe(title="r2", problem="p", solution="s", tags=["b"]))
        c.add(Recipe(title="r3", problem="p", solution="s", tags=["a", "b"]))
        tagged_a = c.by_tag("a")
        self.assertEqual(len(tagged_a), 2)

    def test_render_markdown_includes_titles(self):
        c = Cookbook()
        c.add(Recipe(title="My Recipe", problem="p", solution="s"))
        out = c.render_markdown()
        self.assertIn("# Cookbook", out)
        self.assertIn("My Recipe", out)


class TestDefaultCookbook(unittest.TestCase):
    def test_returns_cookbook_with_recipes(self):
        c = default_cookbook()
        self.assertIsInstance(c, Cookbook)
        self.assertGreaterEqual(len(c.recipes), 4)

    def test_has_quantum_recipe(self):
        c = default_cookbook()
        quantum = c.by_tag("quantum")
        self.assertGreaterEqual(len(quantum), 1)


# ---------------------------------------------------------------------------
# Video tutorial catalogue
# ---------------------------------------------------------------------------

class TestVideoTutorialCatalogue(unittest.TestCase):
    def test_add_and_find_returns_entry(self):
        cat = VideoTutorialCatalogue()
        cat.add(VideoTutorial(title="t1", duration_minutes=5,
                                 script_lines=["line1"]))
        self.assertEqual(cat.find("t1").duration_minutes, 5)

    def test_find_missing_returns_none(self):
        cat = VideoTutorialCatalogue()
        self.assertIsNone(cat.find("nope"))

    def test_render_markdown_includes_scripts(self):
        cat = VideoTutorialCatalogue()
        cat.add(VideoTutorial(title="hello", duration_minutes=2,
                                 script_lines=["l1"]))
        out = cat.render_markdown()
        self.assertIn("# Video Tutorial Catalogue", out)
        self.assertIn("hello", out)
        self.assertIn("l1", out)


class TestDefaultVideoCatalogue(unittest.TestCase):
    def test_returns_catalogue_with_one_entry(self):
        cat = default_video_catalogue()
        self.assertIsInstance(cat, VideoTutorialCatalogue)
        self.assertGreaterEqual(len(cat.entries), 1)


# ---------------------------------------------------------------------------
# Browser playground builder
# ---------------------------------------------------------------------------

class TestBrowserPlaygroundBuilder(unittest.TestCase):
    def test_render_includes_html_header(self):
        b = BrowserPlaygroundBuilder()
        out = b.render(initial_code="hello")
        self.assertIn("<!DOCTYPE html>", out)
        self.assertIn("Eigen playground", out)
        self.assertIn("hello", out)

    def test_render_has_run_button(self):
        b = BrowserPlaygroundBuilder(run_button_id="myRun")
        out = b.render()
        self.assertIn('id="myRun"', out)

    def test_escape_initial_code(self):
        b = BrowserPlaygroundBuilder()
        out = b.render(initial_code="<script>alert(1)</script>")
        # The initial code should be HTML-escaped to prevent XSS.
        self.assertNotIn("<script>alert(1)</script>", out)


if __name__ == "__main__":
    unittest.main()
