"""§11.3 — Community infrastructure envelope tests."""
import unittest

from src.community import (
    ContributionExample,
    render_contribution_guide,
    default_contribution_examples,
    IssueKind,
    IssueTemplate,
    render_issue_template,
    default_issue_templates,
    DiscussionForumLink,
    default_forum_links,
    render_forum_links,
    ExampleEntry,
    ExampleGallery,
    default_gallery,
    PluginManifest,
    PluginLoader,
)


# ---------------------------------------------------------------------------
# Contribution guide
# ---------------------------------------------------------------------------

class TestContributionExample(unittest.TestCase):
    def test_dataclass(self):
        ex = ContributionExample(
            title="t",
            description="d",
            code="print('hi')",
        )
        self.assertEqual(ex.title, "t")
        self.assertEqual(ex.description, "d")
        self.assertEqual(ex.code, "print('hi')")


class TestDefaultContributionExamples(unittest.TestCase):
    def test_returns_at_least_two(self):
        exs = default_contribution_examples()
        self.assertGreaterEqual(len(exs), 2)
        for ex in exs:
            self.assertIsInstance(ex, ContributionExample)
            self.assertTrue(ex.title)
            self.assertTrue(ex.code)

    def test_includes_gate_example(self):
        titles = [e.title.lower() for e in default_contribution_examples()]
        self.assertTrue(any("gate" in t for t in titles))

    def test_includes_lsp_example(self):
        titles = [e.title.lower() for e in default_contribution_examples()]
        self.assertTrue(any("lsp" in t or "completion" in t
                              for t in titles))


class TestRenderContributionGuide(unittest.TestCase):
    def test_uses_project_name_in_header(self):
        md = render_contribution_guide(project_name="EigenLang")
        self.assertIn("Contributing to EigenLang", md)

    def test_lists_prerequisites(self):
        md = render_contribution_guide()
        self.assertIn("Prerequisites", md)
        self.assertIn("Python 3.11+", md)
        self.assertIn("Rust 1.70+", md)

    def test_lists_workflow_steps(self):
        md = render_contribution_guide()
        self.assertIn("Workflow", md)
        self.assertIn("Fork the repository", md)
        self.assertIn("Open a PR", md)

    def test_includes_default_examples(self):
        md = render_contribution_guide()
        self.assertIn("## Examples", md)
        for ex in default_contribution_examples():
            self.assertIn(ex.title, md)
            self.assertIn(ex.code, md)

    def test_custom_examples_rendered(self):
        custom = [ContributionExample(
            title="Custom Step",
            description="Do the thing",
            code="the_thing()",
        )]
        md = render_contribution_guide(custom)
        self.assertIn("Custom Step", md)
        self.assertIn("Do the thing", md)
        self.assertIn("the_thing()", md)


# ---------------------------------------------------------------------------
# Issue templates
# ---------------------------------------------------------------------------

class TestIssueKind(unittest.TestCase):
    def test_four_kinds(self):
        self.assertEqual(len(IssueKind), 4)
        self.assertIn(IssueKind.BUG, IssueKind)
        self.assertIn(IssueKind.FEATURE, IssueKind)
        self.assertIn(IssueKind.DOC, IssueKind)
        self.assertIn(IssueKind.PERFORMANCE, IssueKind)


class TestIssueTemplate(unittest.TestCase):
    def test_dataclass(self):
        t = IssueTemplate(kind=IssueKind.BUG, title="Bug", body="x")
        self.assertEqual(t.kind, IssueKind.BUG)
        self.assertEqual(t.title, "Bug")
        self.assertEqual(t.body, "x")


class TestRenderIssueTemplate(unittest.TestCase):
    def test_yaml_header_has_name_about_title_labels(self):
        t = IssueTemplate(kind=IssueKind.BUG,
                            title="Bug report",
                            body="desc")
        out = render_issue_template(t)
        self.assertIn("name: Bug report", out)
        self.assertIn("about: Bug report", out)
        self.assertIn("labels: bug", out)
        self.assertIn("title:", out)

    def test_body_lines_indented_with_two_spaces(self):
        t = IssueTemplate(kind=IssueKind.DOC,
                            title="Doc",
                            body="line1\nline2")
        out = render_issue_template(t)
        self.assertIn("  - line1", out)
        self.assertIn("  - line2", out)

    def test_feature_label_is_enhancement(self):
        t = IssueTemplate(
            kind=IssueKind.FEATURE,
            title="Feature request",
            body="do this",
        )
        out = render_issue_template(t)
        self.assertIn("labels: enhancement", out)

    def test_doc_label_is_documentation(self):
        t = IssueTemplate(
            kind=IssueKind.DOC,
            title="Documentation issue",
            body="bad doc",
        )
        out = render_issue_template(t)
        self.assertIn("labels: documentation", out)

    def test_performance_label_is_performance(self):
        t = IssueTemplate(
            kind=IssueKind.PERFORMANCE,
            title="Performance issue",
            body="slow",
        )
        out = render_issue_template(t)
        self.assertIn("labels: performance", out)


class TestDefaultIssueTemplates(unittest.TestCase):
    def test_returns_four_templates(self):
        ts = default_issue_templates()
        self.assertEqual(len(ts), 4)

    def test_each_kind_present(self):
        kinds = {t.kind for t in default_issue_templates()}
        self.assertEqual(kinds, {IssueKind.BUG,
                                    IssueKind.FEATURE,
                                    IssueKind.DOC,
                                    IssueKind.PERFORMANCE})

    def test_each_template_has_nonempty_title_and_body(self):
        for t in default_issue_templates():
            self.assertTrue(t.title)
            self.assertTrue(t.body)


# ---------------------------------------------------------------------------
# Forum links
# ---------------------------------------------------------------------------

class TestDiscussionForumLink(unittest.TestCase):
    def test_dataclass(self):
        link = DiscussionForumLink(
            name="Forum",
            url="https://example.com",
            description="A forum",
        )
        self.assertEqual(link.name, "Forum")
        self.assertEqual(link.url, "https://example.com")
        self.assertEqual(link.description, "A forum")


class TestDefaultForumLinks(unittest.TestCase):
    def test_returns_three_links(self):
        links = default_forum_links()
        self.assertEqual(len(links), 3)
        for l in links:
            self.assertTrue(l.name)
            self.assertTrue(l.url.startswith("https://"))

    def test_includes_discussions_and_discord(self):
        names = {l.name for l in default_forum_links()}
        self.assertIn("GitHub Discussions", names)
        self.assertIn("Discord", names)


class TestRenderForumLinks(unittest.TestCase):
    def test_default_header(self):
        out = render_forum_links()
        self.assertIn("# Community forum links", out)
        for l in default_forum_links():
            self.assertIn(l.name, out)
            self.assertIn(l.url, out)

    def test_custom_header(self):
        out = render_forum_links(markdown_header="## Discussion")
        self.assertIn("## Discussion", out)

    def test_custom_links(self):
        custom = [DiscussionForumLink(name="X",
                                         url="https://x.example",
                                         description="D")]
        out = render_forum_links(custom)
        self.assertIn("X", out)
        self.assertIn("https://x.example", out)
        self.assertIn("D", out)


# ---------------------------------------------------------------------------
# Example gallery
# ---------------------------------------------------------------------------

class TestExampleEntry(unittest.TestCase):
    def test_default_tags_empty(self):
        e = ExampleEntry(title="t", description="d", code="c")
        self.assertEqual(e.tags, [])


class TestExampleGallery(unittest.TestCase):
    def test_add_appends_entry(self):
        g = ExampleGallery()
        e1 = ExampleEntry(title="a", description="b", code="c")
        g.add(e1)
        self.assertEqual(len(g.entries), 1)

    def test_by_tag_filter(self):
        g = ExampleGallery()
        e1 = ExampleEntry(title="a", description="b", code="c", tags=["x"])
        e2 = ExampleEntry(title="y", description="z", code="w",
                            tags=["x", "y"])
        e3 = ExampleEntry(title="p", description="q", code="r", tags=["z"])
        g.add(e1)
        g.add(e2)
        g.add(e3)
        self.assertEqual(len(g.by_tag("x")), 2)
        self.assertEqual(len(g.by_tag("y")), 1)
        self.assertEqual(len(g.by_tag("z")), 1)
        self.assertEqual(g.by_tag("y")[0].title, "y")

    def test_by_tag_returns_empty_for_missing_tag(self):
        g = ExampleGallery()
        g.add(ExampleEntry(title="a", description="b", code="c"))
        self.assertEqual(g.by_tag("missing"), [])

    def test_render_markdown_includes_header(self):
        g = ExampleGallery()
        g.add(ExampleEntry(title="Hello", description="greet", code="x"))
        md = g.render_markdown()
        self.assertIn("# Example gallery", md)
        self.assertIn("## Hello", md)
        self.assertIn("greet", md)
        self.assertIn("```eig", md)
        self.assertIn("x", md)

    def test_render_markdown_lists_tags_when_present(self):
        g = ExampleGallery()
        g.add(ExampleEntry(title="t", description="d", code="c",
                             tags=["quantum", "entangle"]))
        md = g.render_markdown()
        self.assertIn("Tags: quantum, entangle", md)


class TestDefaultGallery(unittest.TestCase):
    def test_has_three_entries(self):
        g = default_gallery()
        self.assertEqual(len(g.entries), 3)

    def test_includes_hello_bell_fizzbuzz(self):
        titles = [e.title for e in default_gallery().entries]
        self.assertIn("Hello, Eigen", titles)
        self.assertIn("Bell state", titles)
        self.assertIn("FizzBuzz", titles)

    def test_bell_state_has_quantum_tag(self):
        bell = [e for e in default_gallery().entries
                if e.title == "Bell state"][0]
        self.assertIn("quantum", bell.tags)
        self.assertIn("entanglement", bell.tags)


# ---------------------------------------------------------------------------
# Plugin / extension API
# ---------------------------------------------------------------------------

class TestPluginManifest(unittest.TestCase):
    def test_dataclass(self):
        m = PluginManifest(
            name="p",
            api_version="1.0",
            entry_point="p.main",
        )
        self.assertEqual(m.name, "p")
        self.assertEqual(m.api_version, "1.0")
        self.assertEqual(m.entry_point, "p.main")
        self.assertEqual(m.dependencies, [])
        self.assertEqual(m.description, "")

    def test_with_dependencies(self):
        m = PluginManifest(
            name="p",
            api_version="1.0",
            entry_point="p.main",
            description="d",
            dependencies=["a", "b"],
        )
        self.assertEqual(m.description, "d")
        self.assertEqual(m.dependencies, ["a", "b"])


class TestPluginLoader(unittest.TestCase):
    def test_register_and_load(self):
        loader = PluginLoader()
        m = PluginManifest(name="p",
                             api_version="1.0",
                             entry_point="p.main")
        loader.register(m)
        loaded = loader.load("p")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "p")
        self.assertEqual(loaded.entry_point, "p.main")

    def test_load_unknown_returns_none(self):
        loader = PluginLoader()
        self.assertIsNone(loader.load("nope"))

    def test_list_returns_sorted_names(self):
        loader = PluginLoader()
        loader.register(PluginManifest(name="b", api_version="1.0",
                                          entry_point="b.main"))
        loader.register(PluginManifest(name="a", api_version="1.0",
                                          entry_point="a.main"))
        self.assertEqual(loader.list(), ["a", "b"])

    def test_list_empty_when_no_plugins(self):
        loader = PluginLoader()
        self.assertEqual(loader.list(), [])

    def test_register_rejects_unsupported_api_version(self):
        loader = PluginLoader()
        with self.assertRaises(ValueError):
            loader.register(PluginManifest(name="p",
                                              api_version="2.0",
                                              entry_point="p.main"))

    def test_register_rejects_api_version_3(self):
        loader = PluginLoader()
        with self.assertRaises(ValueError):
            loader.register(PluginManifest(name="p",
                                              api_version="3.0",
                                              entry_point="p.main"))

    def test_register_replace_overwrites_existing_name(self):
        loader = PluginLoader()
        loader.register(PluginManifest(name="p",
                                          api_version="1.0",
                                          entry_point="p.main"))
        loader.register(PluginManifest(name="p",
                                          api_version="1.0",
                                          entry_point="p.newer"))
        self.assertEqual(loader.load("p").entry_point, "p.newer")


if __name__ == "__main__":
    unittest.main()
