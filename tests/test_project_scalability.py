"""§8.3 — Project scalability (workspace/DAG/dep-graph viz) tests."""
import os
import tempfile
import unittest

from src.project_scalability import (
    Package,
    WorkspaceManifest,
    PackageGraph,
    Workspace,
    workspace_from_manifest,
    parse_workspace_toml,
    parse_workspace_file,
    CircularDependencyError,
    render_dot_dag,
    render_ascii_layers,
)


# ---------------------------------------------------------------------------
# Package / WorkspaceManifest dataclasses
# ---------------------------------------------------------------------------

class TestPackage(unittest.TestCase):
    def test_default_construction(self):
        p = Package(name="x")
        self.assertEqual(p.name, "x")
        self.assertEqual(p.path, "")
        self.assertEqual(p.dependencies, [])
        self.assertEqual(p.description, "")

    def test_full_construction(self):
        p = Package(name="y",
                       path="pkgs/y",
                       dependencies=["a", "b"],
                       description="Y pkg")
        self.assertEqual(p.name, "y")
        self.assertEqual(p.path, "pkgs/y")
        self.assertEqual(p.dependencies, ["a", "b"])


class TestWorkspaceManifestFromDict(unittest.TestCase):
    def test_flat_form_takes_keys_directly(self):
        m = WorkspaceManifest.from_dict({
            "name": "ws",
            "members": ["p1", "p2"],
            "shared_dependencies": ["numpy"],
        })
        self.assertEqual(m.name, "ws")
        self.assertEqual(m.members, ["p1", "p2"])
        self.assertEqual(m.shared_dependencies, ["numpy"])

    def test_section_form_picks_workspace_key(self):
        # When a "workspace" key exists, prefer it.
        m = WorkspaceManifest.from_dict({
            "workspace": {"name": "ws_real",
                                "members": ["a", "b"]},
            "name": "ignored_global",
        })
        self.assertEqual(m.name, "ws_real")
        self.assertEqual(m.members, ["a", "b"])

    def test_string_list_parsed_via_parse_list(self):
        # Simulate TOML-pretty string lists: members=["a","b"]
        m = WorkspaceManifest.from_dict({
            "name": "ws",
            "members": '["a", "b"]',
            "shared_dependencies": "c, d",
        })
        self.assertEqual(m.members, ["a", "b"])
        self.assertEqual(m.shared_dependencies, ["c", "d"])

    def test_defaults_for_missing_fields(self):
        m = WorkspaceManifest.from_dict({})
        self.assertEqual(m.name, "")
        self.assertEqual(m.members, [])
        self.assertEqual(m.shared_dependencies, [])
        self.assertEqual(m.description, "")


class TestParseWorkspaceTOML(unittest.TestCase):
    def test_simple_workspace_toml_parses(self):
        toml = """
[workspace]
name = "myws"
description = "test"
members = ["a", "b", "c"]
"""
        m = parse_workspace_toml(toml)
        self.assertEqual(m.name, "myws")
        self.assertEqual(m.description, "test")
        self.assertEqual(m.members, ["a", "b", "c"])

    def test_skips_blank_and_comment_lines(self):
        toml = """
# Top-level comment

[workspace]
name = "x"
# inline comment below
members = ["a"]
"""
        m = parse_workspace_toml(toml)
        self.assertEqual(m.name, "x")
        self.assertEqual(m.members, ["a"])

    def test_handles_boolean_values(self):
        toml = """
[workspace]
name = "withbools"
some_bool = true
"""
        # Just verify it doesn't crash; some_bool isn't read by WorkspaceManifest
        m = parse_workspace_toml(toml)
        self.assertEqual(m.name, "withbools")


class TestParseWorkspaceFile(unittest.TestCase):
    def test_reads_filesystem_path(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toml", delete=False,
                encoding="utf-8") as f:
            f.write("[workspace]\nname = \"wsfile\"\nmembers = [\"a\"]\n")
            path = f.name
        try:
            m = parse_workspace_file(path)
            self.assertEqual(m.name, "wsfile")
            self.assertEqual(m.members, ["a"])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# PackageGraph
# ---------------------------------------------------------------------------

class TestPackageGraphBasics(unittest.TestCase):
    def test_add_and_query_package(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        self.assertEqual(g.names(), ["a"])
        self.assertEqual(g.package("a").name, "a")
        self.assertIsNone(g.package("nonexistent"))
        self.assertEqual(g.dependencies_of("a"), set())

    def test_add_package_with_dependencies(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b", dependencies=["a"]))
        self.assertEqual(g.dependencies_of("b"), {"a"})

    def test_add_dependency_extends_edges(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b"))
        g.add_dependency("b", "a")
        self.assertEqual(g.dependencies_of("b"), {"a"})

    def test_duplicate_package_raises(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        with self.assertRaises(ValueError):
            g.add_package(Package(name="a"))

    def test_empty_name_raises(self):
        g = PackageGraph()
        with self.assertRaises(ValueError):
            g.add_package(Package(name=""))

    def test_add_dependency_unknown_package_raises(self):
        g = PackageGraph()
        with self.assertRaises(ValueError):
            g.add_dependency("ghost", "a")


class TestPackageGraphTopologicalOrder(unittest.TestCase):
    def test_simple_chain(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b", dependencies=["a"]))
        g.add_package(Package(name="c", dependencies=["b"]))
        self.assertEqual(g.topological_order(), ["a", "b", "c"])

    def test_diamond_dependency(self):
        # d depends on b, c which both depend on a
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b", dependencies=["a"]))
        g.add_package(Package(name="c", dependencies=["a"]))
        g.add_package(Package(name="d", dependencies=["b", "c"]))
        order = g.topological_order()
        # a must come first; d must come last
        self.assertEqual(order[0], "a")
        self.assertEqual(order[-1], "d")
        # b and c in any allowed order
        self.assertEqual(set(order[1:3]), {"b", "c"})

    def test_alphabetical_tiebreak(self):
        g = PackageGraph()
        for n in ["alpha", "beta", "gamma", "delta"]:
            g.add_package(Package(name=n))
        # All independent — should be alphabetical
        self.assertEqual(g.topological_order(),
                         ["alpha", "beta", "delta", "gamma"])

    def test_external_dependency_included(self):
        # `b` depends on `a` AND on `external` (which is NOT a package)
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b", dependencies=["external",
                                                          "a"]))
        # `external` is unknown; topo order is still computed
        order = g.topological_order()
        self.assertEqual(order, ["a", "b"])

    def test_cycle_raises(self):
        g = PackageGraph()
        g.add_package(Package(name="x", dependencies=["y"]))
        g.add_package(Package(name="y", dependencies=["x"]))
        with self.assertRaises(CircularDependencyError):
            g.topological_order()


class TestPackageGraphLayers(unittest.TestCase):
    def test_simple_chain_layers(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b", dependencies=["a"]))
        g.add_package(Package(name="c", dependencies=["b"]))
        self.assertEqual(g.layers(), [["a"], ["b"], ["c"]])

    def test_diamond_layers(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b", dependencies=["a"]))
        g.add_package(Package(name="c", dependencies=["a"]))
        g.add_package(Package(name="d",
                                 dependencies=["b", "c"]))
        layers = g.layers()
        # Layer 0: just `a`; Layer 1: both `b` and `c`; Layer 2: `d`
        self.assertEqual(layers[0], ["a"])
        self.assertEqual(set(layers[1]), {"b", "c"})
        self.assertEqual(layers[2], ["d"])

    def test_independent_packages_single_layer(self):
        g = PackageGraph()
        for n in ["a", "b", "c"]:
            g.add_package(Package(name=n))
        self.assertEqual(g.layers(), [["a", "b", "c"]])

    def test_cycle_raises_in_layers(self):
        g = PackageGraph()
        g.add_package(Package(name="x", dependencies=["y"]))
        g.add_package(Package(name="y", dependencies=["x"]))
        with self.assertRaises(CircularDependencyError):
            g.layers()


class TestPackageGraphDetectCycles(unittest.TestCase):
    def test_returns_empty_for_acyclic_graph(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b", dependencies=["a"]))
        self.assertEqual(g.detect_cycles(), [])

    def test_returns_cycle_nodes_for_cyclic_graph(self):
        g = PackageGraph()
        g.add_package(Package(name="x", dependencies=["y"]))
        g.add_package(Package(name="y", dependencies=["x"]))
        cycles = g.detect_cycles()
        self.assertEqual(sorted(cycles), ["x", "y"])


# ---------------------------------------------------------------------------
# Workspace + workspace_from_manifest
# ---------------------------------------------------------------------------

class TestWorkspace(unittest.TestCase):
    def test_add_package_and_count(self):
        ws = Workspace(WorkspaceManifest(name="x"))
        ws.add_package(Package(name="a"))
        ws.add_package(Package(name="b"))
        self.assertEqual(ws.package_count(), 2)

    def test_topological_order_delegates_to_graph(self):
        ws = Workspace(WorkspaceManifest(name="x"))
        ws.add_package(Package(name="a"))
        ws.add_package(Package(name="b", dependencies=["a"]))
        self.assertEqual(ws.topological_order(), ["a", "b"])

    def test_build_layers_delegates(self):
        ws = Workspace(WorkspaceManifest(name="x"))
        ws.add_package(Package(name="a"))
        ws.add_package(Package(name="b"))
        # Both independent
        self.assertEqual(ws.build_layers(), [["a", "b"]])


class TestWorkspaceFromManifest(unittest.TestCase):
    def test_creates_workspace_with_manifest_members(self):
        m = WorkspaceManifest(name="ws", members=["p1", "p2", "p3"])
        ws = workspace_from_manifest(m)
        self.assertEqual(ws.package_count(), 3)
        self.assertEqual(ws.topological_order(), ["p1", "p2", "p3"])

    def test_custom_loader_used_for_packages(self):
        m = WorkspaceManifest(name="ws", members=["alpha"])


        def loader(name):
            return Package(name=name, dependencies=["base"])

        ws = workspace_from_manifest(m, package_loader=loader)
        self.assertEqual(ws.graph.package("alpha").dependencies,
                           ["base"])

    def test_loader_returns_none_falls_back_to_default(self):
        m = WorkspaceManifest(name="ws", members=["x"])

        def loader(name):
            return None

        ws = workspace_from_manifest(m, package_loader=loader)
        self.assertIsNotNone(ws.graph.package("x"))
        self.assertEqual(ws.graph.package("x").dependencies, [])


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class TestRenderDotDag(unittest.TestCase):
    def test_empty_graph_has_valid_dot_syntax(self):
        dot = render_dot_dag(PackageGraph())
        self.assertIn("digraph packages", dot)
        self.assertIn("rankdir=LR;", dot)
        self.assertIn("}", dot)

    def test_lists_all_node_names(self):
        g = PackageGraph()
        g.add_package(Package(name="alpha"))
        g.add_package(Package(name="beta", dependencies=["alpha"]))
        dot = render_dot_dag(g)
        self.assertIn('"alpha"', dot)
        self.assertIn('"beta"', dot)
        self.assertIn('"beta" -> "alpha"', dot)

    def test_external_dependency_rendered_dashed(self):
        g = PackageGraph()
        g.add_package(Package(name="main",
                                 dependencies=["external_pkg"]))
        dot = render_dot_dag(g)
        self.assertIn('"main" -> "external_pkg" [style=dashed]',
                       dot)


class TestRenderAsciiLayers(unittest.TestCase):
    def test_renders_chain_with_layer_indices(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b", dependencies=["a"]))
        ascii_view = render_ascii_layers(g)
        self.assertIn("Layer 0: a", ascii_view)
        self.assertIn("Layer 1: b", ascii_view)

    def test_parallel_layer_in_one_line(self):
        g = PackageGraph()
        g.add_package(Package(name="a"))
        g.add_package(Package(name="b"))
        ascii_view = render_ascii_layers(g)
        self.assertIn("Layer 0: a b", ascii_view)


if __name__ == "__main__":
    unittest.main()
