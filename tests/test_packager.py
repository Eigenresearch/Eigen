"""§9.1 — dedicated tests for the Eigen Packager.

Exercises `src.packager`:
  - parse_toml / write_toml round-trip
  - parse_version / version_satisfies semantics (^, ~, exact, *)
  - EigenPackager init_package / add_dependency / install_dependencies /
    publish_package / search_packages / build_package against a tmp
    workspace root.
"""
import json
import os
import tempfile
import unittest

from src.packager import (
    parse_toml,
    write_toml,
    parse_version,
    version_satisfies,
    EigenPackager,
)


class TestParseToml(unittest.TestCase):
    def test_simple_key_value(self):
        result = parse_toml('name = "foo"\nversion = "1.0.0"')
        self.assertEqual(result, {"name": "foo", "version": "1.0.0"})

    def test_section_creates_nested_dict(self):
        text = "[package]\nname = \"foo\"\nversion = \"1.0.0\""
        result = parse_toml(text)
        self.assertEqual(result, {"package": {"name": "foo",
                                                  "version": "1.0.0"}})

    def test_empty_and_comments_ignored(self):
        text = "# comment line\n\nname = \"x\"\n"
        result = parse_toml(text)
        self.assertEqual(result, {"name": "x"})

    def test_single_quoted_strings(self):
        result = parse_toml("name = 'bar'")
        self.assertEqual(result, {"name": "bar"})

    def test_multiple_sections(self):
        text = "[package]\nname = \"x\"\n\n[dependencies]\nfoo = \"1.0.0\""
        result = parse_toml(text)
        self.assertEqual(result, {
            "package": {"name": "x"},
            "dependencies": {"foo": "1.0.0"},
        })


class TestWriteToml(unittest.TestCase):
    def test_writes_package_section(self):
        text = write_toml({"package": {"name": "foo", "version": "1.0.0"}})
        self.assertIn("[package]", text)
        self.assertIn('name = "foo"', text)
        self.assertIn('version = "1.0.0"', text)

    def test_writes_dependencies_section(self):
        text = write_toml({"dependencies": {"foo": "1.0.0", "bar": "2.0.0"}})
        self.assertIn("[dependencies]", text)
        self.assertIn('foo = "1.0.0"', text)
        self.assertIn('bar = "2.0.0"', text)

    def test_round_trip(self):
        original = {"package": {"name": "foo", "version": "1.0.0"},
                       "dependencies": {"bar": "2.0.0"}}
        text = write_toml(original)
        parsed = parse_toml(text)
        self.assertEqual(parsed, original)


class TestParseVersion(unittest.TestCase):
    def test_three_part_version(self):
        self.assertEqual(parse_version("1.2.3"), (1, 2, 3))

    def test_v_prefix(self):
        self.assertEqual(parse_version("v1.2.3"), (1, 2, 3))

    def test_short_version_padded(self):
        self.assertEqual(parse_version("1.2"), (1, 2, 0))

    def test_invalid_parts_become_zero(self):
        self.assertEqual(parse_version("1.x.y"), (1, 0, 0))

    def test_truncates_extra_parts(self):
        self.assertEqual(parse_version("1.2.3.4"), (1, 2, 3))


class TestVersionSatisfies(unittest.TestCase):
    def test_star_matches_anything(self):
        self.assertTrue(version_satisfies("*", "1.0.0"))
        self.assertTrue(version_satisfies("", "9.9.9"))

    def test_caret_same_major(self):
        self.assertTrue(version_satisfies("^1.2.3", "1.5.0"))
        self.assertTrue(version_satisfies("^1.0.0", "1.99.99"))
        self.assertFalse(version_satisfies("^1.2.3", "2.0.0"))
        self.assertFalse(version_satisfies("^1.2.3", "1.2.2"))

    def test_caret_zero_major_minor_locked(self):
        self.assertTrue(version_satisfies("^0.1.0", "0.1.5"))
        self.assertFalse(version_satisfies("^0.1.0", "0.2.0"))

    def test_caret_zero_major_minor_patch_locked(self):
        self.assertTrue(version_satisfies("^0.0.1", "0.0.1"))
        self.assertFalse(version_satisfies("^0.0.1", "0.0.2"))

    def test_tilde_same_major_minor(self):
        self.assertTrue(version_satisfies("~1.2.3", "1.2.5"))
        self.assertFalse(version_satisfies("~1.2.3", "1.3.0"))

    def test_tilde_short_constraint_just_major(self):
        self.assertTrue(version_satisfies("~1", "1.5.0"))
        self.assertFalse(version_satisfies("~1", "2.0.0"))

    def test_exact_version_matches(self):
        self.assertTrue(version_satisfies("1.2.3", "1.2.3"))
        self.assertFalse(version_satisfies("1.2.3", "1.2.4"))


# ---------------------------------------------------------------------------
# EigenPackager against a tmp workspace root
# ---------------------------------------------------------------------------

class TestEigenPackagerInit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = self.tmp.name
        self.pkg = EigenPackager(self.workspace)

    def tearDown(self):
        self.tmp.cleanup()

    def test_registry_dir_created(self):
        self.assertTrue(os.path.isdir(os.path.join(self.workspace, "registry")))

    def test_init_creates_toml(self):
        ok = self.pkg.init_package(name="hello")
        self.assertTrue(ok)
        self.assertTrue(os.path.isfile(os.path.join(self.workspace,
                                                        "eigen.toml")))

    def test_init_creates_template_main(self):
        self.pkg.init_package(name="hello")
        main_path = os.path.join(self.workspace, "src", "main.eig")
        self.assertTrue(os.path.isfile(main_path))

    def test_init_fails_when_toml_exists(self):
        self.pkg.init_package(name="hello")
        second = self.pkg.init_package(name="other")
        self.assertFalse(second)

    def test_init_uses_workspace_basename_default(self):
        # Create a subdir to control basename
        sub = os.path.join(self.workspace, "myname")
        os.makedirs(sub)
        pkgr = EigenPackager(sub)
        pkgr.init_package()
        toml = os.path.join(sub, "eigen.toml")
        self.assertTrue(os.path.isfile(toml))


class TestEigenPackagerAddDependency(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pkg = EigenPackager(self.tmp.name)
        self.pkg.init_package(name="root")

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_dependency_writes_to_toml(self):
        ok = self.pkg.add_dependency("foo", "1.0.0")
        self.assertTrue(ok)
        with open(os.path.join(self.tmp.name, "eigen.toml")) as f:
            content = f.read()
        self.assertIn('foo = "1.0.0"', content)

    def test_add_dependency_creates_section_if_absent(self):
        ok = self.pkg.add_dependency("bar", "2.5.0")
        self.assertTrue(ok)
        with open(os.path.join(self.tmp.name, "eigen.toml")) as f:
            content = f.read()
        self.assertIn("[dependencies]", content)

    def test_add_dependency_no_toml_fails(self):
        # Create new tmp dir without toml
        with tempfile.TemporaryDirectory() as empty:
            pkgr = EigenPackager(empty)
            self.assertFalse(pkgr.add_dependency("foo", "1.0.0"))


class TestEigenPackagerInstallDependencies(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pkg = EigenPackager(self.tmp.name)
        self.pkg.init_package(name="root")
        self.pkg.add_dependency("foo", "^1.0.0")
        # Pre-populate the registry with a fake tarball
        tar_path = os.path.join(self.tmp.name, "registry", "foo-1.2.0.tar")
        with open(tar_path, "w") as f:
            f.write("FAKE PAYLOAD")

    def tearDown(self):
        self.tmp.cleanup()

    def test_install_writes_lockfile(self):
        self.assertTrue(self.pkg.install_dependencies())
        lock_path = os.path.join(self.tmp.name, "eigen.lock")
        self.assertTrue(os.path.isfile(lock_path))
        with open(lock_path) as f:
            lock = json.load(f)
        self.assertIn("foo", lock)

    def test_install_resolves_to_highest_matching_version(self):
        # Add a second tarball for foo with a different version
        tar2 = os.path.join(self.tmp.name, "registry", "foo-1.5.0.tar")
        with open(tar2, "w") as f:
            f.write("FAKE NEW")
        self.pkg.install_dependencies()
        with open(os.path.join(self.tmp.name, "eigen.lock")) as f:
            lock = json.load(f)
        self.assertEqual(lock["foo"]["version"], "1.5.0")

    def test_install_uses_fallback_when_no_registry_match(self):
        # Use a constraint that doesn't match any tarball
        self.pkg = EigenPackager(self.tmp.name)
        self.pkg.init_package(name="root")
        self.pkg.add_dependency("bar", "^0.5.0")
        self.pkg.install_dependencies()
        with open(os.path.join(self.tmp.name, "eigen.lock")) as f:
            lock = json.load(f)
        self.assertIn("bar", lock)
        # Falls back to constraint base 0.5.0
        self.assertEqual(lock["bar"]["version"], "0.5.0")

    def test_install_uses_lockfile_if_compatible(self):
        self.pkg.install_dependencies()
        # Capture resolved hash
        with open(os.path.join(self.tmp.name, "eigen.lock")) as f:
            first_lock = json.load(f)
        # Run again — should re-use locked version
        self.pkg.install_dependencies()
        with open(os.path.join(self.tmp.name, "eigen.lock")) as f:
            second_lock = json.load(f)
        self.assertEqual(first_lock["foo"]["hash"],
                           second_lock["foo"]["hash"])

    def test_install_creates_packages_manifest(self):
        self.pkg.install_dependencies()
        manifest_path = os.path.join(self.tmp.name, ".eigen_packages",
                                       "foo", "manifest.json")
        self.assertTrue(os.path.isfile(manifest_path))


class TestEigenPackagerPublish(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pkg = EigenPackager(self.tmp.name)
        self.pkg.init_package(name="mylib")

    def tearDown(self):
        self.tmp.cleanup()

    def test_publish_creates_tar_in_registry(self):
        ok = self.pkg.publish_package()
        self.assertTrue(ok)
        tar_path = os.path.join(self.tmp.name, "registry",
                                  "mylib-1.0.0.tar")
        self.assertTrue(os.path.isfile(tar_path))

    def test_publish_no_toml_fails(self):
        with tempfile.TemporaryDirectory() as empty:
            pkgr = EigenPackager(empty)
            self.assertFalse(pkgr.publish_package())


class TestEigenPackagerSearchAndBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pkg = EigenPackager(self.tmp.name)
        self.pkg.init_package(name="root")
        # Publish one so the registry has one package available
        self.pkg.publish_package()

    def tearDown(self):
        self.tmp.cleanup()

    def test_search_returns_at_least_one_match(self):
        import io
        import contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.pkg.search_packages("mylib")
        self.assertIn("mylib", out.getvalue())

    def test_build_invokes_install(self):
        # build_package() calls install_dependencies() internally;
        # we just check the build returns True (no toml absent).
        self.assertTrue(self.pkg.build_package())


if __name__ == "__main__":
    unittest.main()
