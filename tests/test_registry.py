"""
P3 §11.1 — Public Package Registry (surface) tests.

Covers `src/registry.PackageRegistry` end-to-end:
  * add/get/list_versions
  * semver `resolve` across `^`, `~`, wildcard, exact
  * substring `search`
  * `verify` sha256 match + mismatch + signature (HMAC-SHA256)
  * `DependencyConflict` from `resolve_all`
  * `resolve_all` flat closure walking transitive deps
  * vulnerability `scan`
  * persistence (new instance loads existing index.json)
  * `remove`
  * `EigenPackager.publish_package` hook populates index
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from src.packager import EigenPackager
from src.registry import (
    DependencyConflict,
    PackageMetadata,
    PackageRegistry,
)


class TestPackageRegistry(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="eigen_reg_test_")
        self.secret = b"supersecret-key"
        self.reg = PackageRegistry(self.root, secret=self.secret,
                                   clock=lambda: 1700000000)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _add(self, name: str, version: str, deps=None):
        return self.reg.add(
            name, version,
            description=f"{name} v{version}",
            dependencies=deps or {},
            tarball_bytes=f"CONTENT {name} {version}".encode("utf-8"),
        )

    # --------------------------------------------------- basics

    def test_add_and_get(self):
        md = self._add("foo", "1.2.3")
        self.assertIsInstance(md, PackageMetadata)
        self.assertEqual(md.sha256, self.reg.get("foo", "1.2.3").sha256)
        self.assertEqual(md.published_at, 1700000000)
        self.assertEqual(self.reg.get("nope", "0.0.0"), None)
        self.assertIn(("foo", "1.2.3"), self.reg)
        self.assertEqual(len(self.reg), 1)

    def test_list_versions_sorted(self):
        self._add("foo", "1.0.0")
        self._add("foo", "0.9.9")
        self._add("foo", "2.0.0")
        vs = [m.version for m in self.reg.list_versions("foo")]
        self.assertEqual(vs, ["0.9.9", "1.0.0", "2.0.0"])

    def test_add_idempotent_replaces(self):
        md1 = self._add("foo", "1.0.0")
        md2 = self._add("foo", "1.0.0")
        self.assertEqual(len(self.reg), 1)
        # SHA-256 is content-deterministic: identical content yields
        # identical sha (so md1.sha256 == md2.sha256). The point of
        # re-add is that the metadata is replaced atomically, leaving
        # only one entry per (name, version) — which we asserted above.
        self.assertEqual(md1.sha256, md2.sha256)
        # Add with different content → SHA differs, count still 1.
        md3 = self.reg.add("foo", "1.0.0",
                           tarball_bytes=b"Different content")
        self.assertEqual(len(self.reg), 1)
        self.assertNotEqual(md3.sha256, md1.sha256)

    def test_add_requires_tarball(self):
        with self.assertRaises(ValueError):
            self.reg.add("foo", "1.0.0")

    def test_add_with_tarball_path(self):
        path = os.path.join(self.root, "external.tar")
        with open(path, "wb") as f:
            f.write(b"hello-world-payload")
        md = self.reg.add("ext", "0.1.0", tarball_path=path)
        self.assertTrue(md.sha256)
        self.assertTrue(os.path.isfile(
            os.path.join(self.root, md.tarball_relpath)))

    # ----------------------------------------------------- resolve

    def test_resolve_caret_picks_highest_compatible(self):
        self._add("foo", "1.0.0")
        self._add("foo", "1.2.0")
        self._add("foo", "2.0.0")
        # ^1.0.0 should pick 1.2.0 (NOT 2.0.0)
        self.assertEqual(self.reg.resolve("foo", "^1.0.0").version, "1.2.0")

    def test_resolve_tilde(self):
        self._add("foo", "1.2.0")
        self._add("foo", "1.2.5")
        self._add("foo", "1.3.0")
        self.assertEqual(self.reg.resolve("foo", "~1.2.0").version, "1.2.5")
        # ~1.2 also picks 1.2.5
        self.assertEqual(self.reg.resolve("foo", "~1.2").version, "1.2.5")

    def test_resolve_exact(self):
        self._add("foo", "1.2.3")
        self.assertEqual(self.reg.resolve("foo", "1.2.3").version, "1.2.3")
        self.assertIsNone(self.reg.resolve("foo", "9.9.9"))

    def test_resolve_star(self):
        self._add("foo", "1.0.0")
        self._add("foo", "2.0.0")
        self.assertEqual(self.reg.resolve("foo", "*").version, "2.0.0")
        self.assertEqual(self.reg.resolve("foo", "").version, "2.0.0")

    def test_resolve_no_candidates(self):
        self.assertIsNone(self.reg.resolve("missing", "^1.0.0"))

    # ----------------------------------------------------- search

    def test_search_substring(self):
        self._add("foo", "1.0.0")
        self._add("foobar", "1.0.0")
        self._add("quark", "1.0.0")  # not matching 'foo'
        results = self.reg.search("foo")
        names = sorted({m.name for m in results})
        self.assertEqual(names, ["foo", "foobar"])

    def test_search_limit(self):
        for v in ("1.0", "2.0", "3.0"):
            self._add("foo", v)
        results = self.reg.search("foo", limit=2)
        self.assertEqual(len(results), 2)

    def test_search_empty_query(self):
        self._add("foo", "1.0.0")
        self.assertEqual(self.reg.search(""), [])

    # ------------------------------------------------- signature & verify

    def test_sign_and_verify_signature(self):
        md = self._add("signed", "1.0.0")
        sig = self.reg.sign(md, secret=self.secret)
        self.assertTrue(self.reg.verify_signature(
            dataclasses_replace(md, signature=sig), secret=self.secret))

    def test_verify_signature_rejects_wrong_secret(self):
        md = self._add("signed", "1.0.0")
        sig = self.reg.sign(md, secret=self.secret)
        bad = dataclasses_replace(md, signature=sig)
        self.assertFalse(self.reg.verify_signature(bad, secret=b"other"))

    def test_add_with_sign(self):
        md = self.reg.add(
            "foo", "1.0.0",
            tarball_bytes=b"x",
            sign=True,
        )
        # `add(sign=True)` populates the signature field.
        self.assertTrue(md.signature)
        self.assertTrue(self.reg.verify_signature(md, secret=self.secret))

    def test_verify_checksum_match(self):
        self._add("foo", "1.0.0")
        self.assertTrue(self.reg.verify("foo", "1.0.0"))

    def test_verify_checksum_mismatch_after_tamper(self):
        md = self._add("foo", "1.0.0")
        # Mutate the on-disk tarball so its real checksum differs from
        # the recorded md.sha256.
        full = os.path.join(self.root, md.tarball_relpath)
        with open(full, "wb") as f:
            f.write(b"TAMPERED")
        self.assertFalse(self.reg.verify("foo", "1.0.0"))

    def test_verify_unknown_package_returns_false(self):
        self.assertFalse(self.reg.verify("nope", "0.0.0"))

    def test_verify_with_explicit_signature(self):
        md = self._add("foo", "1.0.0")
        sig = self.reg.sign(md, secret=self.secret)
        self.assertTrue(self.reg.verify(
            "foo", "1.0.0",
            expected_signature=sig,
            secret=self.secret,
        ))

    def test_verify_with_wrong_explicit_signature_returns_false(self):
        self._add("foo", "1.0.0")
        self.assertFalse(self.reg.verify(
            "foo", "1.0.0",
            expected_signature="deadbeef",
            secret=self.secret,
        ))

    # ---------------------------------------------------- conflict

    def test_resolve_all_simple(self):
        self._add("a", "1.0.0", deps={"b": "^1.0.0"})
        self._add("b", "1.0.5")
        resolved = self.reg.resolve_all({"a": "^1.0.0"})
        self.assertEqual(set(resolved.keys()), {"a", "b"})
        self.assertEqual(resolved["a"].version, "1.0.0")
        self.assertEqual(resolved["b"].version, "1.0.5")

    def test_resolve_all_conflict_raises(self):
        self._add("a", "1.0.0", deps={"c": "^1.0.0"})
        self._add("b", "1.0.0", deps={"c": "^2.0.0"})
        self._add("c", "1.0.0")
        # c has no ^2.0.0 candidate at all → DependencyConflict
        with self.assertRaises(DependencyConflict) as ctx:
            self.reg.resolve_all({"a": "^1.0.0", "b": "^1.0.0"})
        self.assertIn("no registered version", str(ctx.exception))

    def test_resolve_all_unsatisfiable_root(self):
        self._add("a", "1.0.0")
        with self.assertRaises(DependencyConflict):
            self.reg.resolve_all({"a": "^99.0.0"})

    def test_resolve_all_no_intersection(self):
        # Two explicit, mutually-exclusive version pairs for `c` via
        # different parents. Both candidates registered, but they
        # don't both satisfy all constraints — the intersection is
        # empty.
        self._add("a", "1.0.0", deps={"c": "^1.0.0"})
        self._add("b", "1.0.0", deps={"c": "~2.0"})
        self._add("c", "1.5.0")
        self._add("c", "2.0.5")
        with self.assertRaises(DependencyConflict):
            self.reg.resolve_all({"a": "^1.0.0", "b": "^1.0.0"})

    def test_resolve_all_transitive_chain(self):
        self._add("a", "1.0.0", deps={"b": "^1.0.0"})
        self._add("b", "1.0.0", deps={"c": "^2.0.0"})
        self._add("c", "2.3.4")
        resolved = self.reg.resolve_all({"a": "*"})
        self.assertEqual(resolved["a"].version, "1.0.0")
        self.assertEqual(resolved["b"].version, "1.0.0")
        self.assertEqual(resolved["c"].version, "2.3.4")

    # ----------------------------------------------------- advisory

    def test_scan_matches_name_and_version(self):
        advisories = [
            {"name": "vulnpkg", "version_constraint": "^1.0",
             "advisory_id": "CVE-2026-0001"},
            {"name": "vulnpkg", "version_constraint": "^2.0",
             "advisory_id": "CVE-2026-0002"},
            {"name": "^vuln*", "version_constraint": "*",
             "advisory_id": "CVE-2026-WILDCARD"},
        ]
        self._add("vulnpkg", "1.0.5")
        hits = self.reg.scan("vulnpkg", "1.0.5", advisories=advisories)
        self.assertIn("CVE-2026-0001", hits)
        # wildcard pattern matches via regex on "vuln*"
        self.assertIn("CVE-2026-WILDCARD", hits)
        self.assertNotIn("CVE-2026-0002", hits)

    def test_scan_no_match(self):
        advisories = [{"name": "otherpkg", "version_constraint": "*",
                       "advisory_id": "WHATEVER"}]
        self._add("foo", "1.0.0")
        self.assertEqual(
            self.reg.scan("foo", "1.0.0", advisories=advisories), [])

    # ---------------------------------------------------- persist

    def test_persistence_reload_from_disk(self):
        self._add("foo", "1.0.0", deps={"bar": "^2.0"})
        # New registry instance over the same root should see
        # existing entries.
        reg2 = PackageRegistry(self.root)
        self.assertEqual(len(reg2), 1)
        self.assertEqual(reg2.get("foo", "1.0.0").dependencies,
                         {"bar": "^2.0"})

    def test_remove_drops_entry_and_tarball(self):
        md = self._add("foo", "1.0.0")
        self.assertTrue(self.reg.remove("foo", "1.0.0"))
        self.assertFalse(self.reg.remove("foo", "1.0.0"))
        self.assertFalse(os.path.isfile(
            os.path.join(self.root, md.tarball_relpath)))
        self.assertEqual(len(self.reg), 0)

    def test_clear_wipes_everything(self):
        self._add("foo", "1.0.0")
        self._add("bar", "1.0.0")
        self.reg.clear()
        self.assertEqual(len(self.reg), 0)
        self.assertFalse(os.path.isfile(self.reg.index_path))

    # ---------------------------------------------------- packager

    def test_publish_package_hook_populates_registry(self):
        workspace = tempfile.mkdtemp(prefix="eigen_publish_test_")
        try:
            pkg = EigenPackager(workspace)
            self.assertTrue(pkg.init_package(name="demo"))
            self.assertTrue(pkg.publish_package())
            # The hook should have created a registry index alongside
            # the legacy flat .tar file.
            self.assertTrue(os.path.isfile(
                os.path.join(pkg.registry_dir, "index.json")))
            reg = PackageRegistry(pkg.registry_dir)
            md = reg.get("demo", "1.0.0")
            self.assertIsNotNone(md)
            self.assertEqual(md.sha256, reg.get("demo", "1.0.0").sha256)
            # Legacy flat tarball still present.
            self.assertTrue(os.path.isfile(
                os.path.join(pkg.registry_dir, "demo-1.0.0.tar")))
            # Re-verifying the on-disk tarball under the new registry
            # should succeed.
            self.assertTrue(reg.verify("demo", "1.0.0"))
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


# Helper used by signature tests above.
def dataclasses_replace(obj, **changes):
    import dataclasses
    return dataclasses.replace(obj, **changes)


if __name__ == "__main__":
    unittest.main()
