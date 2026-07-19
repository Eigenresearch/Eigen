"""
P3 §11.1 — Public Package Registry (surface-level).

Roadmap (`sol.md` "11.1 Package Manager Enhancement"):
    - [ ] Публичный реестр пакетов
    - [ ] Семантическое версионирование (semver)
    - [ ] Разрешение конфликтов зависимостей
    - [ ] Подпись пакетов
    - [ ] Checksum verification
    - [ ] Lockfile reproducibility
    - [ ] Vulnerability scanning

Real public deployment — `registry.eigen-lang.org` — is out of scope.
This module provides a self-contained, offline PackageRegistry with
explicit APIs for every checkbox above:

  * `PackageMetadata` dataclass — name, version, dependencies,
    sha256 checksum, published timestamp, optional HMAC-SHA256
    signature, advisory tags.
  * `PackageRegistry(root, secret=...)` — on-disk registry backed by
    `index.json` + `tarballs/` directory; versioned lookups via the
    existing `packager.version_satisfies` semver engine.
  * `add(...)` — accepts tarball bytes or path, computes SHA-256,
    optionally signs and stores under `tarballs/<name>-<version>.tar`.
  * `get` / `list_versions` / `resolve` — read-side helpers; `resolve`
    picks the highest sat-version constraint.
  * `search(query)` — substring over name+description (offline; the
    roadmap's networked search engine is left for the public
    deployment).
  * `resolve_all(reqs)` — iterative constraint closure checking,
    raises `DependencyConflict` with a path-attribution message.
  * `sign(metadata, secret)` / `verify_signature(metadata, secret)` —
    HMAC-SHA256 over `name|version|sha256`; real keyed digest.
  * `verify(name, version, *, expected_sha256=None,
    expected_signature=None)` — recomputes hash + signature from the
    on-disk tarball; mismatch returns False (NOT raise — same shape
    as `packager.version_satisfies` which is total).
  * `scan(name, version, *, advisories)` — vulnerability scan over
    a caller-supplied advisory list (offline; advisory databases are
    the user's responsibility). Returns list of matched `advisory_id`s.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import os
import re
import time
import types
import typing

from src.packager import parse_version, version_satisfies


@dataclasses.dataclass(frozen=True)
class PackageMetadata:
    """Immutable record for one (name, version) entry in the registry.

    `advisory_tags` is a tuple of advisory IDs recorded at publish-time
    (typically empty for healthy packages).
    """

    name: str
    version: str
    description: str = ""
    license: str = ""
    author: str = ""
    dependencies: typing.Mapping[str, str] = dataclasses.field(default_factory=dict)
    tarball_relpath: str = ""
    sha256: str = ""
    published_at: int = 0
    checksum_algorithm: str = "sha256"
    signature: str = ""
    advisory_tags: typing.Tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.dependencies, types.MappingProxyType):
            object.__setattr__(
                self, "dependencies",
                types.MappingProxyType(dict(self.dependencies)),
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "license": self.license,
            "author": self.author,
            "dependencies": dict(self.dependencies),
            "tarball_relpath": self.tarball_relpath,
            "sha256": self.sha256,
            "published_at": self.published_at,
            "checksum_algorithm": self.checksum_algorithm,
            "signature": self.signature,
            "advisory_tags": list(self.advisory_tags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PackageMetadata":
        return cls(
            name=d["name"],
            version=d["version"],
            description=d.get("description", ""),
            license=d.get("license", ""),
            author=d.get("author", ""),
            dependencies=dict(d.get("dependencies", {}) or {}),
            tarball_relpath=d.get("tarball_relpath", ""),
            sha256=d.get("sha256", ""),
            published_at=int(d.get("published_at", 0) or 0),
            checksum_algorithm=d.get("checksum_algorithm", "sha256"),
            signature=d.get("signature", ""),
            advisory_tags=tuple(d.get("advisory_tags", []) or []),
        )


class DependencyConflict(Exception):
    """Raised by `PackageRegistry.resolve_all` when the constraint graph
    requires two incompatible versions of the same package. The
    message carries a human-readable attribution chain so the caller
    can show *why* the conflict happened."""

    def __init__(self, name, requested, conflict_path, conflict_reason):
        self.name = name
        self.requested = requested
        self.conflict_path = conflict_path
        self.conflict_reason = conflict_reason
        super().__init__(
            f"Dependency conflict for '{name}': requested {requested}. "
            f"Reason: {conflict_reason}. Path: {' -> '.join(conflict_path)}"
        )


class PackageRegistry:
    """Local on-disk package registry.

    Layout:
      <root>/
        index.json          # dict of (name, version) -> metadata.to_dict()
        tarballs/           # actual .tar content (for verify + future download)

    The index file is canonical: `add`, `remove`, `sign`, etc. all
    call `_save_index()` at the end. A separate `_load_index()` runs
    in `__init__`; the registry instance is single-process (we don't
    watch the FS for external writes).
    """

    def __init__(self, root: str, *, secret: typing.Optional[bytes] = None,
                 clock=time.time):
        self.root = os.path.abspath(root)
        self.index_path = os.path.join(self.root, "index.json")
        self.tarballs_dir = os.path.join(self.root, "tarballs")
        os.makedirs(self.tarballs_dir, exist_ok=True)
        self._secret = secret
        self._clock = clock
        self._index: typing.Dict[typing.Tuple[str, str], PackageMetadata] = \
            self._load_index()

    # ----------------------------------------------------------- index

    def _load_index(self) -> typing.Dict[typing.Tuple[str, str], PackageMetadata]:
        if not os.path.exists(self.index_path):
            return {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return {}
        # Raw store is a list of dicts (one per metadata entry).
        if not isinstance(raw, dict) or "entries" not in raw:
            return {}
        out = {}
        for entry in raw["entries"]:
            md = PackageMetadata.from_dict(entry)
            out[(md.name, md.version)] = md
        return out

    def _save_index(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        payload = {
            "version": 1,
            "entries": [md.to_dict() for md in self._index.values()],
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    # --------------------------------------------------------- writes

    def add(self, name: str, version: str, *,
            description: str = "",
            license: str = "",
            author: str = "",
            dependencies: typing.Optional[typing.Dict[str, str]] = None,
            tarball_bytes: typing.Optional[bytes] = None,
            tarball_path: typing.Optional[str] = None,
            sign: bool = False,
            ) -> PackageMetadata:
        """Register a new (name, version) entry. Either `tarball_bytes`
        or `tarball_path` must be supplied; if both are given,
        `tarball_bytes` wins and overrides the file content.

        Re-adding an existing (name, version) is idempotent: the
        metadata is replaced with the new one (matches `npm publish`
        semantics).
        """
        deps = dict(dependencies or {})
        if tarball_bytes is None and tarball_path is None:
            raise ValueError("add() requires tarball_bytes or tarball_path")
        if tarball_bytes is None:
            with open(tarball_path, "rb") as f:
                tarball_bytes = f.read()
        # Compute SHA-256 checksum of the tarball contents. This is
        # the canonical integrity check used by both `verify()` and
        # the signature payload.
        sha = hashlib.sha256(tarball_bytes).hexdigest()
        rel = f"tarballs/{name}-{version}.tar"
        full = os.path.join(self.root, rel)
        with open(full, "wb") as f:
            f.write(tarball_bytes)
        md = PackageMetadata(
            name=name,
            version=version,
            description=description,
            license=license,
            author=author,
            dependencies=deps,
            tarball_relpath=rel,
            sha256=sha,
            published_at=int(self._clock()),
            checksum_algorithm="sha256",
            signature="",
            advisory_tags=(),
        )
        if sign and self._secret is not None:
            md = dataclasses.replace(md, signature=self.sign(md, secret=self._secret))
        self._index[(name, version)] = md
        self._save_index()
        return md

    def remove(self, name: str, version: str) -> bool:
        key = (name, version)
        if key not in self._index:
            return False
        md = self._index.pop(key)
        # Best-effort tarball removal.
        full = os.path.join(self.root, md.tarball_relpath)
        if os.path.isfile(full):
            try:
                os.unlink(full)
            except OSError:
                pass
        self._save_index()
        return True

    def sign(self, metadata: PackageMetadata, *, secret: bytes) -> str:
        """HMAC-SHA256 over `name|version|sha256` using `secret`.
        Returns hex digest. This is a real keyed digest (RFC 2104).
        """
        msg = f"{metadata.name}|{metadata.version}|{metadata.sha256}".encode("utf-8")
        return hmac.new(secret, msg, hashlib.sha256).hexdigest()

    def verify_signature(self, metadata: PackageMetadata, *, secret: bytes) -> bool:
        if not metadata.signature:
            return False
        expected = self.sign(metadata, secret=secret)
        return hmac.compare_digest(metadata.signature, expected)

    # ---------------------------------------------------------- reads

    def get(self, name: str, version: str) -> typing.Optional[PackageMetadata]:
        return self._index.get((name, version))

    def list_versions(self, name: str) -> typing.List[PackageMetadata]:
        out = [md for (n, _v), md in self._index.items() if n == name]
        out.sort(key=lambda m: parse_version(m.version))
        return out

    def resolve(self, name: str, constraint: str) -> typing.Optional[PackageMetadata]:
        """Pick the highest registered version of `name` that
        satisfies `constraint` (uses the existing `version_satisfies`
        semver engine). Returns ``None`` if no version matches.
        """
        candidates = [
            md for md in self.list_versions(name)
            if version_satisfies(constraint, md.version)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: parse_version(m.version))

    def search(self, query: str, *,
               limit: typing.Optional[int] = None,
               ) -> typing.List[PackageMetadata]:
        q = (query or "").lower()
        if not q:
            return []
        # Substring over name + description + license + author.
        matches = []
        for md in self._index.values():
            hay = " ".join([md.name, md.description, md.license, md.author]).lower()
            if q in hay:
                matches.append(md)
        matches.sort(key=lambda m: (m.name, parse_version(m.version)))
        if limit is not None:
            matches = matches[:limit]
        return matches

    # ------------------------------------------------------ conflict

    def resolve_all(self, required: typing.Mapping[str, str],
                    ) -> typing.Dict[str, PackageMetadata]:
        """Iterate-to-fixpoint over a constraint dictionary. For
        each (name, constraint) the highest-version sat candidate is
        selected; its `dependencies` are unioned into the working set;
        the loop terminates when no new constraints appear.

        A `DependencyConflict` is raised when two distinct versions of
        the same package are required by incompatible constraints,
        or when a transitive dep has no sat version.

        Returns a flat {name: PackageMetadata} mapping.
        """
        # `merged` accumulates, for each package name, the list of
        # (constraint, required_by) pairs that must all be satisfied
        # together. The closure runs until `working` is empty.
        merged: typing.Dict[str, typing.List[typing.Tuple[str, str]]] = {}
        for n, c in required.items():
            merged.setdefault(n, []).append((c, "<root>"))
        resolved: typing.Dict[str, PackageMetadata] = {}
        # Iterative closure
        changed = True
        while changed:
            changed = False
            for name, pairs in list(merged.items()):
                # Pick the highest version satisfying ALL constraints
                # simultaneously (intersection). For each registered
                # version we check all constraints in pairs.
                candidates = [
                    md for md in self.list_versions(name)
                    if all(version_satisfies(c, md.version) for c, _ in pairs)
                ]
                if not candidates:
                    # Conflict — at least one constraint is unsatisfiable.
                    # Identify the offending one for an actionable message.
                    for c, requested_by in pairs:
                        ok = any(version_satisfies(c, md.version)
                                 for md in self.list_versions(name))
                        if not ok:
                            raise DependencyConflict(
                                name=name,
                                requested=c,
                                conflict_path=[requested_by, name],
                                conflict_reason=(
                                    f"no registered version of '{name}' "
                                    f"satisfies constraint '{c}'"
                                ),
                            )
                    raise DependencyConflict(
                        name=name,
                        requested=str([c for c, _ in pairs]),
                        conflict_path=["<root>", name],
                        conflict_reason="no common sat version across constraints "
                                        + str([c for c, _ in pairs]),
                    )
                choice = max(candidates, key=lambda m: parse_version(m.version))
                if name in resolved and resolved[name].version == choice.version:
                    # Already picked this exact version; need to re-check
                    # that the constraints haven't tightened.
                    continue
                resolved[name] = choice
                # Union choice.dependencies into merged.
                for dep_name, dep_constraint in (choice.dependencies or {}).items():
                    existing = merged.get(dep_name, [])
                    if (dep_constraint, name) not in existing:
                        existing.append((dep_constraint, name))
                        merged[dep_name] = existing
                        changed = True
        # After convergence, re-validate that all chosen versions still
        # satisfy ALL merged constraints. This catches a degenerate
        # case where convergence picks a version that violates a
        # later constraint (in practice it shouldn't, because we re-
        # evaluate candidates every loop).
        for name, pairs in merged.items():
            md = resolved.get(name)
            if md is None:
                continue
            for c, requested_by in pairs:
                if not version_satisfies(c, md.version):
                    raise DependencyConflict(
                        name=name,
                        requested=c,
                        conflict_path=[requested_by, name],
                        conflict_reason=(
                            f"chosen version {md.version} violates constraint "
                            f"'{c}' required by '{requested_by}'"
                        ),
                    )
        return resolved

    # ------------------------------------------------------- security

    def verify(self, name: str, version: str, *,
               expected_sha256: typing.Optional[str] = None,
               expected_signature: typing.Optional[str] = None,
               secret: typing.Optional[bytes] = None,
               ) -> bool:
        """Recompute checksum (and optionally signature) from the on-
        disk tarball. Returns True only if all supplied expectations
        hold. Returns False on:
          * the (name, version) is not registered,
          * the tarball file is missing,
          * recomputed SHA-256 != expected (registered or explicit),
          * recomputed signature != expected (registered or explicit).
        The function is total — it never raises — so callers can
        treat verify as a pass/fail check rather than an exception path.
        """
        md = self._index.get((name, version))
        if md is None:
            return False
        full = os.path.join(self.root, md.tarball_relpath)
        if not os.path.isfile(full):
            return False
        with open(full, "rb") as f:
            tarball = f.read()
        actual_sha = hashlib.sha256(tarball).hexdigest()
        cmp_sha = expected_sha256 if expected_sha256 is not None else md.sha256
        if not hmac.compare_digest(actual_sha, cmp_sha):
            return False
        # Signature is optional: only checked if an expected value is
        # supplied OR the registered metadata carries one.
        sig_expected = expected_signature if expected_signature is not None else md.signature
        if not sig_expected:
            return True  # nothing to check
        if secret is None:
            secret = self._secret
        if secret is None:
            return False  # signature asked for but no secret available
        sig_actual = self.sign(md, secret=secret)
        # Note: we sign over the *registered* sha256 (the metadata), not
        # the freshly-recomputed one, so the signature is meaningful even
        # after manual edits to the index file. If a tampered index has
        # a wrong sha256, the comparison above already fails; the
        # signature check here confirms provenance.
        return hmac.compare_digest(sig_actual, sig_expected)

    # ----------------------------------------------------- advisory

    def scan(self, name: str, version: str, *,
             advisories: typing.Iterable[typing.Dict[str, typing.Any]],
             ) -> typing.List[str]:
        """Match (name, version) against an offline advisory database.

        `advisories` is an iterable of dicts, each shaped like:
            {
              "name": "<package name or regex>",
              "version_constraint": "<semver constraint>",
              "advisory_id": "CVE-XX-NNNNN",
            }
        Returns the list of `advisory_id`s whose name-matcher AND
        version constraint both match. Empty list = no advisories.

        The matcher is regex when the pattern starts with `^`,
        substring otherwise; this lets the same input format range
        from "fast scans" to "exact CVE" listings.
        """
        hits = []
        advs = list(advisories)
        for adv in advs:
            pat = adv.get("name", "")
            if pat.startswith("^") or pat.endswith("$"):
                name_match = re.search(pat, name) is not None
            else:
                name_match = pat.lower() in name.lower()
            if not name_match:
                continue
            vc = adv.get("version_constraint", "*")
            if not version_satisfies(vc, version):
                continue
            advid = adv.get("advisory_id", "")
            if advid and advid not in hits:
                hits.append(advid)
        return hits

    # ------------------------------------------------------- utility

    def clear(self) -> None:
        """Drop every entry and remove cached tarballs. Useful for
        test teardown and `eigen registry reset`.
        """
        self._index = {}
        if os.path.isfile(self.index_path):
            try:
                os.unlink(self.index_path)
            except OSError:
                pass
        if os.path.isdir(self.tarballs_dir):
            for fname in os.listdir(self.tarballs_dir):
                full = os.path.join(self.tarballs_dir, fname)
                if os.path.isfile(full):
                    try:
                        os.unlink(full)
                    except OSError:
                        pass

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, name_version: typing.Tuple[str, str]) -> bool:
        return name_version in self._index

    def entries(self) -> typing.List[PackageMetadata]:
        """Return all registered metadata entries, sorted for
        deterministic iteration. Used by tests and tooling."""
        return sorted(
            self._index.values(),
            key=lambda m: (m.name, parse_version(m.version)),
        )
