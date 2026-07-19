"""Canonical Eigen release metadata.

The package version is sourced from installed distribution metadata when
available and otherwise from ``pyproject.toml`` for source checkouts.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import re

PACKAGE_NAME = "eigen-lang"
CODENAME = "Mars"


def _source_version() -> str | None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        return None
    project_section = pyproject.read_text(encoding="utf-8").split("[project]", 1)[-1]
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', project_section, re.MULTILINE)
    return match.group(1) if match else None


def get_version() -> str:
    source_version = _source_version()
    if source_version is not None:
        return source_version
    try:
        return package_version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0+unknown"


VERSION = get_version()
SHORT_VERSION = ".".join(VERSION.split(".")[:2])
RELEASE_LABEL = f"{SHORT_VERSION} — {CODENAME}"
