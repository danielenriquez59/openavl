"""Pytest fixtures for OpenAVL tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import FIXTURES_DIR, GEOMETRIES_DIR, REF_DIR


@pytest.fixture(scope="session")
def geometries_dir() -> Path:
    """Path to AVL geometry input files."""
    return GEOMETRIES_DIR


@pytest.fixture(scope="session")
def ref_dir() -> Path:
    """Path to Fortran reference binary directory."""
    return REF_DIR


@pytest.fixture(scope="session")
def ref_binary(ref_dir: Path):
    """Callable that resolves a Fortran reference binary path by name."""

    def _resolve(name: str) -> Path | None:
        for candidate in (ref_dir / name, ref_dir / f"{name}.exe"):
            if candidate.is_file():
                return candidate
        return None

    return _resolve


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to JSON test fixtures."""
    return FIXTURES_DIR


