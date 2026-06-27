"""Shared helpers for OpenAVL tests."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data" / "avl"
GEOMETRIES_DIR = DATA_DIR / "geometries"
REF_DIR = DATA_DIR / "ref"
FIXTURES_DIR = TESTS_DIR / "fixtures"


def load_json_fixture(fixtures_dir: Path, name: str) -> dict:
    """Load a JSON fixture file from tests/fixtures/."""
    path = fixtures_dir / name
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def run_ref_binary(ref_path: Path, stdin: str | None = None) -> list[float]:
    """Run a Fortran reference binary and parse numeric stdout."""
    try:
        proc = subprocess.run(
            [str(ref_path)],
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        pytest.skip(f"Cannot execute reference binary: {exc}")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or f"ref exited with {proc.returncode}")
    matches = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?",
        proc.stdout,
    )
    return [float(v.replace("d", "e").replace("D", "e")) for v in matches]
