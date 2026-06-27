"""Import JAX once for backend tests, skipping cleanly when unavailable."""

from __future__ import annotations

import importlib
import sys

import pytest


def require_jax():
    """Return the JAX module or skip the enclosing test module."""
    if "jax" in sys.modules:
        mod = sys.modules["jax"]
        if getattr(mod, "version", None) is None:
            del sys.modules["jax"]
            for name in list(sys.modules):
                if name == "jax" or name.startswith("jax."):
                    del sys.modules[name]
    try:
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        return jax
    except Exception as exc:
        pytest.skip(f"jax unavailable: {exc}", allow_module_level=True)
