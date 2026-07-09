"""Conditional JAX import with a helpful error when JAX is not installed."""

from __future__ import annotations

try:
    import jax
except Exception as exc:  # pragma: no cover - ImportError or jaxlib load failure
    raise ImportError(
        "JAX is required for openavl.jax. Install with: pip install 'openavl[jax]'"
    ) from exc

# A1: enable float64 at first import of openavl.jax, before any array is created.
# Without this, every `dtype=jnp.float64` annotation elsewhere in src/jax is
# silently downcast to float32, which halves solver precision and typically
# fails finite-difference gradient checks.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402 - must follow the x64 config update above

__all__ = ["jax", "jnp"]
