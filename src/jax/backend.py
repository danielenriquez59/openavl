"""Conditional JAX import with a helpful error when JAX is not installed."""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - ImportError or jaxlib load failure
    raise ImportError(
        "JAX is required for openavl.jax. Install with: pip install 'openavl[jax]'"
    ) from exc

__all__ = ["jax", "jnp"]
