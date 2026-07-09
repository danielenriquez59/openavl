"""Custom-VJP linear solve for circulation (BAKSUBTRANS adjoint pattern)."""

from __future__ import annotations

from openavl.jax.backend import jax, jnp
from jax.scipy.linalg import lu_factor, lu_solve


def lu_factor_aicn(aicn: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """LU factorization of the AIC matrix for repeated solves at fixed geometry."""
    return lu_factor(aicn)


@jax.custom_vjp
def solve_circulation(aicn: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve ``aicn @ gamma = rhs`` for circulation strengths."""
    return jnp.linalg.solve(aicn, rhs)


def _solve_circulation_fwd(
    aicn: jnp.ndarray, rhs: jnp.ndarray
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    """Forward pass: primal solve plus saved values for the transpose adjoint."""
    gamma = solve_circulation(aicn, rhs)
    return gamma, (aicn, gamma)


def _solve_circulation_bwd(
    res: tuple[jnp.ndarray, jnp.ndarray], gamma_bar: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward pass: one transpose solve instead of differentiating LU factorization.

    ``jnp.outer`` flattens its inputs, so it only forms the correct rank-1
    cotangent ``-lam @ gamma.T`` when ``rhs``/``gamma`` is 1-D; a multi-column
    RHS (e.g. a multi-unit-solve) would silently collapse into the wrong
    shape. Branch on ``gamma.ndim`` (static, shape-derived) instead so both
    the vector and matrix RHS cases form the correct outer/matrix product.
    """
    aicn, gamma = res
    lam = jnp.linalg.solve(aicn.T, gamma_bar)
    rhs_bar = lam
    if gamma.ndim == 1:
        aicn_bar = -jnp.outer(lam, gamma)
    else:
        aicn_bar = -lam @ gamma.T
    return aicn_bar, rhs_bar


solve_circulation.defvjp(_solve_circulation_fwd, _solve_circulation_bwd)


@jax.custom_vjp
def solve_from_lu(lu_piv: tuple[jnp.ndarray, jnp.ndarray], rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve ``aicn @ gamma = rhs`` using a precomputed LU factorization."""
    return lu_solve(lu_piv, rhs)


def _solve_from_lu_fwd(
    lu_piv: tuple[jnp.ndarray, jnp.ndarray], rhs: jnp.ndarray
) -> tuple[jnp.ndarray, tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]]:
    """Forward pass: LU back-substitution plus saved circulation for the adjoint."""
    gamma = solve_from_lu(lu_piv, rhs)
    return gamma, (lu_piv, gamma)


def _solve_from_lu_bwd(
    res: tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    gamma_bar: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Backward pass: transpose LU solve; ``lu_piv`` is treated as non-differentiable.

    ``lu_piv`` is a ``(lu, piv)`` tuple-of-arrays primal input, so its
    cotangent must match that pytree structure — a bare ``None`` is only a
    valid cotangent for a single leaf, not a 2-tuple of arrays, and recent
    JAX versions raise as soon as a gradient actually flows through this
    path (e.g. ``JaxAVLSolver.grad`` with ``use_jit=True``). Return
    zeros-like arrays instead: ``lu``/``piv`` are treated as constants (the
    factorization itself is not differentiated; see module docstring), so
    their true cotangent contribution is zero.
    """
    lu_piv, _gamma = res
    lu, piv = lu_piv
    lam = lu_solve(lu_piv, gamma_bar, trans=1)
    return (jnp.zeros_like(lu), jnp.zeros_like(piv)), lam


solve_from_lu.defvjp(_solve_from_lu_fwd, _solve_from_lu_bwd)
