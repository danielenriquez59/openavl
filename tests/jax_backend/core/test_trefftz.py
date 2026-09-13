"""Independent source-exclusion and gradient checks for Trefftz kernels."""

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

jax = require_jax()

from openavl.jax.backend import jnp
from openavl.jax.trefftz import _filament_velocity

pytestmark = pytest.mark.core


@pytest.mark.parametrize("sign", [1.0, -1.0, 0.0])
def test_inactive_filament_has_zero_velocity_and_geometry_gradient(sign):
    """A coincident inactive source contributes neither velocity nor gradients."""
    def evaluate(parameters):
        y1, z1, gamma, core = parameters
        vy, vz = _filament_velocity(
            jnp.array([0.25]), jnp.array([0.4]),
            jnp.array([[1.0, y1]]), jnp.array([[2.0, 3.0]]),
            jnp.array([[0.8, z1]]), jnp.array([[1.2, 3.0]]),
            jnp.array([[0.6, gamma]]), jnp.array([[0.0, core]]),
            jnp.array(1.0 / (2.0 * np.pi)), jnp.array(sign),
            jnp.array([[True, False]]),
        )
        return jnp.array([jnp.sum(vy), jnp.sum(vz)])

    parameters = jnp.array([0.25, 0.4, 7.0, 0.0])
    actual = jax.jit(evaluate)(parameters)
    dy1, dz1 = 0.25 - 1.0, 0.4 - 0.8
    dy2, dz2 = 0.25 - 2.0, 0.4 - 1.2
    rsq1, rsq2 = dy1**2 + dz1**2, dy2**2 + dz2**2
    expected = sign * 0.6 / (2.0 * np.pi) * np.array([
        dz1 / rsq1 - dz2 / rsq2, -dy1 / rsq1 + dy2 / rsq2,
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(jax.jit(jax.jacrev(evaluate))(parameters), np.zeros((2, 4)))


def test_disabled_image_has_zero_velocity_and_gradient_at_coincidence():
    """JIT evaluates disabled image kernels too; their arithmetic must be safe."""
    def evaluate(y):
        vy, vz = _filament_velocity(
            jnp.array([0.0]), jnp.array([0.0]),
            y, jnp.array([1.0]), jnp.array([0.0]), jnp.array([0.0]),
            jnp.array([1.0]), jnp.array([0.0]), jnp.array(1.0),
            jnp.array(0.0), jnp.array([True]),
        )
        return jnp.sum(vy + vz)

    value, gradient = jax.jit(jax.value_and_grad(evaluate))(jnp.array(0.0))
    np.testing.assert_array_equal([value, gradient], [0.0, 0.0])
