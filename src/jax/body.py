"""Live body source/doublet influences at the current lattice points."""

from openavl.jax.backend import jax, jnp
from openavl.jax.aic import srdvelc_jax


def rebuild_body(geom, flow, refs):
    """Rebuild body strengths and induced velocities for Mach and geometry AD.

    Body centerlines stay fixed; source strengths depend on Mach, and their
    velocity influences depend on the moving wing control/vortex points.
    """
    body = geom.body
    if body is None or body.seg_i1.shape[0] == 0:
        return geom
    circ = geom.circulation
    beta = jnp.sqrt(1.0 - flow.mach**2)
    l1, l2 = body.seg_i1, body.seg_i2
    start, end = body.rl[:, l1].T, body.rl[:, l2].T
    dr = (end - start) / jnp.array([beta, 1.0, 1.0])
    length2 = jnp.sum(dr**2, axis=1)
    length = jnp.sqrt(jnp.where(length2 > 0.0, length2, 1.0))
    axis = dr / length[:, None]
    mid = (start + end) * 0.5 - refs.xyzref
    translation = jnp.broadcast_to(jnp.eye(3), (l1.shape[0], 3, 3))
    rotation = jnp.cross(mid[:, None, :], jnp.eye(3)[None, :, :]).transpose(0, 2, 1)
    velocity = jnp.concatenate([translation, rotation], axis=2)
    velocity = velocity / jnp.array([beta, 1.0, 1.0])[None, :, None]
    along = jnp.einsum("si,sij->sj", axis, velocity)
    normal = velocity - axis[:, :, None] * along[:, None, :]

    # Each segment inherits its body's symmetry factor (SRDSET).
    owner = jnp.argmax((l1[:, None] >= body.lfrst) &
                       (l1[:, None] < body.lfrst + body.nl - 1), axis=1)
    first = body.lfrst[owner]
    last = first + body.nl[owner] - 1
    blen = jnp.abs(body.rl[0, last] - body.rl[0, first])
    factor = jnp.where((circ.iysym == 1) & (body.rl[1, first] <= 0.001 * blen), 0.5, 1.0)
    area_delta = jnp.pi * (body.radl[l2]**2 - body.radl[l1]**2) * factor
    area_mean = jnp.pi * 0.5 * (body.radl[l2]**2 + body.radl[l1]**2) * factor
    source = area_delta[:, None] * along
    doublet = normal * (2.0 * area_mean * length)[:, None, None]
    src_u = jnp.zeros_like(body.src_u).at[l1].set(source)

    radius = jnp.sqrt(0.5 * (body.radl[l1]**2 + body.radl[l2]**2))
    core = circ.srcore * jnp.where(circ.srcore > 0.0, radius, jnp.linalg.norm(end - start, axis=1))

    def influences(points):
        def image_influence(sign, offset):
            a, b = start * sign + offset, end * sign + offset
            dbl = doublet * sign[None, :, None]

            def at_point(point):
                def segment(p1, p2, rc, src, dipole):
                    vs, vd = srdvelc_jax(*point, *p1, *p2, beta, rc)
                    # SRDVELC rows are output velocity components; columns
                    # contract with the doublet orientation (NumPy VSRD).
                    return vs[:, None] * src[None, :] + vd @ dipole
                return jnp.sum(jax.vmap(segment)(a, b, core, source, dbl), axis=0)

            return jax.vmap(at_point)(points.T).transpose(1, 0, 2)

        result = image_influence(jnp.ones(3), jnp.zeros(3))
        if circ.iysym != 0:
            result += circ.iysym * image_influence(jnp.array([1., -1., 1.]), jnp.array([0., 2. * circ.ysym, 0.]))
        if circ.izsym != 0:
            result += circ.izsym * image_influence(jnp.array([1., 1., -1.]), jnp.array([0., 0., 2. * circ.zsym]))
            if circ.iysym != 0:
                result += circ.iysym * circ.izsym * image_influence(jnp.array([1., -1., -1.]), jnp.array([0., 2. * circ.ysym, 2. * circ.zsym]))
        return result

    circ = circ._replace(wcsrd_u=influences(circ.rc), wvsrd_u=influences(geom.force.rv))
    return geom._replace(circulation=circ, body=body._replace(src_u=src_u))
