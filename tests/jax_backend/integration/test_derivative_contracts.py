"""Independent checks of units, axes, and complete geometry sensitivities."""

from copy import deepcopy

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

jax = require_jax()
import jax.numpy as jnp

from openavl.solver import AVLSolver
from openavl.jax.analysis import run_analysis
from openavl.jax.freestream import stability_rates_to_body
from openavl.jax.geom_jax import design_params_from_state, snapshot_topology, run_analysis_with_geometry, update_geometry
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_flow, snapshot_refs
from openavl.analysis.deriv import compute_stability_derivatives
from tests.helpers import GEOMETRIES_DIR


def _outputs(result):
    return jnp.concatenate((jnp.array([result.CL, result.CD, result.CY]), result.CM))


def _compact_model(body=False):
    model = deepcopy(AVLSolver(GEOMETRIES_DIR / ("supra.avl" if body else "plane.avl")).model)
    for surf in model.surfaces:
        surf.n_chord = 2
        surf.n_span = 0
        for i, sec in enumerate(surf.sections):
            sec.n_span = 2
            for ctrl in sec.controls:
                ctrl.xhinge = 0.61 + 0.025 * i
    for item in model.bodies:
        item.n_body = 5
    return model


def _solve(model, mach=0.3):
    solver = AVLSolver._from_model(deepcopy(model))
    solver.set_variable("alpha", 4.0)
    solver.set_variable("beta", 3.0)
    solver.set_parameter("mach", mach)
    for n, name in enumerate(solver.state.control_names):
        solver.set_variable(name, 2.0 - n)
    solver.execute_run(max_iter=0)
    return solver


@pytest.mark.parametrize("body", [False, True])
def test_geometry_against_rebuilt_numpy(body):
    """All outputs and all five DV types against fresh NumPy geometry builds."""
    model = _compact_model(body)
    solver = _solve(model)
    state = solver.state
    baseline, refs, flow = snapshot_analysis_geometry(state), snapshot_refs(state), snapshot_flow(state)
    topo = snapshot_topology(state, solver.model)
    params = design_params_from_state(state, solver.model)

    def analysis(p):
        return _outputs(run_analysis_with_geometry(flow, p, topo, baseline, refs))

    np.testing.assert_allclose(analysis(params), np.r_[state.cltot, state.cdtot, state.cytot, state.cmtot], atol=2e-6, rtol=2e-6)
    updated = update_geometry(topo, params, baseline, mach=flow.mach)
    np.testing.assert_allclose(updated.circulation.enc_d, baseline.circulation.enc_d, atol=1e-10, rtol=1e-10)
    jac = jax.jacrev(analysis)(params)
    fields = {"aincs": "ainc_deg", "chords": "chord", "xles": "xle", "yles": "yle", "zles": "zle"}
    # Perturb an outboard section to exercise span/dihedral and hinge movement.
    index = 1
    for field, attribute in fields.items():
        for step in (1e-4, 1e-5):
            results = []
            for sign in (1., -1.):
                changed = deepcopy(model)
                sec = changed.surfaces[0].sections[index]
                delta = sign * step * (180. / np.pi if field == "aincs" else 1.)
                setattr(sec, attribute, getattr(sec, attribute) + delta)
                s = _solve(changed).state
                results.append(np.r_[s.cltot, s.cdtot, s.cytot, s.cmtot])
            fd = (results[0] - results[1]) / (2. * step)
            np.testing.assert_allclose(getattr(jac, field)[:, index], fd, atol=2e-5, rtol=2e-4, err_msg=field)


def test_asymmetric_stability_derivatives():
    """Reference derivatives include output rotation and rate-input scaling."""
    solver = _solve(_compact_model())
    state = solver.state
    flow, refs = snapshot_flow(state), snapshot_refs(state)
    geom = snapshot_analysis_geometry(state)
    hand = compute_stability_derivatives(state)
    direction = -1. if state.lnasa_sa else 1.

    def analysis(f):
        core = f._replace(wrot=stability_rates_to_body(f.alfa, f.wrot, refs, state.lnasa_sa))
        result = run_analysis(core, geom, refs)
        ca, sa = jnp.cos(f.alfa), jnp.sin(f.alfa)
        mx, my, mz = result.CM
        return jnp.array([result.CL, result.CD, result.CY, direction*(mx*ca+mz*sa), my, direction*(mz*ca-mx*sa)])

    # The solved reference has zero rates; nonzero beta/controls expose moment
    # rotation terms that vanish in the old symmetric test cases.
    jac = jax.jacrev(analysis)(flow)
    for i, name in enumerate(("CL", "CD", "CY", "Cl", "Cm", "Cn")):
        np.testing.assert_allclose(jac.alfa[i], getattr(hand, name + "_a"), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(jac.beta[i], getattr(hand, name + "_b"), atol=2e-6, rtol=2e-6)
        for n, rate in enumerate(("p", "q", "r")):
            np.testing.assert_allclose(jac.wrot[i, n], getattr(hand, name + "_" + rate), atol=2e-6, rtol=2e-6)
        for n, control in enumerate(state.control_names):
            np.testing.assert_allclose(jac.delcon[i, n] * 180. / np.pi, getattr(hand, name + "_d")[control], atol=2e-6, rtol=2e-6)


def test_body_mach_derivative_against_numpy():
    """Mach affects body strengths and body-to-wing influences as well as AIC."""
    model = _compact_model(body=True)
    solver = _solve(model)
    geom = snapshot_analysis_geometry(solver.state)
    refs, flow = snapshot_refs(solver.state), snapshot_flow(solver.state)
    jac = jax.jacrev(lambda mach: _outputs(run_analysis(flow._replace(mach=mach), geom, refs)))(flow.mach)
    for step in (1e-4, 1e-5):
        results = []
        for sign in (1., -1.):
            s = _solve(model, mach=float(flow.mach) + sign * step).state
            results.append(np.r_[s.cltot, s.cdtot, s.cytot, s.cmtot])
        np.testing.assert_allclose(jac, (results[0] - results[1]) / (2 * step), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("hinge", [-0.31, 0.67])
@pytest.mark.parametrize("explicit", [False, True])
def test_control_geometry_with_scaled_mirrors(hinge, explicit):
    """Hinge directions/overlap match NumPy for LE/TE and explicit hinges."""
    model = _compact_model()
    wing = model.surfaces[0]
    wing.scale = [1.4, 1.1, 0.7]
    for sec in wing.sections:
        for control in sec.controls:
            control.xhinge = hinge
            if explicit:
                control.vhinge = [0.2, 1.0, 0.1]
    solver = _solve(model)
    baseline = snapshot_analysis_geometry(solver.state)
    topo = snapshot_topology(solver.state, solver.model)
    params = design_params_from_state(solver.state, solver.model)
    changed = deepcopy(model)
    changed.surfaces[0].sections[1].chord += 0.05
    changed.surfaces[0].sections[1].zle += 0.1
    params = params._replace(chords=params.chords.at[1].add(0.05), zles=params.zles.at[1].add(0.1))
    expected = snapshot_analysis_geometry(_solve(changed).state).circulation.enc_d
    actual = update_geometry(topo, params, baseline).circulation.enc_d
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("iysym,izsym", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_body_influence_symmetry(iysym, izsym):
    from openavl.jax.body import rebuild_body
    from openavl.jax.aic import vsrd_jax

    model = _compact_model(body=True)
    model.header.iysym, model.header.izsym = iysym, izsym
    model.header.zsym = -10.
    solver = _solve(model)
    geom = snapshot_analysis_geometry(solver.state)
    rebuilt = rebuild_body(geom, snapshot_flow(solver.state), snapshot_refs(solver.state))
    np.testing.assert_allclose(rebuilt.body.src_u, geom.body.src_u, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rebuilt.circulation.wcsrd_u, geom.circulation.wcsrd_u, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rebuilt.circulation.wvsrd_u, geom.circulation.wvsrd_u, atol=1e-10, rtol=1e-10)
    # Also cover the standalone low-level assembler's row/column convention.
    b, c, s = geom.body, geom.circulation, solver.state
    direct = vsrd_jax(s.betm, iysym, s.ysym, izsym, s.zsym, s.srcore,
                      s.nbody, b.lfrst, b.nl, b.rl, b.radl, 6,
                      b.src_u, jnp.asarray(s.dbl_u), c.rc[:, :1])
    np.testing.assert_allclose(direct, c.wcsrd_u[:, :1], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_openmdao_units_rates_and_totals(mode):
    om = pytest.importorskip("openmdao.api")
    from openavl.jax.openmdao_group import OpenAVLGroup
    from openavl.jax.openmdao import JaxAVLComp

    path = GEOMETRIES_DIR / "plane.avl"
    prob = om.Problem(reports=False)
    prob.model = OpenAVLGroup(geo_file=str(path))
    prob.setup(mode=mode)
    values = dict(alpha=4., beta=3., pb2v=.02, qc2v=.01, rb2v=-.015, mach=.3, aileron=2., elevator=-1.5, rudder=1.)
    for name, value in values.items():
        prob.set_val(name, value)
    prob.run_model()

    # Independent NumPy run with the same physical inputs.
    solver = AVLSolver(path)
    solver.set_variable("alpha", values["alpha"])
    solver.set_variable("beta", values["beta"])
    solver.set_parameter("mach", values["mach"])
    refs = snapshot_refs(solver.state)
    a = np.deg2rad(values["alpha"])
    sign = -1. if solver.state.lnasa_sa else 1.
    p, q, r = values["pb2v"] * 2 / float(refs.bref), values["qc2v"] * 2 / float(refs.cref), values["rb2v"] * 2 / float(refs.bref)
    solver.state.wrot[:] = [sign*(p*np.cos(a)-r*np.sin(a)), q, sign*(p*np.sin(a)+r*np.cos(a))]
    for name in ("aileron", "elevator", "rudder"):
        solver.set_variable(name, values[name])
    solver.execute_run(max_iter=0)
    s = solver.state
    expected = dict(CL=s.cltot, CD=s.cdtot, CY=s.cytot, Cl=sign*(s.cmtot[0]*np.cos(a)+s.cmtot[2]*np.sin(a)), Cm=s.cmtot[1], Cn=sign*(s.cmtot[2]*np.cos(a)-s.cmtot[0]*np.sin(a)))
    for name, value in expected.items():
        np.testing.assert_allclose(prob.get_val(name), value, atol=2e-7, rtol=2e-7, err_msg=name)

    old = om.Problem(reports=False)
    old.model.add_subsystem("avl", JaxAVLComp(geo_file=str(path)))
    old.setup()
    for name in ("alpha", "beta", "pb2v", "qc2v", "rb2v", "mach"):
        old.set_val("avl." + name, values[name], units="deg" if name in ("alpha", "beta") else None)
    for n, name in enumerate(("aileron", "elevator", "rudder")):
        old.set_val(f"avl.delcon_{n}", values[name], units="deg")
    old.run_model()
    for name, value in zip(("CL", "CD", "CY", "CMx", "CMy", "CMz"), np.r_[s.cltot, s.cdtot, s.cytot, s.cmtot]):
        np.testing.assert_allclose(old.get_val("avl." + name), value, atol=2e-7, rtol=2e-7)

    wrt = list(values)
    checks = prob.check_totals(of=list(expected), wrt=wrt, method="fd", form="central", step=1e-5, out_stream=None)
    assert len(checks) == len(expected) * len(wrt)
    for key, data in checks.items():
        np.testing.assert_allclose(data["J_" + mode], data["J_fd"], atol=2e-5, rtol=2e-4, err_msg=str(key))
