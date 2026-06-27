#!/usr/bin/env python3
"""Extract surface forces for a V-n diagram maneuver point on the Supra sailplane.

Given a target load factor and airspeed, this example computes the lift
coefficient required for equilibrium in a steady pull-up (n = L/W), trims
the aircraft so that CL matches that target and Cm = 0, then reports
per-surface force and moment coefficients.

"""

from pathlib import Path

from openavl import AVLSolver

from supra_geometry import calculate_aero_accelerations, format_vector


def surface_labels(solver):
    """Build human-readable labels for each aerodynamic surface index."""
    labels = []
    for surf in solver.model.surfaces:
        labels.append(surf.name)
        if surf.yduplicate is not None:
            labels.append(f"{surf.name} (mirror)")
    return labels


def main():
    repo_root = Path(__file__).resolve().parents[1]
    supra_avl = repo_root / "tests" / "data" / "avl" / "geometries" / "supra.avl"
    supra_mass = repo_root / "tests" / "data" / "avl" / "mass" / "supra.mass"

    # --- Maneuver point on the V-n diagram -----------------------------------
    #
    # Positive maneuvering envelope: steady pull-up at fixed airspeed.
    # 15 m/s is a typical thermal/soaring speed for this 3.4 m F3J sailplane.
    load_factor = 3.0
    vinf = 10.0  # m/s
    rho = 1.225  # kg/m^3
    gravity = 9.81  # m/s^2

    solver = AVLSolver(
        supra_avl,
        mass_file=supra_mass,
        cd0=0.015,
        rho=rho,
        gravity=gravity,
        xcg=3.75,
    )

    state  = solver.state
    sref_d = state.sref * state.unitl * state.unitl
    mass   = state.rmass0

    # n = L/W, L = CL * q * S  =>  CL = n * W / (q * S)
    weight     = mass * gravity
    q_pressure = 0.5 * rho * vinf**2
    cl_target  = load_factor * weight / (q_pressure * sref_d)
 
    print("V-n maneuver point")
    print(f"  aircraft : {supra_avl.name}")
    print(f"  mass     : {mass:.3f} kg")
    print(f"  Sref     : {sref_d:.4f} m^2")
    print(f"  n        : {load_factor:.1f} g")
    print(f"  Vinf     : {vinf:.1f} m/s")
    print(f"  CL target: {cl_target:.4f}")
    print()

    # --- Trim: alpha adjusts CL, elevator adjusts Cm -------------------------
    solver.set_parameter("cl", cl_target)
    solver.set_parameter("velocity", vinf)
    solver.setup_trim(mode=1)
    solver.set_constraint("elevator", "cm", 0.0)

    solver.execute_run(max_iter=20)
    results = solver.get_results()

    n_achieved = results["CL"] * q_pressure * sref_d / (mass * gravity)

    print("Trimmed flight")
    print(f"  converged : {results['converged']}")
    print(f"  alpha     : {results['alpha_deg']:.2f} deg")
    print(f"  CL        : {results['CL']:.4f}")
    print(f"  CD        : {results['CD']:.4f}")
    print(f"  Cm        : {results['Cm']:.5f}")
    print(f"  n (check) : {n_achieved:.3f} g")
    print()

    accelerations = calculate_aero_accelerations(solver)

    print("Aero accelerations from integrated loads")
    print(f"  linear body [x, y, z] : {format_vector(accelerations['linear_acceleration_body'])} m/s^2")
    print(f"  angular body [p, q, r]: {format_vector(accelerations['rotational_acceleration_body'])} rad/s^2")
    print()

    # --- Per-surface force coefficients --------------------------------------
    #
    # clsurf, cdsurf, cysurf, cfsurf, and cmsurf hold stability-axis
    # coefficients integrated over each lifting surface.

    labels = surface_labels(solver)
    print(f"{'Surface':<22s}  {'CL':>8s}  {'CD':>8s}  {'CY':>8s}  {'Cl':>8s}  {'Cm':>8s}  {'Cn':>8s}")
    print("-" * 78)

    for isurf in range(state.nsurf):
        if not state.lfload[isurf]:
            continue
        name = labels[isurf] if isurf < len(labels) else f"surface {isurf}"
        cl_s = state.clsurf[isurf]
        cd_s = state.cdsurf[isurf]
        cy_s = state.cysurf[isurf]
        cm_s = state.cmsurf[:, isurf]
        print(
            f"{name:<22s}  "
            f"{cl_s:8.4f}  "
            f"{cd_s:8.4f}  "
            f"{cy_s:8.4f}  "
            f"{cm_s[0]:8.4f}  "
            f"{cm_s[1]:8.4f}  "
            f"{cm_s[2]:8.4f}"
        )

    print()
    print("Strip lift distribution")
    unitl = state.unitl

    for isurf in range(state.nsurf):
        if not state.lfload[isurf]:
            continue
        name = labels[isurf] if isurf < len(labels) else f"surface {isurf}"
        j0 = state.jfrst[isurf]
        nj = state.nj[isurf]

        print()
        print(name)
        print(
            f"  {'strip':>5s}  {'y (m)':>8s}  {'z (m)':>8s}  {'c (m)':>8s}  "
            f"{'CL_strip':>9s}  {'c_nc':>8s}  {'CL_local':>9s}  {'lift (N)':>9s}"
        )
        print("  " + "-" * 76)

        total_lift = 0.0
        for jj in range(nj):
            j = j0 + jj
            if state.lstripoff[j] or state.wstrip[j] == 0.0:
                continue
            y = state.rle[1, j] * unitl
            z = state.rle[2, j] * unitl
            chord = state.chord[j] * unitl
            strip_area = state.chord[j] * state.wstrip[j] * unitl * unitl
            lift = state.clstrp[j] * q_pressure * strip_area
            total_lift += lift
            print(
                f"  {jj + 1:5d}  "
                f"{y:8.3f}  "
                f"{z:8.3f}  "
                f"{chord:8.3f}  "
                f"{state.clstrp[j]:9.4f}  "
                f"{state.cnc[j]:8.3f}  "
                f"{state.clt_lstrp[j]:9.4f}  "
                f"{lift:9.3f}"
            )

        print(f"  {'total':>5s}  {'':>8s}  {'':>8s}  {'':>8s}  {'':>9s}  {'':>8s}  {'':>9s}  {total_lift:9.3f}")


if __name__ == "__main__":
    main()
