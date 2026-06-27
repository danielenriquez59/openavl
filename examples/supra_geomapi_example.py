#!/usr/bin/env python3
"""Walkthrough: analyze the Supra 3.4 m F3J sailplane via the Geometry API.

This example builds the same aircraft programmatically (no ``.avl`` file) and
runs the same trim / stability analysis as ``supra_example.py``.

Run from the repository root after installing OpenAVL::

    pip install -e .
    python examples/supra_geomapi_example.py
"""

from openavl import AVLSolver

from supra_geometry import (
    build_supra_aircraft,
    calculate_aero_accelerations,
    format_vector,
    geometries_dir,
    supra_mass,
)


def main():
    aircraft = build_supra_aircraft()

    # --- 1. Load the aircraft -------------------------------------------------
    #
    # AVLSolver accepts an Aircraft directly.  base_dir resolves relative
    # airfoil and body file paths (ag40d.dat, fuseSupra.dat, etc.).

    solver = AVLSolver(
        aircraft,
        base_dir=geometries_dir,
        mass_file=supra_mass,
        cd0=0.015,
        rho=1.225,
        gravity=9.81,
        xcg=3.75,
    )

    print("Loaded:", aircraft.name)
    print(f"  surfaces : {solver.state.nsurf}")
    print(f"  vortices : {solver.state.nvor}")
    print(f"  controls : {', '.join(solver.state.control_names)}")
    print(f"  mass     : {solver.model.mass.mass:.3f} kg")
    print()

    # --- 2. Set trim constraints ------------------------------------------------

    solver.set_parameter("cl", 0.7)
    solver.setup_trim(mode=1)

    solver.set_constraint("elevator", "cm", 0.0)
    # The following default to 0.0, but can be changed.
    # solver.set_constraint("beta", "beta", 0.0)
    # solver.set_constraint("aileron", "cll", 0.0)
    # solver.set_constraint("rudder", "cn", 0.0)
    # solver.set_constraint("alpha", "cl", 0.7)

    # --- 3. Run the solver ------------------------------------------------------

    solver.execute_run(max_iter=20)
    results = solver.get_results()

    print("Trimmed flight (CL = 0.7)")
    print(f"  alpha  : {results['alpha_deg']:.2f} deg")
    print(f"  CL     : {results['CL']:.4f}")
    print(f"  CD     : {results['CD']:.4f}")
    print(f"  Cm     : {results['Cm']:.5f}")
    print()

    accelerations = calculate_aero_accelerations(solver)

    print("Aero accelerations from integrated loads")
    print(f"  linear body [x, y, z] : {format_vector(accelerations['linear_acceleration_body'])} m/s^2")
    print(f"  angular body [p, q, r]: {format_vector(accelerations['rotational_acceleration_body'])} rad/s^2")
    print()

    # --- 4. Stability derivatives -----------------------------------------------

    derivs = solver.get_stability_derivatives()

    print("Stability derivatives")
    print(f"  CL_a = {derivs.CL_a:.4f}   Cm_a = {derivs.Cm_a:.4f}")
    print(f"  CL_q = {derivs.CL_q:.4f}   Cm_q = {derivs.Cm_q:.4f}")
    print(f"  Cn_r = {derivs.Cn_r:.4f}   Cl_p = {derivs.Cl_p:.4f}")
    print()

    print("Control derivatives (per radian deflection)")
    for name in solver.state.control_names:
        print(
            f"  {name:8s}  "
            f"CL_d = {derivs.CL_d[name]:8.4f}  "
            f"Cm_d = {derivs.Cm_d[name]:8.4f}  "
            f"Cl_d = {derivs.Cl_d[name]:8.4f}  "
            f"Cn_d = {derivs.Cn_d[name]:8.4f}"
        )
    print()

    # --- 5. Flight-dynamic modes ------------------------------------------------

    modes = solver.eigenvalues()

    print("Flight-dynamic modes")
    for mode in modes.modes:
        if mode.frequency_hz > 0.0:
            print(
                f"  {mode.name:16s}  "
                f"freq = {mode.frequency_hz:.3f} Hz,  "
                f"damping = {mode.damping_ratio:.3f}"
            )
        elif mode.time_constant is not None and mode.time_constant > 0.0:
            print(f"  {mode.name:16s}  time constant = {mode.time_constant:.1f} s")
        else:
            print(f"  {mode.name:16s}  real = {mode.eigenvalue.real:.4f}")
    print()

    # --- 6. Visualization -----------------------------------------------------

    solver.plot_aircraft()

    # --- 7. Wing lift distribution (trimmed, CL = 0.7) ------------------------

    solver.plot_lift_distribution(
        component=1,
        title=f"{aircraft.name} — wing lift distribution (CL = 0.7)",
    )


if __name__ == "__main__":
    main()
