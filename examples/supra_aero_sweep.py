#!/usr/bin/env python3
"""Alpha sweep polars for the Supra 3.4 m F3J sailplane.

Runs fixed-alpha cases from -5° to 15° (1° steps) with neutral controls and
plots CL, CD, and Cm versus angle of attack.

Run from the repository root after installing OpenAVL::

    pip install -e .
    python examples/supra_aero_sweep.py
"""

import matplotlib.pyplot as plt
import numpy as np

from openavl import AVLSolver

from supra_geometry import build_supra_aircraft, geometries_dir, supra_mass

# Angle-of-attack sweep: [start, stop, step] in degrees (inclusive of stop).
AOA_START, AOA_STOP, AOA_STEP = -5, 15, 1
aoa_array = np.arange(AOA_START, AOA_STOP + AOA_STEP, AOA_STEP)


def run_alpha_sweep(solver, aoa_deg):
    """Evaluate CL, CD, and Cm at each angle of attack with neutral controls."""
    alpha_out = np.empty_like(aoa_deg, dtype=np.float64)
    cl_out = np.empty_like(aoa_deg, dtype=np.float64)
    cd_out = np.empty_like(aoa_deg, dtype=np.float64)
    cm_out = np.empty_like(aoa_deg, dtype=np.float64)
    linear_accel_out = np.empty((len(aoa_deg), 3), dtype=np.float64)
    rotational_accel_out = np.empty((len(aoa_deg), 3), dtype=np.float64)

    for i, alpha in enumerate(aoa_deg):
        solver.set_variable("alpha", float(alpha))
        solver.execute_run(max_iter=0)
        results = solver.get_results()
        accelerations = solver.get_aero_accel()
        alpha_out[i] = results["alpha_deg"]
        cl_out[i] = results["CL"]
        cd_out[i] = results["CD"]
        cm_out[i] = results["Cm"]
        linear_accel_out[i, :] = accelerations["linear_acceleration_body"]
        rotational_accel_out[i, :] = accelerations["rotational_acceleration_body"]

    return alpha_out, cl_out, cd_out, cm_out, linear_accel_out, rotational_accel_out


def plot_polars(alpha_deg, cl, cd, cm, title):
    """Plot CL, CD, and Cm versus angle of attack."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharex=True)

    axes[0].plot(alpha_deg, cl, marker="o", linewidth=1.5, markersize=4)
    axes[0].set_ylabel("CL")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alpha_deg, cd, marker="o", linewidth=1.5, markersize=4, color="#d62728")
    axes[1].set_ylabel("CD")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(alpha_deg, cm, marker="o", linewidth=1.5, markersize=4, color="#2ca02c")
    axes[2].set_ylabel("Cm")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Angle of attack (deg)")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def main():
    vinf = 12.0  # m/s — fixed airspeed for the polar sweep
    aircraft = build_supra_aircraft()

    solver = AVLSolver(
        aircraft,
        base_dir=geometries_dir,
        mass_file=supra_mass,
        cd0=0.015,
        rho=1.225,
        gravity=9.81,
        xcg=3.75,
        velocity=vinf,
        beta=0.0,
    )

    solver.print_settings()
    print("Alpha sweep")
    print(f"  aoa range : {AOA_START} to {AOA_STOP} deg, step {AOA_STEP}")
    print(f"  velocity  : {vinf:.1f} m/s")
    print(f"  points    : {len(aoa_array)}")
    print()

    alpha_deg, cl, cd, cm, linear_accel, rotational_accel = run_alpha_sweep(solver, aoa_array)

    print(
        f"{'alpha':>8s}  {'CL':>8s}  {'CD':>8s}  {'Cm':>8s}  "
        f"{'ax':>8s}  {'ay':>8s}  {'az':>8s}  {'pdot':>8s}  {'qdot':>8s}  {'rdot':>8s}"
    )
    for a, cl_i, cd_i, cm_i, lin_i, rot_i in zip(alpha_deg, cl, cd, cm, linear_accel, rotational_accel):
        print(
            f"{a:8.2f}  {cl_i:8.4f}  {cd_i:8.4f}  {cm_i:8.5f}  "
            f"{lin_i[0]:8.3f}  {lin_i[1]:8.3f}  {lin_i[2]:8.3f}  "
            f"{rot_i[0]:8.3f}  {rot_i[1]:8.3f}  {rot_i[2]:8.3f}"
        )
    print()

    plot_polars(
        alpha_deg,
        cl,
        cd,
        cm,
        title=f"{aircraft.name} — alpha sweep (V = {vinf:.0f} m/s, neutral controls)",
    )


if __name__ == "__main__":
    main()
