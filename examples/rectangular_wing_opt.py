#!/usr/bin/env python3
"""Tutorial: Hershey bar wing analysis and chord optimization with OpenMDAO.

This script walks through OpenAVL's OpenMDAO wrapper on the classic Hershey
bar geometry (``w.avl``):

1. Run a baseline aerodynamic analysis and inspect JAX derivatives.
2. Optimize section chords (and trim angle of attack) to minimize drag at CL = 0.7.

Install dependencies from the repository root::

    pip install -e ".[jax]"
    pip install openmdao

Then run::

    python examples/rectangular_wing_opt.py

On Windows, if JAX fails to import, try a standard venv or WSL.
"""

from pathlib import Path

import jax
import numpy as np
import openmdao.api as om

from openavl.jax.openmdao_group import OpenAVLGroup

jax.config.update("jax_enable_x64", True)

# Path to the bundled Hershey bar geometry (wing only, YDUPLICATE, two sections).
repo_root = Path(__file__).resolve().parents[1]
hershey_avl = repo_root / "tests" / "data" / "avl" / "geometries" / "w.avl"

# OpenAVLGroup exposes section chords as "SurfaceName:chords".
# w.avl has one surface named "Wing" with root and tip sections.
chord_var = "Wing:chords"

# --- 1. Build the OpenMDAO problem ------------------------------------------
#
# OpenAVLGroup wraps the JAX analysis pipeline and promotes inputs/outputs
# (alpha, CL, CD, Wing:chords, ...) to the top level.
print("Building problem...")
prob = om.Problem()
prob.model = OpenAVLGroup(geo_file=str(hershey_avl))
prob.setup()

prob.set_val("alpha", 4.0, units="deg")
prob.set_val("beta", 0.0)
prob.set_val("mach", 0.0)

# --- 2. Baseline analysis -----------------------------------------------------
#
# run_model() evaluates CL, CD, Cm, ... at the current flight condition and
# geometry.  Derivatives come from JAX automatic differentiation.
print("Running baseline analysis...")
prob.run_model()

chords = np.asarray(prob.get_val(chord_var)).ravel()
cl = prob.get_val("CL").item()
cd = prob.get_val("CD").item()
cm = prob.get_val("Cm").item()

print("Hershey bar baseline (alpha = 4 deg)")
print(f"  chords : {chords}")
print(f"  taper  : {chords[-1] / chords[0]:.3f}")
print(f"  CL     : {cl:.4f}")
print(f"  CD     : {cd:.6f}")
print(f"  Cm     : {cm:.4f}")

totals = prob.compute_totals(of=["CL", "CD"], wrt=["alpha", chord_var])
print("  dCL/dalpha  :", totals[("CL", "alpha")].item(), "per deg")
print("  dCD/dalpha  :", totals[("CD", "alpha")].item(), "per deg")
print("  dCL/dchords :", np.asarray(totals[("CL", chord_var)]).ravel())
print()

# --- 3. Chord optimization ----------------------------------------------------
#
# Design variables : section chords and angle of attack (for trim)
# Objective        : minimize CD
# Constraint       : CL = 0.7

target_cl = 0.7
cd_before = cd

prob.model.add_design_var(chord_var, lower=0.15, upper=2.5)
prob.model.add_design_var("alpha", lower=-2.0, upper=12.0, units="deg")
prob.model.add_objective("CD")
prob.model.add_constraint("CL", equals=target_cl)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["maxiter"] = 40
prob.driver.options["disp"] = True

prob.setup()

print(f"Optimizing chords to minimize CD at CL = {target_cl} ...")
prob.run_driver()

chords = np.asarray(prob.get_val(chord_var)).ravel()
cl = prob.get_val("CL").item()
cd = prob.get_val("CD").item()
alpha = prob.get_val("alpha").item()

print()
print("Optimized")
print(f"  chords : {chords}")
print(f"  taper  : {chords[-1] / chords[0]:.3f}")
print(f"  alpha  : {alpha:.2f} deg")
print(f"  CL     : {cl:.4f}")
print(f"  CD     : {cd:.6f}")
print(f"  CD cut : {100 * (cd_before - cd) / cd_before:.1f}%")
