# OpenAVL

A numerically faithful Python port of [AVL (Athena Vortex Lattice)](https://web.mit.edu/drela/Public/web/avl/), Mark Drela's widely-used Fortran aerodynamics solver. OpenAVL reproduces AVL's results to machine precision using pure Python and NumPy, making it scriptable, testable, and composable with the broader scientific Python ecosystem.

[Try the OpenAVL app here](https://openavl.onrender.com/) (the app is very slow due to server cpu limitations)

## What It Does

OpenAVL solves the **vortex lattice method (VLM)** to compute aerodynamic forces and moments on aircraft configurations defined by AVL geometry files (`.avl`). It supports:

- Lift, drag, and side-force coefficients (CL, CD, CY)
- Pitching, rolling, and yawing moment coefficients (Cm, Cl, Cn)
- **Stability derivatives** — partial derivatives of forces/moments with respect to angle of attack, sideslip, rotation rates, and control surface deflections
- **Trim analysis** — Newton-Raphson iteration to satisfy flight constraints (e.g., trim to a target CL)
- **Flight dynamics eigenanalysis** — identifies phugoid, short-period, Dutch-roll, and spiral modes from a linearized state-space model
- Multi-surface configurations: wings, tails, fins, fuselage bodies, control surfaces
- **Geometry API** — build aircraft programmatically (`Aircraft`, `Wing`, `Section`) without `.avl` files
- **Sectional CLmax capping** — approximate stall onset by limiting peak strip lift (Geometry API only)
- Prandtl-Glauert compressibility correction
- CD(CL) viscous drag polars
- **3D geometry preview**, **spanwise lift distribution**, and **Cp** plots via `AVLSolver.plot_aircraft()`, `plot_lift_distribution()`, and `plot_cp()`

## Installation

```bash
git clone https://github.com/danielenriquez59/openavl.git
cd openavl
pip install -e .
```

**Requirements:** Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10, Matplotlib >= 3.7, Numba >= 0.60

Optional extras install with bracket notation. Combine multiple extras as needed (e.g. `pip install -e ".[dev,jax,web]"`).

```bash
pip install -e ".[dev]"         # pytest — run the test suite
pip install -e ".[web]"         # FastAPI + uvicorn — browser-based GUI
pip install -e ".[jax]"         # JAX + jaxlib (CPU) — AD and OpenMDAO integration
pip install -e ".[jax-cuda12]"  # JAX with CUDA 12 GPU support
```

#### Web GUI (`[web]`)

Installs FastAPI, uvicorn, and python-multipart. See [Running the Web GUI Locally](#running-the-web-gui-locally) below.

#### JAX backend (`[jax]`) — validation in progress

Enables `openavl.jax` for reverse-mode automatic differentiation, gradient-based optimization, and OpenMDAO wrappers (see `examples/rectangular_wing_opt.py`). JAX is **not** installed with the base package.

```bash
pip install -e ".[jax]"
```

On Windows, if JAX fails to import, install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) (x64) and try `pip install -e ".[jax-cuda12]"` for CUDA 12 wheels. A standard `venv` (rather than Anaconda) or WSL can help if native wheels remain problematic.

Enable 64-bit floats before running JAX analysis (required for AVL numerics):

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Run JAX tests with `pytest tests/jax_backend`.

## Running the Web GUI Locally

OpenAVL ships a browser-based GUI for loading `.avl` models, editing flight conditions, running the solver, and viewing 3D geometry and results. It wraps the core NumPy solver with a FastAPI backend and is optional — not installed with the base package.

From the repository root (after `pip install -e .`):

```bash
pip install -e ".[web]"
python -m openavl.web
```

This starts a local server at `http://127.0.0.1:8000` and opens it in your default browser. Stop the server with `Ctrl+C`.

The `[web]` extra pulls in FastAPI, uvicorn, and python-multipart (for airfoil file uploads). Run web tests with `pytest tests/ui`.

## Documentation

Full user manual: [docs/user_manual.html](docs/user_manual.html) — API reference, Geometry API, examples, file formats, and module layout.

## Quick Start

```python
from openavl import AVLSolver

# Load an AVL geometry file and set flight conditions
solver = AVLSolver(
    "path/to/aircraft.avl",
    alpha=2.0,        # angle of attack (degrees)
    beta=0.0,         # sideslip angle (degrees)
    vel=64.5,         # airspeed (ft/s or m/s, consistent with geometry units)
    rho=0.0005846,    # air density
    gravity=32.18,    # gravitational acceleration
    cd0=0.00835,      # parasitic drag coefficient
    xcg=0.02463,      # CG x-position (reference frame)
    ycg=0.0,
    zcg=0.2239,
)

solver.execute_run(max_iter=20)
results = solver.get_results()

print(results["CL"])   # lift coefficient
print(results["CD"])   # drag coefficient
print(results["Cm"])   # pitching moment coefficient
```

## Trimmed Flight

Use constraints to trim the aircraft to a target aerodynamic condition:

```python
from openavl import AVLSolver

solver = AVLSolver("aircraft.avl")

# Set flight condition parameters
solver.set_parameter("velocity", 64.5)
solver.set_parameter("density", 0.0005846)
solver.set_parameter("gravity", 32.18)
solver.set_parameter("mach", 0.1)

# Trim alpha to achieve CL = 0.5, and elevator to achieve Cm = 0
solver.set_constraint("alpha", "cl", 0.5)
solver.set_constraint("elevator", "cm", 0.0)

solver.execute_run(max_iter=20)
results = solver.get_results()
```

For coordinated level flight with mass data, use `setup_trim()` to preset longitudinal constraints before adding lateral trim:

```python
solver.set_parameter("cl", 0.7)
solver.setup_trim(mode=1)  # level flight or banked turn
solver.set_constraint("elevator", "cm", 0.0)
solver.execute_run()
```

## Stability Derivatives

```python
solver.execute_run()
derivs = solver.get_stability_derivatives()

print(derivs.CL_a)   # dCL/d(alpha) — lift curve slope
print(derivs.Cm_a)   # dCm/d(alpha) — pitch stiffness
print(derivs.CL_q)   # dCL/d(pitch rate)
print(derivs.Cn_r)   # dCn/d(yaw rate) — yaw damping

# Control surface derivatives (per radian; keyed by surface name)
print(derivs.CL_d["elevator"])
print(derivs.Cm_d["elevator"])
```

## Eigenvalue / Flight Dynamics Analysis

```python
modes = solver.eigenvalues()

for mode in modes.modes:
    print(f"{mode.name}: "
          f"freq={mode.frequency_hz:.3f} Hz, "
          f"damping={mode.damping_ratio:.3f}")
# Example output:
# Short Period: freq=1.842 Hz, damping=0.712
# Phugoid: freq=0.043 Hz, damping=0.041
# Dutch Roll: freq=0.963 Hz, damping=0.187
# Spiral: time_constant=38.4 s
```

## AVL Geometry Files

OpenAVL reads standard `.avl` geometry files unchanged from AVL. The format defines:

- Reference quantities (area, chord, span, reference point)
- Surfaces (wings, tails, fins) with sections and airfoil profiles
- Bodies (fuselages)
- Control surfaces with hinge-line definitions

See the [AVL documentation](https://web.mit.edu/drela/Public/web/avl/avl_doc.txt) for the geometry file format. Example geometries are provided in `tests/data/avl/geometries/`.

## Geometry API

Build aircraft without `.avl` files and pass the result directly to `AVLSolver`:

```python
from openavl import AVLSolver, Aircraft

ac = Aircraft(name="Demo", sref=10.0, cref=1.0, bref=8.0)
wing = ac.add_wing("Wing", n_chord=8, n_span=16, symmetric=True)
wing.clmax = 1.2  # optional: cap sectional lift (0 = disabled)
wing.add_section(xyzle=[0, 0, 0], chord=1.0).set_airfoil_naca("2412")
wing.add_section(xyzle=[0.2, 4, 0], chord=0.6)

solver = AVLSolver(ac, base_dir="/path/to/airfoils", alpha=2.0)
solver.execute_run()
```

See `examples/supra_geomapi_example.py` for a full programmatic reproduction of the bundled Supra sailplane.

## Examples

Five walkthrough scripts in `examples/` analyze the Supra 3.4 m F3J sailplane. Run from the repository root after `pip install -e .`:

| Script | Demonstrates |
|--------|--------------|
| `supra_example.py` | Load `.avl` + `.mass`, trim, stability derivatives, eigenmodes |
| `supra_geomapi_example.py` | Same analysis via the Geometry API (no `.avl` file) |
| `supra_geometry.py` | Shared Supra geometry module imported by other examples |
| `supra_load_factor.py` | V-n maneuver point: per-surface forces, strip lift distribution |
| `supra_aero_sweep.py` | Alpha sweep polars (CL, CD, Cm vs angle of attack) |

```bash
python examples/supra_example.py
python examples/supra_geomapi_example.py
python examples/supra_load_factor.py
python examples/supra_aero_sweep.py
```

## Visualization

```python
solver.plot_aircraft()            # 3D wireframe geometry preview
solver.plot_lift_distribution()   # spanwise cl_lstrp after execute_run()
solver.plot_cp()                  # pressure coefficient on lifting surfaces
```

Lower-level helpers in `openavl.plotting` accept `.avl` paths, `Aircraft`, or `AVLModel` objects directly when you do not have a solver instance.

## Running Tests

```bash
pip install -e ".[dev]"

# Run all tests
pytest

# Run core unit/module tests only
pytest tests/core

# Run end-to-end integration tests
pytest tests/integration

# Run web UI tests (requires optional [web] deps)
pytest tests/ui

# Run JAX tests (requires optional [jax] deps)
pytest tests/jax_backend

# Run by marker (works across all subdirectories)
pytest -m core
pytest -m integration
pytest -m ui

# Run all numerically validated tests (Fortran binaries or AVL run-case refs)
pytest -m reference
```

Test markers:
- `core` — unit and module tests for solver internals and APIs
- `integration` — full end-to-end AVLSolver runs on real geometries
- `ui` — web GUI session and export helpers
- `reference` — numerically validated against Fortran binaries or AVL run-case outputs
- `fixture` — validated against stored JSON fixtures
- `smoke` — quick convergence or structural checks without tight reference tolerances

## Project Structure

```
src/
├── core/       # AVLSolver API, state dataclass, Newton solver, AIC setup
├── aero/       # Force integration, vortex kernels, Trefftz plane, drag polars
├── geom/       # Panel geometry construction, airfoil camber, spacing functions
├── geometry/   # Programmatic Geometry API (Aircraft, Wing, Section, CdclPolar)
├── fileio/     # AVL geometry parser, mass file loader
├── analysis/   # Stability derivatives, trim, eigenvalue analysis
├── plotting/   # 3D geometry preview and spanwise lift distribution plots
└── math/       # LU factorization wrappers, rotation utilities

tests/
├── core/           # Unit and module tests (kernels, geometry, parser, trim, …)
├── integration/    # End-to-end AVLSolver runs on reference geometries
├── ui/             # Web GUI session and export helpers
├── jax_backend/    # JAX differentiable backend (requires pip install -e ".[jax]")
│   ├── core/       # JAX kernels vs NumPy parity (vortex, forces, solve, snapshots)
│   └── integration/ # End-to-end run_analysis, AD validation, OpenMDAO
├── data/avl/       # Reference .avl geometries, mass files, Fortran binaries
└── fixtures/       # JSON golden fixtures from the JS port
```

## Support the Project

[https://buymeacoffee.com/denriquez](https://buymeacoffee.com/denriquez)

## Reference Material

Official AVL homepage and documentation:  
<https://web.mit.edu/drela/Public/web/avl/>

## License

OpenAVL is licensed under the [GNU General Public License v2 (or later)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html). 

As a Python port of AVL's vortex-lattice aerodynamics code, OpenAVL is distributed under the same GPL terms: you may copy, modify, and redistribute it provided derivative works remain under the GPL. See the [full license text](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) for details.
