"""Shared Supra 3.4 m F3J geometry built with the OpenAVL Geometry API."""

from pathlib import Path

from openavl import Aircraft

repo_root = Path(__file__).resolve().parents[1]
geometries_dir = repo_root / "tests" / "data" / "avl" / "geometries"
supra_mass = repo_root / "tests" / "data" / "avl" / "mass" / "supra.mass"


def format_vector(values, precision=3):
    """Format a three-component vector for compact example output."""
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def add_flaperon(section):
    """Attach flap and aileron controls with AVL control mixing on one hinge.

    Both controls share the same hinge line (``xhinge=0.75``). AVL sums their
    contributions at each section::

        delta_local = gain_flap * flap + gain_aileron * aileron

    ``flap`` uses ``sgn_dup=+1`` so both wing halves deflect together;
    ``aileron`` uses ``sgn_dup=-1`` so the mirrored half deflects in
    opposition. Together this is a flaperon-style mixer: one surface,
    two independent trim variables for symmetric and differential motion.
    """
    section.add_control("flap", gain=1.0, xhinge=0.75, sgn_dup=1.0)
    section.add_control("aileron", gain=-1.0, xhinge=0.75, sgn_dup=-1.0)


def build_supra_aircraft():
    """Build a programmatic equivalent of tests/data/avl/geometries/supra.avl."""
    aircraft = Aircraft(
        name="Supra 3.4m F3J",
        mach=0.0,
        sref=1034.0,
        cref=7.60,
        bref=133.86,
        xref=3.750,
        yref=0.0,
        zref=1.5,
    )

    aircraft.add_body(
        "Fuse pod",
        n_body=28,
        b_space=2.0,
        body_file="fuseSupra.dat",
        translate=[0.0, 0.0, -1.75],
    )

    inner = aircraft.add_wing(
        "Inner Wing",
        n_chord=7,
        c_space=1.0,
        n_span=8,
        s_space=-2.9,
        symmetric=True,
        angle=1.0,
        scale=[1.0, 1.0, 0.0437],
        component=1,
    )
    # inner.clmax = 1.2 # example of using clmax on a wing
    root = inner.add_section(xyzle=[0.0, 0.0, 0.0], chord=9.75, n_span=1, s_space=0.0)
    root.set_airfoil_file("ag40d.dat")
    add_flaperon(root)
    break_ = inner.add_section(xyzle=[0.25, 31.5, 31.5], chord=8.75, n_span=1, s_space=0.0)
    break_.set_airfoil_file("ag41d.dat")
    add_flaperon(break_)

    outer = aircraft.add_wing(
        "Outer Wing",
        n_chord=7,
        c_space=1.0,
        n_span=18,
        s_space=-2.0,
        symmetric=True,
        angle=1.0,
        scale=[1.0, 1.0, 0.13165],
        translate=[0.25, 31.5, 1.37655],
        component=1,
    )
    # outer.clmax = 1.35 # example of using clmax on a wing
    s1 = outer.add_section(xyzle=[0.0, 0.0, 0.0], chord=8.75, n_span=1, s_space=0.0)
    s1.set_airfoil_file("ag41d.dat")
    add_flaperon(s1)
    s2 = outer.add_section(xyzle=[1.0, 23.5, 23.5], chord=6.25, ainc=-0.5, n_span=1, s_space=0.0)
    s2.set_airfoil_file("ag42d.dat")
    add_flaperon(s2)
    s3 = outer.add_section(xyzle=[1.72, 29.5, 29.5], chord=5.00, ainc=-0.5, n_span=1, s_space=0.0)
    s3.set_airfoil_file("ag42d.dat")
    add_flaperon(s3)
    s4 = outer.add_section(xyzle=[2.75, 34.0, 34.0], chord=3.40, ainc=-0.5, n_span=1, s_space=0.0)
    s4.set_airfoil_file("ag43d.dat")
    add_flaperon(s4)
    s5 = outer.add_section(xyzle=[3.50, 35.50, 35.50], chord=2.3, ainc=-0.5, n_span=1, s_space=0.0)
    s5.set_airfoil_file("ag43d.dat")
    add_flaperon(s5)


    stab = aircraft.add_wing(
        "Stab",
        n_chord=5,
        c_space=1.0,
        n_span=12,
        s_space=-1.0,
        symmetric=True,
        translate=[37.5, 0.0, 2.1],
    )
    stab_sections = [
        ([0.0, 0.0, 0.0], 4.40),
        ([0.15385, 2.0, 0.0], 4.1154),
        ([0.7692, 10.0, 0.0], 2.577),
        ([1.173, 12.0, 0.0], 1.942),
        ([1.50, 12.7, 0.0], 1.52),
        ([2.00, 13.0, 0.0], 1.0),
    ]
    for xyzle, chord in stab_sections:
        sec = stab.add_section(xyzle=xyzle, chord=chord)
        sec.add_control("elevator", gain=1.0, xhinge=0.0, sgn_dup=1.0)

    fin = aircraft.add_wing(
        "Fin",
        n_chord=10,
        c_space=1.0,
        n_span=12,
        s_space=-1.0,
        scale=[1.15, 1.15, 1.1],
        translate=[42.5, 0.0, 0.0],
    )
    fin_sections = [
        ([0.0, 0.0, 0.0], 7.0, 0.43),
        ([1.125, 0.0, 9.0], 4.0, 0.50),
        ([1.875, 0.0, 11.25], 2.8333, 0.50),
        ([2.5, 0.0, 12.0], 2.0, 0.50),
    ]
    for xyzle, chord, xhinge in fin_sections:
        sec = fin.add_section(xyzle=xyzle, chord=chord)
        sec.add_control("rudder", gain=1.0, xhinge=xhinge, sgn_dup=1.0)

    return aircraft
