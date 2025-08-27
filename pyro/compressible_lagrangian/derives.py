"""
Derived variables for Lagrangian compressible flow.
This matches the interface of compressible.derives.
"""

import numpy as np
import compressible_lagrangian.eos as eos


def derive_primitives(myd, varnames):
    """
    Derive primitive variables from the conserved state.
    """

    # Get the variables we need
    dens = myd.get_var("density")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")
    ener = myd.get_var("energy")

    # Get gamma
    gamma = myd.get_aux("gamma")

    derived = {}

    for v in varnames:
        if v == "velocity":
            # Velocity magnitude
            u = xmom / dens
            v_vel = ymom / dens
            derived[v] = np.sqrt(u ** 2 + v_vel ** 2)

        elif v == "x-velocity":
            derived[v] = xmom / dens

        elif v == "y-velocity":
            derived[v] = ymom / dens

        elif v == "pressure":
            # Compute pressure from conserved variables
            u = xmom / dens
            v_vel = ymom / dens
            e = ener / dens - 0.5 * (u ** 2 + v_vel ** 2)
            derived[v] = eos.pres(gamma, dens, e)

        elif v == "soundspeed":
            # Sound speed
            u = xmom / dens
            v_vel = ymom / dens
            e = ener / dens - 0.5 * (u ** 2 + v_vel ** 2)
            p = eos.pres(gamma, dens, e)
            derived[v] = eos.soundspeed(gamma, dens, p)

        elif v == "Mach":
            # Mach number
            u = xmom / dens
            v_vel = ymom / dens
            e = ener / dens - 0.5 * (u ** 2 + v_vel ** 2)
            p = eos.pres(gamma, dens, e)
            cs = eos.soundspeed(gamma, dens, p)
            vel_mag = np.sqrt(u ** 2 + v_vel ** 2)
            derived[v] = vel_mag / cs

        elif v == "vorticity":
            # Vorticity (curl of velocity)
            u = xmom / dens
            v_vel = ymom / dens

            myg = myd.grid
            vort = myg.scratch_array()

            # Simple finite differences
            vort[myg.ilo:myg.ihi + 1, myg.jlo:myg.jhi + 1] = \
                (v_vel[myg.ilo + 1:myg.ihi + 2, myg.jlo:myg.jhi + 1] -
                 v_vel[myg.ilo - 1:myg.ihi, myg.jlo:myg.jhi + 1]) / (2.0 * myg.dx) - \
                (u[myg.ilo:myg.ihi + 1, myg.jlo + 1:myg.jhi + 2] -
                 u[myg.ilo:myg.ihi + 1, myg.jlo - 1:myg.jhi]) / (2.0 * myg.dy)

            derived[v] = vort

        else:
            derived[v] = None

    return derived