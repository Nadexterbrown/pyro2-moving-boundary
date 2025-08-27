"""
Lagrangian flux calculation for compressible flow.
"""

import numpy as np
from numba import njit
import compressible_lagrangian.riemann as riemann
import compressible_lagrangian.reconstruction as reconstruction


def lagrangian_fluxes(my_data, aux_data, geometry, rp, ivars, solid, tc, dt):
    """
    Compute the fluxes for Lagrangian compressible flow.
    This computes the interface pressures and velocities needed for
    the Lagrangian update and mesh motion.
    """

    tm_flux = tc.timer("fluxes")
    tm_flux.begin()

    myg = my_data.grid

    # Get conserved variables
    U = my_data.data

    # Convert to primitive variables
    gamma = my_data.get_aux("gamma")
    q = my_data.get_var(["density", "x-velocity", "y-velocity", "pressure"])

    rho = q[0]
    u = q[1]
    v = q[2]
    p = q[3]

    # Allocate flux arrays
    flux_x = myg.scratch_array(nvar=ivars.nvar)
    flux_y = myg.scratch_array(nvar=ivars.nvar)

    # Face velocity arrays for mesh motion
    face_vel_x = myg.scratch_array(nvar=1)
    face_vel_y = myg.scratch_array(nvar=1)

    # X-direction
    _compute_lagrangian_fluxes_x(
        rho, u, v, p, gamma, flux_x, face_vel_x,
        myg.ilo, myg.ihi, myg.jlo, myg.jhi, dt, solid, rp)

    # Y-direction
    _compute_lagrangian_fluxes_y(
        rho, u, v, p, gamma, flux_y, face_vel_y,
        myg.ilo, myg.ihi, myg.jlo, myg.jhi, dt, solid, rp)

    tm_flux.end()

    return flux_x, flux_y, face_vel_x, face_vel_y


@njit(cache=True)
def _compute_lagrangian_fluxes_x(rho, u, v, p, gamma, flux_x, face_vel_x,
                                 ilo, ihi, jlo, jhi, dt, solid, rp):
    """Compute x-direction Lagrangian fluxes."""

    for j in range(jlo, jhi + 1):
        for i in range(ilo, ihi + 2):
            # Get left and right states
            rho_l = rho[i - 1, j]
            u_l = u[i - 1, j]
            v_l = v[i - 1, j]
            p_l = p[i - 1, j]

            rho_r = rho[i, j]
            u_r = u[i, j]
            v_r = v[i, j]
            p_r = p[i, j]

            # Solve Riemann problem
            p_star, u_star, v_star = riemann.lagrangian_riemann_x(
                rho_l, u_l, v_l, p_l, rho_r, u_r, v_r, p_r, gamma)

            # Store interface velocity for mesh motion
            face_vel_x[i, j, 0] = u_star

            # Lagrangian "fluxes" are pressure forces
            flux_x[i, j, 0] = 0.0  # mass is conserved by construction
            flux_x[i, j, 1] = p_star  # x-momentum
            flux_x[i, j, 2] = 0.0  # y-momentum (no coupling in 1D slice)
            flux_x[i, j, 3] = p_star * u_star  # energy


@njit(cache=True)
def _compute_lagrangian_fluxes_y(rho, u, v, p, gamma, flux_y, face_vel_y,
                                 ilo, ihi, jlo, jhi, dt, solid, rp):
    """Compute y-direction Lagrangian fluxes."""

    for i in range(ilo, ihi + 1):
        for j in range(jlo, jhi + 2):
            # Get left and right states (in y-direction)
            rho_l = rho[i, j - 1]
            u_l = u[i, j - 1]  # Note: u and v roles
            v_l = v[i, j - 1]
            p_l = p[i, j - 1]

            rho_r = rho[i, j]
            u_r = u[i, j]
            v_r = v[i, j]
            p_r = p[i, j]

            # Solve Riemann problem (with u,v swapped for y-direction)
            p_star, v_star, u_star = riemann.lagrangian_riemann_y(
                rho_l, v_l, u_l, p_l, rho_r, v_r, u_r, p_r, gamma)

            # Store interface velocity for mesh motion
            face_vel_y[i, j, 0] = v_star

            # Lagrangian "fluxes" are pressure forces
            flux_y[i, j, 0] = 0.0  # mass is conserved by construction
            flux_y[i, j, 1] = 0.0  # x-momentum (no coupling in 1D slice)
            flux_y[i, j, 2] = p_star  # y-momentum
            flux_y[i, j, 3] = p_star * v_star  # energy