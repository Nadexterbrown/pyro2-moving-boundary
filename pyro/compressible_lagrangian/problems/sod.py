"""
Sod shock tube problem setup for Lagrangian compressible flow.
"""

import numpy as np


def init_data(my_data, rp):
    """Initialize the Sod shock tube problem."""

    # Default Sod parameters
    rho_left = rp.get_param("sod.rho_left", default=1.0)
    u_left = rp.get_param("sod.u_left", default=0.0)
    p_left = rp.get_param("sod.p_left", default=1.0)

    rho_right = rp.get_param("sod.rho_right", default=0.125)
    u_right = rp.get_param("sod.u_right", default=0.0)
    p_right = rp.get_param("sod.p_right", default=0.1)

    interface = rp.get_param("sod.interface", default=0.5)
    direction = rp.get_param("sod.direction", default="x")

    gamma = rp.get_param("eos.gamma")

    # Get the grid and variables
    myg = my_data.grid

    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # Initialize
    for i in range(myg.ilo, myg.ihi + 1):
        for j in range(myg.jlo, myg.jhi + 1):

            if direction == "x":
                coord = myg.x2d[i, j]
            else:
                coord = myg.y2d[i, j]

            if coord < interface:
                # Left state
                rho = rho_left
                u = u_left if direction == "x" else 0.0
                v = u_left if direction == "y" else 0.0
                p = p_left
            else:
                # Right state
                rho = rho_right
                u = u_right if direction == "x" else 0.0
                v = u_right if direction == "y" else 0.0
                p = p_right

            # Set conserved variables
            dens[i, j] = rho
            xmom[i, j] = rho * u
            ymom[i, j] = rho * v

            # Total energy
            rhoe = p / (gamma - 1.0)
            ke = 0.5 * rho * (u ** 2 + v ** 2)
            ener[i, j] = rhoe + ke


def finalize():
    """Print out any information to the user at the end of the run."""
    print("Sod problem completed")