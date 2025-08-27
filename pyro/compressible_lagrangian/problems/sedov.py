"""
Sedov blast wave problem for Lagrangian compressible flow.
"""

import numpy as np


def init_data(my_data, rp):
    """Initialize the Sedov blast wave problem."""

    # Problem parameters
    rho_ambient = rp.get_param("sedov.rho_ambient", default=1.0)
    exp_energy = rp.get_param("sedov.exp_energy", default=1.0)
    exp_r = rp.get_param("sedov.exp_r", default=0.0625)
    p_ambient = rp.get_param("sedov.p_ambient", default=1.e-5)

    gamma = rp.get_param("eos.gamma")

    # Get grid
    myg = my_data.grid

    xc_exp = 0.5 * (myg.xmin + myg.xmax)
    yc_exp = 0.5 * (myg.ymin + myg.ymax)

    # Get variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # Compute explosion region volume
    vol_exp = np.pi * exp_r ** 2
    e_exp = exp_energy / (rho_ambient * vol_exp)

    # Initialize
    for i in range(myg.ilo, myg.ihi + 1):
        for j in range(myg.jlo, myg.jhi + 1):

            # Distance from center
            dist = np.sqrt((myg.x2d[i, j] - xc_exp) ** 2 +
                           (myg.y2d[i, j] - yc_exp) ** 2)

            # Set state
            dens[i, j] = rho_ambient
            xmom[i, j] = 0.0
            ymom[i, j] = 0.0

            if dist <= exp_r:
                # Inside explosion region
                rhoe = rho_ambient * e_exp
            else:
                # Ambient
                rhoe = p_ambient / (gamma - 1.0)

            ener[i, j] = rhoe


def finalize():
    """Print out any information to the user at the end of the run."""
    print("Sedov problem completed")