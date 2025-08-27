"""
Boundary conditions for Lagrangian compressible flow.
"""

import numpy as np
import compressible_lagrangian.riemann as riemann


def user(bc_name, bc_edge, variable, my_data):
    """
    User boundary condition dispatcher.
    """

    if bc_name == "piston":
        piston(bc_edge, variable, my_data)
    elif bc_name == "hse":
        # Use standard compressible BC for hydrostatic equilibrium
        from compressible import BC as comp_BC
        comp_BC.user(bc_name, bc_edge, variable, my_data)
    elif bc_name == "ramp":
        # Use standard compressible BC for ramp
        from compressible import BC as comp_BC
        comp_BC.user(bc_name, bc_edge, variable, my_data)
    else:
        raise ValueError(f"Unknown boundary condition: {bc_name}")


def piston(bc_edge, variable, my_data):
    """
    Piston boundary condition - moving wall.
    """

    myg = my_data.grid

    # Piston velocity (could be time-dependent)
    u_piston = 0.1  # m/s - could read from runtime parameters

    if bc_edge == "xleft":
        _fill_piston_x_left(variable, my_data, u_piston)
    elif bc_edge == "xright":
        _fill_piston_x_right(variable, my_data, u_piston)
    elif bc_edge == "yleft":
        _fill_piston_y_left(variable, my_data, u_piston)
    elif bc_edge == "yright":
        _fill_piston_y_right(variable, my_data, u_piston)


def _fill_piston_x_left(variable, my_data, u_piston):
    """Fill left x-boundary for piston."""

    myg = my_data.grid
    var = my_data.get_var(variable)

    if variable == "density":
        # Density symmetric
        for i in range(myg.ilo):
            var[i, :] = var[2 * myg.ilo - 1 - i, :]

    elif variable == "x-momentum":
        # Momentum antisymmetric about wall velocity
        rho = my_data.get_var("density")
        for i in range(myg.ilo):
            interior_u = var[2 * myg.ilo - 1 - i, :] / rho[2 * myg.ilo - 1 - i, :]
            ghost_u = 2 * u_piston - interior_u
            var[i, :] = ghost_u * rho[i, :]

    elif variable == "y-momentum":
        # Tangential momentum symmetric
        for i in range(myg.ilo):
            var[i, :] = var[2 * myg.ilo - 1 - i, :]

    elif variable == "energy":
        # Energy from wall Riemann solution
        _fill_energy_piston_x_left(my_data, u_piston)


def _fill_piston_x_right(variable, my_data, u_piston):
    """Fill right x-boundary for piston."""

    myg = my_data.grid
    var = my_data.get_var(variable)

    if variable == "density":
        for i in range(myg.ihi + 1, myg.qx):
            var[i, :] = var[2 * myg.ihi + 1 - i, :]

    elif variable == "x-momentum":
        rho = my_data.get_var("density")
        for i in range(myg.ihi + 1, myg.qx):
            interior_u = var[2 * myg.ihi + 1 - i, :] / rho[2 * myg.ihi + 1 - i, :]
            ghost_u = 2 * u_piston - interior_u
            var[i, :] = ghost_u * rho[i, :]

    elif variable == "y-momentum":
        for i in range(myg.ihi + 1, myg.qx):
            var[i, :] = var[2 * myg.ihi + 1 - i, :]

    elif variable == "energy":
        _fill_energy_piston_x_right(my_data, u_piston)


def _fill_piston_y_left(variable, my_data, v_piston):
    """Fill left y-boundary for piston."""

    myg = my_data.grid
    var = my_data.get_var(variable)

    if variable == "density":
        for j in range(myg.jlo):
            var[:, j] = var[:, 2 * myg.jlo - 1 - j]

    elif variable == "y-momentum":
        rho = my_data.get_var("density")
        for j in range(myg.jlo):
            interior_v = var[:, 2 * myg.jlo - 1 - j] / rho[:, 2 * myg.jlo - 1 - j]
            ghost_v = 2 * v_piston - interior_v
            var[:, j] = ghost_v * rho[:, j]

    elif variable == "x-momentum":
        for j in range(myg.jlo):
            var[:, j] = var[:, 2 * myg.jlo - 1 - j]

    elif variable == "energy":
        _fill_energy_piston_y_left(my_data, v_piston)


def _fill_piston_y_right(variable, my_data, v_piston):
    """Fill right y-boundary for piston."""

    myg = my_data.grid
    var = my_data.get_var(variable)

    if variable == "density":
        for j in range(myg.jhi + 1, myg.qy):
            var[:, j] = var[:, 2 * myg.jhi + 1 - j]

    elif variable == "y-momentum":
        rho = my_data.get_var("density")
        for j in range(myg.jhi + 1, myg.qy):
            interior_v = var[:, 2 * myg.jhi + 1 - j] / rho[:, 2 * myg.jhi + 1 - j]
            ghost_v = 2 * v_piston - interior_v
            var[:, j] = ghost_v * rho[:, j]

    elif variable == "x-momentum":
        for j in range(myg.jhi + 1, myg.qy):
            var[:, j] = var[:, 2 * myg.jhi + 1 - j]

    elif variable == "energy":
        _fill_energy_piston_y_right(my_data, v_piston)


def _fill_energy_piston_x_left(my_data, u_piston):
    """Fill energy ghost cells for x-left piston boundary."""

    myg = my_data.grid
    gamma = my_data.get_aux("gamma")

    # Get interior state
    rho_int = my_data.get_var("density")[myg.ilo, :]
    u_int = (my_data.get_var("x-momentum")[myg.ilo, :] /
             my_data.get_var("density")[myg.ilo, :])
    v_int = (my_data.get_var("y-momentum")[myg.ilo, :] /
             my_data.get_var("density")[myg.ilo, :])
    E_int = my_data.get_var("energy")[myg.ilo, :]

    # Compute interior pressure
    e_int = E_int / rho_int - 0.5 * (u_int ** 2 + v_int ** 2)
    p_int = (gamma - 1) * rho_int * e_int

    # Solve wall Riemann problem
    p_wall = _solve_wall_riemann(rho_int, u_int, p_int, u_piston, gamma)

    # Ghost state energy
    rho_ghost = rho_int  # Density is symmetric
    u_ghost = u_piston
    v_ghost = v_int  # Tangential velocity symmetric

    e_ghost = p_wall / ((gamma - 1) * rho_ghost)
    E_ghost = rho_ghost * (e_ghost + 0.5 * (u_ghost ** 2 + v_ghost ** 2))

    # Fill ghost cells
    energy = my_data.get_var("energy")
    for i in range(myg.ilo):
        energy[i, :] = E_ghost


def _fill_energy_piston_x_right(my_data, u_piston):
    """Fill energy ghost cells for x-right piston boundary."""

    myg = my_data.grid
    gamma = my_data.get_aux("gamma")

    # Get interior state
    rho_int = my_data.get_var("density")[myg.ihi, :]
    u_int = (my_data.get_var("x-momentum")[myg.ihi, :] /
             my_data.get_var("density")[myg.ihi, :])
    v_int = (my_data.get_var("y-momentum")[myg.ihi, :] /
             my_data.get_var("density")[myg.ihi, :])
    E_int = my_data.get_var("energy")[myg.ihi, :]

    # Compute interior pressure
    e_int = E_int / rho_int - 0.5 * (u_int ** 2 + v_int ** 2)
    p_int = (gamma - 1) * rho_int * e_int

    # Solve wall Riemann problem
    p_wall = _solve_wall_riemann(rho_int, u_int, p_int, u_piston, gamma)

    # Ghost state energy
    rho_ghost = rho_int
    u_ghost = u_piston
    v_ghost = v_int

    e_ghost = p_wall / ((gamma - 1) * rho_ghost)
    E_ghost = rho_ghost * (e_ghost + 0.5 * (u_ghost ** 2 + v_ghost ** 2))

    # Fill ghost cells
    energy = my_data.get_var("energy")
    for i in range(myg.ihi + 1, myg.qx):
        energy[i, :] = E_ghost


def _fill_energy_piston_y_left(my_data, v_piston):
    """Fill energy ghost cells for y-left piston boundary."""
    # Similar to x-left but for y-direction
    # Implementation analogous to _fill_energy_piston_x_left
    pass


def _fill_energy_piston_y_right(my_data, v_piston):
    """Fill energy ghost cells for y-right piston boundary."""
    # Similar to x-right but for y-direction
    # Implementation analogous to _fill_energy_piston_x_right
    pass


def _solve_wall_riemann(rho, u, p, u_wall, gamma):
    """
    Solve 1D Riemann problem against a moving wall.
    Returns the wall pressure.
    """

    cs = np.sqrt(gamma * p / rho)
    Z = rho * cs  # Acoustic impedance

    # Acoustic approximation for wall pressure
    p_wall = p + Z * (u - u_wall)

    return np.maximum(p_wall, 0.1 * p)  # Pressure floor