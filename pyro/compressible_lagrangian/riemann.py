"""
Riemann solvers for Lagrangian compressible flow.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def lagrangian_riemann_x(rho_l, u_l, v_l, p_l, rho_r, u_r, v_r, p_r, gamma):
    """
    Solve Riemann problem in Lagrangian frame for x-direction.
    Returns interface pressure and velocities.
    """

    # Sound speeds
    cs_l = np.sqrt(gamma * p_l / rho_l)
    cs_r = np.sqrt(gamma * p_r / rho_r)

    # Simple HLLC-like solver for Lagrangian frame
    # Wave speed estimates
    S_L = min(u_l - cs_l, u_r - cs_r)
    S_R = max(u_l + cs_l, u_r + cs_r)

    # Contact speed (interface velocity in Lagrangian frame)
    numer = p_r - p_l + rho_l * u_l * (S_L - u_l) - rho_r * u_r * (S_R - u_r)
    denom = rho_l * (S_L - u_l) - rho_r * (S_R - u_r)

    if abs(denom) > 1e-12:
        S_star = numer / denom
    else:
        S_star = 0.5 * (u_l + u_r)

    # Interface pressure
    p_star = p_l + rho_l * (u_l - S_star) * (S_L - u_l)
    p_star = max(p_star, 0.1 * min(p_l, p_r))  # Pressure floor

    # Interface velocities
    u_star = S_star

    # Tangential velocity (contact preserving)
    if S_star >= 0:
        v_star = v_l
    else:
        v_star = v_r

    return p_star, u_star, v_star


@njit(cache=True)
def lagrangian_riemann_y(rho_l, v_l, u_l, p_l, rho_r, v_r, u_r, p_r, gamma):
    """
    Solve Riemann problem in Lagrangian frame for y-direction.
    Returns interface pressure and velocities.
    """

    # This is similar to x-direction but with v as normal velocity
    cs_l = np.sqrt(gamma * p_l / rho_l)
    cs_r = np.sqrt(gamma * p_r / rho_r)

    S_L = min(v_l - cs_l, v_r - cs_r)
    S_R = max(v_l + cs_l, v_r + cs_r)

    numer = p_r - p_l + rho_l * v_l * (S_L - v_l) - rho_r * v_r * (S_R - v_r)
    denom = rho_l * (S_L - v_l) - rho_r * (S_R - v_r)

    if abs(denom) > 1e-12:
        S_star = numer / denom
    else:
        S_star = 0.5 * (v_l + v_r)

    p_star = p_l + rho_l * (v_l - S_star) * (S_L - v_l)
    p_star = max(p_star, 0.1 * min(p_l, p_r))

    v_star = S_star

    if S_star >= 0:
        u_star = u_l
    else:
        u_star = u_r

    return p_star, v_star, u_star