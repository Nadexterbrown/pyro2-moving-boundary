"""
Equation of state for Lagrangian compressible flow.
This provides the same interface as compressible.eos.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def pres(gamma, rho, e):
    """
    Given the density and the specific internal energy, return the
    pressure
    """
    return rho * e * (gamma - 1.0)


@njit(cache=True)
def dens(gamma, p, e):
    """
    Given the pressure and the specific internal energy, return the
    density
    """
    return p / (e * (gamma - 1.0))


@njit(cache=True)
def rhoe(gamma, p):
    """
    Given the pressure, return (rho * e)
    """
    return p / (gamma - 1.0)


@njit(cache=True)
def h_from_eps(gamma, e):
    """
    Given the specific internal energy, return the specific enthalpy
    """
    return gamma * e


@njit(cache=True)
def soundspeed(gamma, rho, p):
    """
    Given the density and pressure, return the sound speed
    """
    return np.sqrt(gamma * p / rho)