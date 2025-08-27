# pyro/mesh/face_motion.py
"""
Registry for moving-face speeds, analogous to boundary.define_bc.

Usage:
    from pyro.mesh.face_motion import define_face_speed
    define_face_speed("piston", my_speed_fn)

The solver will look up the function by the boundary name you already set
(e.g., mesh.xlboundary = "piston") and call it ONLY on boundary faces.

Function signature expected (fully general):
    S_f = my_speed_fn(t, edge, i, j, x_face, y_face)

where:
    t       : current time [s]
    edge    : "xlb","xrb","ylb","yrb"  (x-left/right, y-bottom/top)
    i, j    : face-line indices
    x_face  : x-coordinate of face center
    y_face  : y-coordinate of face center

Return the face normal speed S_f [m/s]. Return 0.0 for stationary faces.
"""

from typing import Callable, Dict

_REGISTRY: Dict[str, Callable] = {}

def define_face_speed(name: str, fn: Callable):
    """
    Register a speed function under boundary name `name`.

    Expected signature:
        S_f = fn(t, edge, i, j, x_face, y_face)
    """
    _REGISTRY[name.lower()] = fn

def get_face_speed(name: str) -> Callable:
    """
    Fetch the function by boundary name. If none is registered,
    return a stationary-face function that always yields 0.0 m/s.
    """
    def _stationary(_t, _edge, _i, _j, _xf, _yf) -> float:
        return 0.0

    if not name:
        return _stationary
    return _REGISTRY.get(str(name).lower(), _stationary)