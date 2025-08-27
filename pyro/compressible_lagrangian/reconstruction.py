"""
Reconstruction routines for Lagrangian compressible flow.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def piecewise_linear(q, myg, limiter=2):
    """
    Piecewise linear reconstruction with limiters.

    Parameters:
    -----------
    q : array
        Cell-centered quantity to reconstruct
    myg : Grid2d
        Grid object
    limiter : int
        Limiter type (0=none, 1=minmod, 2=MC, 3=van Leer)
    """

    qx = myg.qx
    qy = myg.qy

    ql = np.zeros((qx, qy))
    qr = np.zeros((qx, qy))

    for i in range(1, qx - 1):
        for j in range(qy):

            # Slopes
            dql = q[i, j] - q[i - 1, j]
            dqr = q[i + 1, j] - q[i, j]

            # Apply limiter
            if limiter == 0:  # No limiter
                dq = 0.5 * (dql + dqr)
            elif limiter == 1:  # Minmod
                if dql * dqr <= 0:
                    dq = 0.0
                else:
                    dq = np.sign(dql) * min(abs(dql), abs(dqr))
            elif limiter == 2:  # Monotonized central (MC)
                if dql * dqr <= 0:
                    dq = 0.0
                else:
                    dqc = 0.5 * (dql + dqr)
                    dq = np.sign(dqc) * min(2 * abs(dql), 2 * abs(dqr), abs(dqc))
            elif limiter == 3:  # van Leer
                if dql * dqr <= 0:
                    dq = 0.0
                else:
                    dq = 2 * dql * dqr / (dql + dqr)
            else:
                dq = 0.0

            # Interface values
            ql[i, j] = q[i, j] - 0.5 * dq
            qr[i, j] = q[i, j] + 0.5 * dq

    return ql, qr