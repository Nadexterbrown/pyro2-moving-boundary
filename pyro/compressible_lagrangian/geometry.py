"""
Lagrangian mesh geometry management.
"""

import numpy as np


class LagrangianGeometry:
    """
    Manages the moving mesh geometry for Lagrangian simulation.
    """

    def __init__(self, grid, rp):
        """Initialize with grid and runtime parameters."""
        self.grid = grid
        self.rp = rp

        # Store initial face positions
        self.x_faces = np.copy(grid.x1d)
        self.y_faces = np.copy(grid.y1d)

        # Cell masses (conserved in Lagrangian approach)
        self.cell_masses = None

    def initialize_cell_masses(self, density):
        """Initialize cell masses from initial density."""
        myg = self.grid

        # Initial cell volumes
        dx = myg.dx
        dy = myg.dy
        cell_volume = dx * dy

        # Cell masses = density * volume
        self.cell_masses = np.zeros((myg.qx, myg.qy))
        self.cell_masses[myg.ilo:myg.ihi + 1, myg.jlo:myg.jhi + 1] = \
            density[myg.ilo:myg.ihi + 1, myg.jlo:myg.jhi + 1] * cell_volume

    def get_current_dx(self):
        """Get current cell sizes in x-direction."""
        myg = self.grid
        dx_current = np.zeros((myg.qx, myg.qy))

        for i in range(myg.ilo, myg.ihi + 1):
            dx_current[i, :] = self.x_faces[i + 1] - self.x_faces[i]

        return dx_current[myg.ilo:myg.ihi + 1, myg.jlo:myg.jhi + 1]

    def get_current_dy(self):
        """Get current cell sizes in y-direction."""
        myg = self.grid
        dy_current = np.zeros((myg.qx, myg.qy))

        for j in range(myg.jlo, myg.jhi + 1):
            dy_current[:, j] = self.y_faces[j + 1] - self.y_faces[j]

        return dy_current[myg.ilo:myg.ihi + 1, myg.jlo:myg.jhi + 1]

    def get_cell_masses(self):
        """Return conserved cell masses."""
        myg = self.grid
        return self.cell_masses[myg.ilo:myg.ihi + 1, myg.jlo:myg.jhi + 1]

    def update_mesh(self, face_vel_x, face_vel_y, dt):
        """Update face positions using interface velocities."""
        myg = self.grid

        # Update x-face positions
        for i in range(myg.ilo, myg.ihi + 2):
            # Average face velocity in y-direction
            vel_avg = np.mean(face_vel_x[i, myg.jlo:myg.jhi + 1, 0])
            self.x_faces[i] += vel_avg * dt

        # Update y-face positions
        for j in range(myg.jlo, myg.jhi + 2):
            # Average face velocity in x-direction
            vel_avg = np.mean(face_vel_y[myg.ilo:myg.ihi + 1, j, 0])
            self.y_faces[j] += vel_avg * dt

    def update_density(self, density):
        """Update density from conserved mass and new cell volumes."""
        myg = self.grid

        # Compute new cell volumes
        for i in range(myg.ilo, myg.ihi + 1):
            for j in range(myg.jlo, myg.jhi + 1):
                dx = self.x_faces[i + 1] - self.x_faces[i]
                dy = self.y_faces[j + 1] - self.y_faces[j]
                new_volume = dx * dy

                # Update density: rho = mass / volume
                density[i, j] = self.cell_masses[i, j] / new_volume