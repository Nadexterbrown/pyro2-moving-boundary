"""
The main simulation module for the Lagrangian compressible hydrodynamics solver.
"""

from __future__ import print_function
import importlib
import numpy as np
import matplotlib.pyplot as plt

import compressible_lagrangian.BC as BC
import compressible_lagrangian.eos as eos
import compressible_lagrangian.derives as derives
import compressible_lagrangian.lagrangian_fluxes as flx
import mesh.boundary as bnd
from simulation_null import NullSimulation, grid_setup, bc_setup
import util.plot_tools as plot_tools
import particles.particles as particles


class Variables(object):
    """
    A container class for easy access to the different compressible
    variables by an integer key. Matches the original compressible solver
    but handles Lagrangian evolution underneath.
    """

    def __init__(self, myd):
        self.nvar = len(myd.names)

        # conserved variables -- we set these when we initialize so
        # they match the CellCenterData2d object
        self.idens = myd.names.index("density")
        self.ixmom = myd.names.index("x-momentum")
        self.iymom = myd.names.index("y-momentum")
        self.iener = myd.names.index("energy")

        # if there are any additional variables, we treat them as
        # passively advected scalars
        self.naux = self.nvar - 4
        if self.naux > 0:
            self.irhox = 4
        else:
            self.irhox = -1

        # primitive variables
        self.nq = 4 + self.naux
        self.irho = 0
        self.iu = 1
        self.iv = 2
        self.ip = 3

        if self.naux > 0:
            self.ix = 4  # advected scalar
        else:
            self.ix = -1


def cons_to_prim(U, gamma, ivars, myg):
    """Convert an input vector of conserved variables to primitive variables."""
    q = myg.scratch_array(nvar=ivars.nq)

    q[:, :, ivars.irho] = U[:, :, ivars.idens]
    q[:, :, ivars.iu] = U[:, :, ivars.ixmom] / U[:, :, ivars.idens]
    q[:, :, ivars.iv] = U[:, :, ivars.iymom] / U[:, :, ivars.idens]

    e = (U[:, :, ivars.iener] -
         0.5 * q[:, :, ivars.irho] * (q[:, :, ivars.iu] ** 2 +
                                      q[:, :, ivars.iv] ** 2)) / q[:, :, ivars.irho]

    q[:, :, ivars.ip] = eos.pres(gamma, q[:, :, ivars.irho], e)

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix + ivars.naux),
                          range(ivars.irhox, ivars.irhox + ivars.naux)):
            q[:, :, nq] = U[:, :, nu] / q[:, :, ivars.irho]

    return q


def prim_to_cons(q, gamma, ivars, myg):
    """Convert an input vector of primitive variables to conserved variables."""
    U = myg.scratch_array(nvar=ivars.nvar)

    U[:, :, ivars.idens] = q[:, :, ivars.irho]
    U[:, :, ivars.ixmom] = q[:, :, ivars.iu] * U[:, :, ivars.idens]
    U[:, :, ivars.iymom] = q[:, :, ivars.iv] * U[:, :, ivars.idens]

    rhoe = eos.rhoe(gamma, q[:, :, ivars.ip])
    U[:, :, ivars.iener] = rhoe + 0.5 * q[:, :, ivars.irho] * (q[:, :, ivars.iu] ** 2 +
                                                               q[:, :, ivars.iv] ** 2)

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix + ivars.naux),
                          range(ivars.irhox, ivars.irhox + ivars.naux)):
            U[:, :, nu] = q[:, :, nq] * q[:, :, ivars.irho]

    return U


class Simulation(NullSimulation):
    """
    The main simulation class for the Lagrangian compressible
    hydrodynamics solver. This follows the same API as the
    original compressible solver.
    """

    def initialize(self, extra_vars=None, ng=4):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """

        my_grid = grid_setup(self.rp, ng=ng)
        my_data = self.data_class(my_grid)

        # define solver specific boundary condition routines
        bnd.define_bc("hse", BC.user, is_solid=False)
        bnd.define_bc("ramp", BC.user, is_solid=False)
        bnd.define_bc("piston", BC.user, is_solid=True)  # Add piston BC

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = bnd.bc_is_solid(bc)

        # density and energy
        my_data.register_var("density", bc)
        my_data.register_var("energy", bc)
        my_data.register_var("x-momentum", bc_xodd)
        my_data.register_var("y-momentum", bc_yodd)

        # any extras?
        if extra_vars is not None:
            for v in extra_vars:
                my_data.register_var(v, bc)

        # store the EOS gamma as an auxiliary quantity so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        my_data.set_aux("grav", self.rp.get_param("compressible_lagrangian.grav", default=0.0))

        my_data.create()
        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            self.particles = particles.Particles(self.cc_data, bc, self.rp)

        # some auxiliary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = self.data_class(my_grid)
        aux_data.register_var("ymom_src", bc_yodd)
        aux_data.register_var("E_src", bc)
        aux_data.create()
        self.aux_data = aux_data

        self.ivars = Variables(my_data)

        # Initialize Lagrangian mesh geometry
        from . import geometry
        self.geom = geometry.LagrangianGeometry(my_grid, self.rp)

        # derived variables
        self.cc_data.add_derived(derives.derive_primitives)

        # initial conditions for the problem
        problem = importlib.import_module("{}.problems.{}".format(
            self.solver_name, self.problem_name))
        problem.init_data(self.cc_data, self.rp)

        # Initialize cell masses for Lagrangian evolution
        self.geom.initialize_cell_masses(self.cc_data.get_var("density"))

        if self.verbose > 0:
            print(my_data)

    def method_compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint. The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        u, v, cs = self.cc_data.get_var(["velocity", "soundspeed"])

        # For Lagrangian, we need current cell sizes
        dx_current = self.geom.get_current_dx()
        dy_current = self.geom.get_current_dy()

        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = dx_current / (abs(u) + cs + 1e-50)
        ytmp = dy_current / (abs(v) + cs + 1e-50)

        self.dt = cfl * float(min(xtmp.min(), ytmp.min()))

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt using Lagrangian approach.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        dens = self.cc_data.get_var("density")
        xmom = self.cc_data.get_var("x-momentum")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")

        grav = self.rp.get_param("compressible_lagrangian.grav", default=0.0)

        myg = self.cc_data.grid

        # Lagrangian flux calculation
        Flux_x, Flux_y, face_vel_x, face_vel_y = flx.lagrangian_fluxes(
            self.cc_data, self.aux_data, self.geom, self.rp,
            self.ivars, self.solid, self.tc, self.dt)

        old_dens = dens.copy()
        old_ymom = ymom.copy()

        # Conservative update in Lagrangian coordinates
        dtdm_x = self.dt / self.geom.get_cell_masses()
        dtdm_y = dtdm_x  # Same cell mass

        # Apply fluxes (these are actually source terms in Lagrangian coordinates)
        for n in range(self.ivars.nvar):
            var = self.cc_data.get_var_by_index(n)
            var.v()[:, :] += \
                dtdm_x * (Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                dtdm_y * (Flux_y.v(n=n) - Flux_y.jp(1, n=n))

        # Update mesh positions using face velocities
        self.geom.update_mesh(face_vel_x, face_vel_y, self.dt)

        # Update density from new cell volumes and conserved mass
        self.geom.update_density(self.cc_data.get_var("density"))

        # gravitational source terms
        if grav != 0.0:
            ymom[:, :] += 0.5 * self.dt * (dens[:, :] + old_dens[:, :]) * grav
            ener[:, :] += 0.5 * self.dt * (ymom[:, :] + old_ymom[:, :]) * grav

        if self.particles is not None:
            self.particles.update_particles(self.dt)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

        tm_evolve.end()

    def dovis(self):
        """
        Do runtime visualization.
        """
        plt.clf()
        plt.rc("font", size=10)

        # we do this even though ivars is in self, so this works when
        # we are plotting from a file
        ivars = Variables(self.cc_data)

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        q = cons_to_prim(self.cc_data.data, gamma, ivars, self.cc_data.grid)

        rho = q[:, :, ivars.irho]
        u = q[:, :, ivars.iu]
        v = q[:, :, ivars.iv]
        p = q[:, :, ivars.ip]
        e = eos.rhoe(gamma, p) / rho

        magvel = np.sqrt(u ** 2 + v ** 2)

        myg = self.cc_data.grid

        fields = [rho, magvel, p, e]
        field_names = [r"$\rho$", r"U", "p", "e"]

        _, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))

        for n, ax in enumerate(axes):
            v = fields[n]
            img = ax.imshow(np.transpose(v.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap=self.cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # needed for PDF rendering
            cb = axes.cbar_axes[n].colorbar(img)
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        if self.particles is not None:
            ax = axes[0]
            particle_positions = self.particles.get_positions()
            # dye particles
            colors = self.particles.get_init_positions()[:, 0]
            # plot particles
            ax.scatter(particle_positions[:, 0],
                       particle_positions[:, 1], s=5, c=colors, alpha=0.8, cmap="Greys")
            ax.set_xlim([myg.xmin, myg.xmax])
            ax.set_ylim([myg.ymin, myg.ymax])

        plt.figtext(0.05, 0.0125, "t = {:10.5g}".format(self.cc_data.t))
        plt.pause(0.001)
        plt.draw()