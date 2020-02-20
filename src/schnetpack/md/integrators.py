"""
Integrators are used to propagate the simulated system in time. SchNetPack
provides two basic types of integrators. The Velocity Verlet integrator is a standard
integrator for a purely classical simulations of the nuclei. The ring polymer molecular dynamics
integrator simulates multiple replicas of the system coupled by harmonic springs and recovers
a certain extent of nuclear quantum effects (e.g. tunneling).
"""
import torch
import numpy as np


from schnetpack.md.utils import NormalModeTransformer, MDUnits


__all__ = ["VelocityVerlet", "RingPolymer"]


class Integrator:
    """
    Basic integrator class template. Uses the typical scheme of propagating
    system momenta in two half steps and system positions in one main step.
    The half steps are defined by default and only the _main_step function
    needs to be specified. Uses atomic time units internally.

    If required, the torch graphs generated by this routine can be detached
    every step via the detach flag.

    Args:
        time_step (float): Integration time step in femto seconds.
        detach (bool): If set to true, torch graphs of the propagation are
                       detached after every step (recommended, due to extreme
                       memory usage). This functionality could in theory be used
                       to formulate differentiable MD.
    """

    def __init__(self, time_step, detach=True, device="cuda"):
        self.time_step = time_step * MDUnits.fs2atu
        self.detach = detach
        self.device = device

    def main_step(self, system):
        """
        Main integration step wrapper routine to make a default detach
        behavior possible. Calls upon _main_step to perform the actual
        propagation of the system.

        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        self._main_step(system)
        if self.detach:
            system.positions = system.positions.detach()
            system.momenta = system.momenta.detach()

    def half_step(self, system):
        """
        Half steps propagating the system momenta according to:

        ..math::
            p = p + \frac{1}{2} F \delta t

        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        system.momenta = system.momenta + 0.5 * system.forces * self.time_step
        if self.detach:
            system.momenta = system.momenta.detach()

    def _main_step(self, system):
        """
        Main integration step to be implemented in derived routines.

        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        raise NotImplementedError


class VelocityVerlet(Integrator):
    """
    Standard velocity Verlet integrator for non ring-polymer simulations.

    Args:
        time_step (float): Integration time step in femto seconds.
    """

    def __init__(self, time_step, device="cuda"):
        super(VelocityVerlet, self).__init__(time_step, device=device)

    def _main_step(self, system):
        """
        Propagate the positions of the system according to:

        ..math::
            q = q + \frac{p}{m} \delta t

        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        system.positions = (
            system.positions + self.time_step * system.momenta / system.masses
        )
        system.positions = system.positions.detach()


class RingPolymer(Integrator):
    """
    Integrator for ring polymer molecular dynamics, as e.g. described in
    [#rpmd1]_

    During the main step, ring polymer positions and momenta are transformed
    from bead to normal mode representation, propagated deterministically and
    then transformed back. Needs the number of beads and the ring polymer
    temperature in order to initialize the propagator matrix. The integrator
    reverts to standard velocity Verlet integration if only one bead is used.

    Uses atomic units of time internally.

    Args:
        n_beads (int): Number of beads in the ring polymer.
        time_step (float): Time step in femto seconds.
        temperature (float): Ring polymer temperature in Kelvin.
        transformation (object): Normal mode transformer class.
        device (str): Device used for computations, default is GPU ('cuda')

    References
    ----------
    .. [#rpmd1] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133, 124105. 2010.
    """

    def __init__(
        self,
        n_beads,
        time_step,
        temperature,
        transformation=NormalModeTransformer,
        device="cuda",
    ):
        super(RingPolymer, self).__init__(time_step, device=device)

        self.n_beads = n_beads

        # Compute the ring polymer frequency
        self.omega = MDUnits.kB * n_beads * temperature / MDUnits.hbar
        self.transformation = transformation(n_beads, device=self.device)

        # Set up omega_normal, the ring polymer frequencies in normal mode
        # representation
        self.omega_normal = (
            2
            * self.omega
            * torch.sin(
                torch.arange(self.n_beads, device=device).float() * np.pi / self.n_beads
            )
        )

        # Initialize the propagator matrices
        self.propagator = self._init_propagator()

    def _init_propagator(self):
        """
        Constructs the ring polymer propagator in normal mode representation
        as for example given in [#rpmd2]_.

        Returns:
            torch.Tensor: Propagator with the dimension n_beads x 2 x 2,
                          where the last two dimensions mix the systems
                          momenta and positions in normal mode representation.

        References
        ----------
        .. [#rpmd2] Ceriotti, Parrinello, Markland, Manolopoulos:
           Efficient stochastic thermostatting of path integral molecular
           dynamics.
           The Journal of Chemical Physics, 133, 124105. 2010.
        """

        # Compute basic terms
        omega_dt = self.omega_normal * self.time_step
        cos_dt = torch.cos(omega_dt)
        sin_dt = torch.sin(omega_dt)

        # Initialize the propagator
        propagator = torch.zeros(self.n_beads, 2, 2, device=self.device)

        # Define the propagator elements, the central normal mode is treated
        # special
        propagator[:, 0, 0] = cos_dt
        propagator[:, 1, 1] = cos_dt
        propagator[:, 0, 1] = -sin_dt * self.omega_normal
        propagator[1:, 1, 0] = sin_dt[1:] / self.omega_normal[1:]

        # Centroid normal mode is special as reverts to standard velocity
        # Verlet for one bead.
        propagator[0, 1, 0] = self.time_step

        # Expand dimensions to avoid broadcasting in main_step
        propagator = propagator[..., None, None, None]

        return propagator

    def _main_step(self, system):
        """
        Main propagation step for ring polymer dynamics. First transforms
        positions and momenta to their normal mode representations,
        then applies the propagator defined above (mixing momenta and
        positions accordingly) and performs a backtransformation to the bead
        momenta and positions afterwards, which are used to update the
        current system state.

        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        # Transform to normal mode representation
        positions_normal = self.transformation.beads2normal(system.positions)
        momenta_normal = self.transformation.beads2normal(system.momenta)

        # Propagate ring polymer
        momenta_normal = (
            self.propagator[:, 0, 0] * momenta_normal
            + self.propagator[:, 0, 1] * positions_normal * system.masses
        )
        positions_normal = (
            self.propagator[:, 1, 0] * momenta_normal / system.masses
            + self.propagator[:, 1, 1] * positions_normal
        )

        # Transform back to bead representation
        system.positions = self.transformation.normal2beads(positions_normal)
        system.momenta = self.transformation.normal2beads(momenta_normal)
