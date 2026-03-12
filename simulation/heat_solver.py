"""
2D heat diffusion solver using an explicit finite difference scheme.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from agents.llm_parser import SimulationManifest
from core.constants import MaterialProperties


logger = logging.getLogger(__name__)


@dataclass
class HeatSolverConfig:
    nx: int = 50
    ny: int = 50
    total_time_s: float = 10.0
    log_every_n_steps: int = 10


class HeatSolver:
    """
    Simple 2D heat equation solver:

        du/dt = alpha * (d2u/dx2 + d2u/dy2)

    using an explicit finite-difference method on a uniform grid.
    """

    def __init__(
        self,
        manifest: SimulationManifest,
        material: MaterialProperties,
        config: HeatSolverConfig | None = None,
    ) -> None:
        self.manifest = manifest
        self.material = material
        self.config = config or HeatSolverConfig()

        self.alpha: float = (
            material.thermal_conductivity_w_m_k
            / (material.density_kg_m3 * material.specific_heat_j_kg_k)
        )

        if self.alpha <= 0:
            raise ValueError("Thermal diffusivity alpha must be positive.")

        logger.info("Initialized HeatSolver with alpha=%g", self.alpha)

    def _stable_timestep(self, dx: float, dy: float) -> float:
        """
        Compute a stable time step for the explicit scheme (CFL condition).
        """
        return (dx * dx * dy * dy) / (2.0 * self.alpha * (dx * dx + dy * dy))

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Run the simulation and return:
        - final 2D temperature field (numpy array, Kelvin)
        - simulation_max_temp (float, Kelvin)
        """
        dims = self.manifest.dimensions
        lx = dims["length"]
        ly = dims["width"]

        nx, ny = self.config.nx, self.config.ny
        dx = lx / (nx - 1)
        dy = ly / (ny - 1)

        dt = self._stable_timestep(dx, dy)
        n_steps = int(self.config.total_time_s / dt)
        if n_steps < 1:
            n_steps = 1

        logger.info(
            "Running heat diffusion: nx=%d, ny=%d, dt=%.3e, steps=%d",
            nx,
            ny,
            dt,
            n_steps,
        )

        # Initialize temperature field
        u = np.full((ny, nx), self.manifest.temp_k, dtype=float)

        # Example boundary condition: left boundary slightly hotter
        u[:, 0] = self.manifest.temp_k * 1.1

        alpha_dt_dx2 = self.alpha * dt / (dx * dx)
        alpha_dt_dy2 = self.alpha * dt / (dy * dy)

        for step in range(n_steps):
            u_new = u.copy()

            # interior points
            u_new[1:-1, 1:-1] = (
                u[1:-1, 1:-1]
                + alpha_dt_dx2
                * (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, 0:-2])
                + alpha_dt_dy2
                * (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[0:-2, 1:-1])
            )

            # Dirichlet boundary conditions: keep fixed at initial values
            u_new[0, :] = u[0, :]
            u_new[-1, :] = u[-1, :]
            u_new[:, 0] = u[:, 0]
            u_new[:, -1] = u[:, -1]

            u = u_new

            if step % self.config.log_every_n_steps == 0:
                max_temp = float(u.max())
                logger.debug(
                    "Step %d / %d, max temp = %.3f K", step, n_steps, max_temp
                )

        simulation_max_temp = float(u.max())
        logger.info("Simulation complete. max_temp=%.3f K", simulation_max_temp)

        return u, simulation_max_temp

