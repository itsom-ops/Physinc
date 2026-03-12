"""
Physics-aware validation using Pint and SymPy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import sympy as sp
from pint import UnitRegistry

from agents.llm_parser import SimulationManifest
from core.constants import MATERIAL_DATABASE, MaterialProperties


logger = logging.getLogger(__name__)

ureg = UnitRegistry()


class PhysicalConsistencyError(Exception):
    """Raised when a manifest violates basic physical constraints."""


@dataclass
class PhysicsValidatorResult:
    logs: List[str] = field(default_factory=list)


class PhysicsValidator:
    """
    Performs unit-aware and symbolically-informed validation of simulation
    manifests before they are passed to the numerical solver.
    """

    def __init__(self) -> None:
        # Example symbolic expression: thermal diffusivity alpha = k / (rho * cp)
        self._k_sym, self._rho_sym, self._cp_sym = sp.symbols(
            "k rho cp", positive=True, real=True
        )
        self._alpha_expr = self._k_sym / (self._rho_sym * self._cp_sym)

    def _log(self, result: PhysicsValidatorResult, message: str) -> None:
        logger.info(message)
        result.logs.append(message)

    def _get_material(self, manifest: SimulationManifest) -> MaterialProperties:
        key = manifest.material.strip().lower()
        return MATERIAL_DATABASE[key]

    def _validate_temperature(
        self,
        manifest: SimulationManifest,
        material: MaterialProperties,
        result: PhysicsValidatorResult,
    ) -> None:
        temp_quantity = manifest.temp_k * ureg.kelvin
        melting_quantity = material.melting_point_k * ureg.kelvin

        self._log(
            result,
            f"Temperature check: T = {temp_quantity:~P}, "
            f"T_melt({material.name}) = {melting_quantity:~P}",
        )

        if temp_quantity > melting_quantity:
            raise PhysicalConsistencyError(
                f"Requested temperature {temp_quantity:~P} exceeds the "
                f"melting point of {material.name} ({melting_quantity:~P})."
            )

    def _validate_dimensions(
        self,
        dimensions: Dict[str, float],
        result: PhysicsValidatorResult,
    ) -> None:
        length = dimensions["length"] * ureg.meter
        width = dimensions["width"] * ureg.meter
        height = dimensions["height"] * ureg.meter

        volume = length * width * height
        self._log(
            result,
            f"Geometry check: L={length:~P}, W={width:~P}, H={height:~P}, "
            f"V={volume.to(ureg.meter ** 3):~P}",
        )

        if any(x.magnitude <= 0 for x in (length, width, height)):
            raise PhysicalConsistencyError("All geometric dimensions must be > 0.")

    def _symbolic_diffusivity_check(
        self,
        material: MaterialProperties,
        result: PhysicsValidatorResult,
    ) -> None:
        k_val = material.thermal_conductivity_w_m_k
        rho_val = material.density_kg_m3
        cp_val = material.specific_heat_j_kg_k

        alpha_sym = self._alpha_expr.subs(
            {self._k_sym: k_val, self._rho_sym: rho_val, self._cp_sym: cp_val}
        )
        alpha_val = float(alpha_sym.evalf())

        self._log(
            result,
            "Symbolic check: alpha = k / (rho * cp) = "
            f"{alpha_val:.3e} m^2/s for {material.name}",
        )

        if alpha_val <= 0:
            raise PhysicalConsistencyError(
                "Computed thermal diffusivity is non-positive, which is unphysical."
            )

    def validate(self, manifest: SimulationManifest) -> PhysicsValidatorResult:
        """
        Validate the manifest against basic physical constraints and return a
        log of the reasoning steps.
        """
        result = PhysicsValidatorResult()

        material = self._get_material(manifest)
        self._log(
            result,
            f"Material resolved: {material.name} "
            f"(k={material.thermal_conductivity_w_m_k} W/mK, "
            f"rho={material.density_kg_m3} kg/m^3, "
            f"cp={material.specific_heat_j_kg_k} J/kgK, "
            f"T_melt={material.melting_point_k} K)",
        )

        self._validate_temperature(manifest, material, result)
        self._validate_dimensions(manifest.dimensions, result)
        self._symbolic_diffusivity_check(material, result)

        self._log(result, "Manifest passed all physics validation checks.")
        return result

