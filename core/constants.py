"""
Physical constants and material properties for PhysiSync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MaterialProperties:
    name: str
    thermal_conductivity_w_m_k: float  # W / (m*K)
    density_kg_m3: float  # kg / m^3
    specific_heat_j_kg_k: float  # J / (kg*K)
    melting_point_k: float  # Kelvin


MATERIAL_DATABASE: Dict[str, MaterialProperties] = {
    "copper": MaterialProperties(
        name="Copper",
        thermal_conductivity_w_m_k=401.0,
        density_kg_m3=8960.0,
        specific_heat_j_kg_k=385.0,
        melting_point_k=1357.77,
    ),
    "aluminum": MaterialProperties(
        name="Aluminum",
        thermal_conductivity_w_m_k=237.0,
        density_kg_m3=2700.0,
        specific_heat_j_kg_k=897.0,
        melting_point_k=933.47,
    ),
    "steel": MaterialProperties(
        name="Steel",
        thermal_conductivity_w_m_k=50.0,
        density_kg_m3=7850.0,
        specific_heat_j_kg_k=486.0,
        melting_point_k=1811.0,
    ),
}

