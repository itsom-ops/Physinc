"""
Logic for quantifying and explaining the 'LLM Physics Gap'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class PhysicsGapResult:
    epsilon_abs: float
    epsilon_relative: float
    correction_report: str
    physics_reliability_score: float


def evaluate_physics_gap(llm_prediction: float, sim_result: float) -> PhysicsGapResult:
    """
    Compare the LLM's intuitive prediction with the numerical simulation.

    Parameters
    ----------
    llm_prediction:
        LLM's predicted maximum temperature (Kelvin).
    sim_result:
        Simulated maximum temperature (Kelvin).
    """
    if sim_result <= 0.0:
        logger.warning(
            "Simulation result is non-positive (%.3f). Using fallback denominator.",
            sim_result,
        )

    epsilon_abs = abs(llm_prediction - sim_result)
    denom = max(abs(sim_result), 1e-9)
    epsilon_relative = epsilon_abs / denom

    logger.info(
        "Evaluating physics gap: T_llm=%.3f, T_sim=%.3f, "
        "epsilon_abs=%.3f, epsilon_rel=%.3f",
        llm_prediction,
        sim_result,
        epsilon_abs,
        epsilon_relative,
    )

    if epsilon_relative > 0.10:
        correction_report = (
            "Significant physics gap detected. The LLM's intuitive prediction "
            "deviates from the numerically simulated result by more than 10%. "
            "A likely cause is that the LLM under-accounted for material-specific "
            "properties such as thermal conductivity, density, and specific heat, "
            "which jointly control the thermal diffusivity and thus the rate of "
            "temperature evolution."
        )
    else:
        correction_report = (
            "The LLM's intuitive prediction is broadly consistent with the "
            "numerical simulation (within 10%). Any remaining discrepancy may be "
            "due to idealized boundary conditions or discretization effects."
        )

    # Map relative error to a reliability score in [0, 100].
    # 0% error -> 100, 10% error -> 70, 50%+ error -> ~0.
    raw_score = 100.0 * max(0.0, 1.0 - 3.0 * epsilon_relative)
    physics_reliability_score = max(0.0, min(100.0, raw_score))

    logger.info(
        "Physics reliability score: %.2f", physics_reliability_score
    )

    return PhysicsGapResult(
        epsilon_abs=epsilon_abs,
        epsilon_relative=epsilon_relative,
        correction_report=correction_report,
        physics_reliability_score=physics_reliability_score,
    )

