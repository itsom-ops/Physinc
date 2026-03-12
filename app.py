from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from agents.llm_parser import LLMParser
from core.constants import MATERIAL_DATABASE
from core.correction_loop import evaluate_physics_gap
from core.physics_validator import PhysicalConsistencyError, PhysicsValidator
from simulation.heat_solver import HeatSolver


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_api_key() -> Optional[str]:
    """
    Automatically resolve the Gemini API key from (in order of priority):
    1. Streamlit secrets
    2. Environment variables / .env
    """
    # Load from `.env` if present
    load_dotenv()

    # 1) Streamlit secrets (if configured in .streamlit/secrets.toml)
    try:
        secrets = st.secrets
        if "GEMINI_API_KEY" in secrets:
            return str(secrets["GEMINI_API_KEY"])
    except Exception:
        # No secrets configured; fall through to environment variables.
        pass

    # 2) Environment variables
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key

    return None


def _init_llm_parser(api_key: Optional[str]) -> Optional[LLMParser]:
    if not api_key:
        return None
    os.environ["GEMINI_API_KEY"] = api_key
    try:
        return LLMParser()
    except Exception as exc:  # pragma: no cover - UI/runtime path
        logger.error("Failed to initialize LLMParser: %s", exc)
        return None


def _plot_temperature_surface(field: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Surface(
                z=field,
                colorscale="Viridis",
                colorbar={"title": "Temperature (K)"},
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X index",
            yaxis_title="Y index",
            zaxis_title="Temperature (K)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_dark",
        title="2D Heat Diffusion Surface",
    )
    return fig


def _plot_gap_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "lime"},
                "steps": [
                    {"range": [0, 40], "color": "crimson"},
                    {"range": [40, 70], "color": "orange"},
                    {"range": [70, 100], "color": "seagreen"},
                ],
            },
            title={"text": "Physics Reliability Score"},
        )
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=80, b=40))
    return fig


def main() -> None:
    st.set_page_config(
        page_title="PhysiSync – Prompt-to-Sim",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Global styling ---
    st.markdown(
        """
        <style>
        .physi-header {
            background: linear-gradient(90deg, #121826, #1f2937);
            padding: 1.2rem 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #374151;
            box-shadow: 0 18px 45px rgba(0,0,0,0.6);
            margin-bottom: 1.2rem;
        }
        .physi-header h2 {
            color: #e5e7eb;
            margin: 0;
            font-size: 1.6rem;
        }
        .physi-header p {
            color: #9ca3af;
            margin: 0.3rem 0 0 0;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="physi-header">
            <h2>PhysiSync · Prompt-to-Sim Physics Validator</h2>
            <p>Neuro-symbolic loop: language → physics → numerical simulation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Configuration")

        auto_api_key = _resolve_api_key()
        api_key: Optional[str]
        if auto_api_key:
            api_key = auto_api_key
            st.success("Using Gemini API key from environment / secrets.")
        else:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="If you configure GEMINI_API_KEY in a .env file or "
                "Streamlit secrets, this will be picked up automatically.",
            )

        material_key = st.selectbox(
            "Material",
            options=list(MATERIAL_DATABASE.keys()),
            format_func=lambda k: MATERIAL_DATABASE[k].name,
        )
        st.markdown("---")
        st.markdown(
            "This demo showcases a neuro-symbolic loop:\n\n"
            "- LLM intuition → structured manifest\n"
            "- Pint/SymPy physics validation\n"
            "- SciPy/NumPy heat diffusion\n"
            "- Gap analysis and reliability scoring"
        )

    llm_parser = _init_llm_parser(api_key)
    validator = PhysicsValidator()

    col_prompt, col_meta = st.columns([2, 1])

    with col_prompt:
        user_prompt = st.text_area(
            "Engineering prompt",
            value="Simulate a 10cm aluminum cube at 400 Kelvin.",
            height=140,
        )

    with col_meta:
        st.markdown("#### Run configuration")
        st.write(
            "Once you are happy with the prompt and material, "
            "launch the full Prompt → Physics → Simulation pipeline."
        )
        run_clicked = st.button("🚀 Run PhysiSync Simulation", key="run_button")

    if run_clicked:
        if llm_parser is None:
            st.error(
                "LLM parser is not available. Please configure a Gemini API key "
                "via environment / secrets or the sidebar."
            )
            return

        with st.spinner("Parsing prompt with LLM and building manifest..."):
            try:
                manifest = llm_parser.parse(user_prompt)
            except Exception as exc:  # pragma: no cover - UI/runtime path
                st.error(f"Failed to parse prompt: {exc}")
                logger.exception("Error while parsing prompt.")
                return

        material = MATERIAL_DATABASE[material_key]

        # --- Physics validation ---
        try:
            validation_result = validator.validate(manifest)
        except PhysicalConsistencyError as exc:
            st.error(f"Physical consistency error: {exc}")
            return
        except Exception as exc:  # pragma: no cover - UI/runtime path
            st.error(f"Unexpected error during validation: {exc}")
            logger.exception("Validation error.")
            return

        # --- Numerical simulation ---
        try:
            solver = HeatSolver(manifest=manifest, material=material)
            field, sim_max_temp = solver.run()
        except Exception as exc:  # pragma: no cover - UI/runtime path
            st.error(f"Failed to run simulation: {exc}")
            logger.exception("Simulation error.")
            return

        # --- Gap analysis ---
        gap_result = evaluate_physics_gap(
            llm_prediction=manifest.target_prediction,
            sim_result=sim_max_temp,
        )

        # Top-level metrics strip
        st.markdown("### Summary metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LLM predicted max T (K)", f"{manifest.target_prediction:.2f}")
        m2.metric("Simulated max T (K)", f"{sim_max_temp:.2f}")
        m3.metric(
            "Relative gap (%)", f"{gap_result.epsilon_relative * 100.0:.1f}"
        )
        m4.metric(
            "Physics Reliability Score",
            f"{gap_result.physics_reliability_score:.1f}",
        )

        # Detailed views in tabs
        tab_manifest, tab_validation, tab_sim, tab_gap = st.tabs(
            [
                "Manifest & Material",
                "Physics Workings",
                "Simulation View",
                "Gap Analysis",
            ]
        )

        with tab_manifest:
            st.subheader("Parsed Simulation Manifest")
            st.json(manifest.model_dump())

            st.subheader("Material Properties")
            st.json(
                {
                    "name": material.name,
                    "thermal_conductivity_w_m_k": material.thermal_conductivity_w_m_k,
                    "density_kg_m3": material.density_kg_m3,
                    "specific_heat_j_kg_k": material.specific_heat_j_kg_k,
                    "melting_point_k": material.melting_point_k,
                }
            )

        with tab_validation:
            st.subheader("Workings · Physics Validation Log")
            for idx, line in enumerate(validation_result.logs, start=1):
                st.markdown(f"**Step {idx}.** {line}")

        with tab_sim:
            st.subheader("Numerical Simulation · 2D Heat Diffusion")
            fig_surface = _plot_temperature_surface(field)
            st.plotly_chart(fig_surface, width="stretch")

        with tab_gap:
            st.subheader("LLM Physics Gap Analysis")

            col_gap_left, col_gap_right = st.columns([2, 1])

            with col_gap_left:
                st.markdown("#### Gap Metrics")
                st.metric(
                    "LLM predicted max T (K)",
                    f"{manifest.target_prediction:.2f}",
                )
                st.metric("Simulated max T (K)", f"{sim_max_temp:.2f}")
                st.metric("Absolute gap (K)", f"{gap_result.epsilon_abs:.2f}")
                st.metric(
                    "Relative gap (%)",
                    f"{gap_result.epsilon_relative * 100.0:.1f}",
                )
                st.markdown("#### Correction Report")
                st.write(gap_result.correction_report)

            with col_gap_right:
                st.markdown("#### Physics Gap Score")
                gauge_fig = _plot_gap_gauge(gap_result.physics_reliability_score)
                st.plotly_chart(gauge_fig, width="stretch")
                st.caption(
                    "Scores above 80 indicate high agreement between LLM intuition and "
                    "simulation. 40–80 suggests partial agreement. Below 40 the LLM "
                    "is likely hallucinating or ignoring key physics."
                )


if __name__ == "__main__":
    main()

