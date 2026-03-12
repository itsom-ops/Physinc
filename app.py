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

GEMINI_MODEL = "gemini-2.5-flash"


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


def _call_gemini_raw(prompt: str) -> str:
    """
    Lightweight helper for additional reasoning steps (requirements, critique, etc.)
    using the same Gemini API as the main parser.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    import requests

    response = requests.post(
        url, headers=headers, params={"key": api_key}, json=payload, timeout=30
    )
    if response.status_code != 200:
        logger.error(
            "Gemini raw call error: %s - %s",
            response.status_code,
            response.text,
        )
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text}"
        )

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Unexpected Gemini raw response format: %s", data)
        raise RuntimeError("Unexpected Gemini response format") from exc


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

    # High-level capability cards
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            "**Prompt → Simulation Pipelines**  \n"
            "Translate free-form engineering intent into structured manifests and "
            "domain-specific simulation workflows."
        )
    with col_b:
        st.markdown(
            "**Design Exploration Assistant**  \n"
            "Sweep over geometries and boundary conditions to see how design choices "
            "shift temperature fields and reliability."
        )
    with col_c:
        st.markdown(
            "**Result Interpretation & Gap Analysis**  \n"
            "Explain where and why LLM intuition diverges from PDE-based reality via "
            "Physics Gap metrics, reports, and visualizations."
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

        domain_label = st.selectbox(
            "Simulation domain",
            options=[
                "Heat diffusion (PDE-backed)",
                "Conceptual fluid pipeline",
                "Conceptual structural pipeline",
            ],
        )
        if "Heat diffusion" in domain_label:
            domain = "heat_diffusion"
        elif "fluid" in domain_label:
            domain = "conceptual_fluid"
        else:
            domain = "conceptual_structural"

        material_key = st.selectbox(
            "Material",
            options=list(MATERIAL_DATABASE.keys()),
            format_func=lambda k: MATERIAL_DATABASE[k].name,
        )
        st.markdown("---")
        st.markdown(
            "This demo showcases a neuro-symbolic loop:\n\n"
            "- Human intent → LLM-structured manifest\n"
            "- Symbolic + unit-aware physics validation\n"
            "- Domain-specific simulation or conceptual pipeline\n"
            "- Gap analysis between language intuition and numeric reality"
        )

    llm_parser = _init_llm_parser(api_key)
    validator = PhysicsValidator()

    col_prompt, col_meta = st.columns([2, 1])

    with col_prompt:
        if "prompt_text" not in st.session_state:
            st.session_state["prompt_text"] = (
                "Simulate a 10cm aluminum cube at 400 Kelvin with one face held hotter "
                "than the others. I want to know the steady-state max temperature."
            )

        user_prompt = st.text_area(
            "Engineering prompt",
            value=st.session_state["prompt_text"],
            height=140,
        )

    with col_meta:
        st.markdown("#### Run configuration")
        st.write(
            "Once you are happy with the prompt and material, "
            "launch the full Prompt → Physics → Simulation pipeline."
        )

        st.markdown("#### Example intents")
        if st.button("Heat diffusion · cooling plate example"):
            st.session_state[
                "prompt_text"
            ] = "Design a copper plate heat spreader 20cm by 10cm, thickness 5mm, with the left edge held at 360K and the right edge exposed to ambient 300K. Estimate the maximum temperature after steady state."
            st.experimental_rerun()
        if st.button("Fluid · conceptual airfoil pipeline"):
            st.session_state[
                "prompt_text"
            ] = "Set up a conceptual CFD pipeline for flow over a 2D airfoil at 30 m/s in air at 300K. I care about drag and lift coefficients rather than detailed flow fields."
            st.experimental_rerun()
        if st.button("Structural · conceptual bracket analysis"):
            st.session_state[
                "prompt_text"
            ] = "Outline a structural simulation workflow for a steel L-bracket anchored on one face and loaded with 2kN at the free tip. Emphasize where stress concentrations and safety factors will be evaluated."
            st.experimental_rerun()

        multi_agent = st.checkbox(
            "Use multi-stage intent → pipeline reasoning", value=True
        )
        run_clicked = st.button("🚀 Run PhysiSync Simulation", key="run_button")

    if run_clicked:
        if llm_parser is None:
            st.error(
                "LLM parser is not available. Please configure a Gemini API key "
                "via environment / secrets or the sidebar."
            )
            return

        requirements_summary: Optional[str] = None
        strategy_summary: Optional[str] = None

        with st.spinner("Translating intent into a simulation-ready manifest..."):
            try:
                if multi_agent:
                    # Stage 1: extract engineering requirements from the raw prompt.
                    req_prompt = (
                        "You are helping interpret an engineering request before "
                        "setting up a simulation.\n\n"
                        "Summarize the user's intent as structured requirements with "
                        "bullets for:\n"
                        "- Objectives\n"
                        "- Constraints\n"
                        "- Relevant physics and scales\n"
                        "- Any ambiguities that might matter for a simulation.\n\n"
                        f"User request:\n\"\"\"{user_prompt}\"\"\"\n"
                    )
                    requirements_summary = _call_gemini_raw(req_prompt)

                    # Stage 2: propose a simulation strategy / workflow.
                    strat_prompt = (
                        "You are designing a simulation workflow based on interpreted "
                        "requirements.\n\n"
                        f"Domain hint: {domain}\n\n"
                        "Given the requirements below, propose a concrete simulation "
                        "strategy as a short numbered list covering:\n"
                        "- Governing equations\n"
                        "- Discretization / mesh strategy\n"
                        "- Boundary and initial conditions\n"
                        "- Outputs of interest\n"
                        "- Any trade-offs between accuracy and speed.\n\n"
                        f"Requirements:\n{requirements_summary}\n"
                    )
                    strategy_summary = _call_gemini_raw(strat_prompt)

                # Stage 3: build the structured manifest that drives the rest of the system.
                manifest = llm_parser.parse(user_prompt, domain=domain)
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

        # --- Numerical simulation or conceptual pipeline ---
        field = None
        sim_max_temp = 0.0
        if manifest.domain == "heat_diffusion":
            try:
                solver = HeatSolver(manifest=manifest, material=material)
                field, sim_max_temp = solver.run()
            except Exception as exc:  # pragma: no cover - UI/runtime path
                st.error(f"Failed to run simulation: {exc}")
                logger.exception("Simulation error.")
                return

            gap_result = evaluate_physics_gap(
                llm_prediction=manifest.target_prediction,
                sim_result=sim_max_temp,
            )
        else:
            # For conceptual domains we do not yet have a numeric backend.
            gap_result = evaluate_physics_gap(
                llm_prediction=manifest.target_prediction,
                sim_result=manifest.temp_k,
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
        tab_manifest, tab_validation, tab_pipeline, tab_sim, tab_gap = st.tabs(
            [
                "Manifest & Material",
                "Physics Workings",
                "Reasoning Pipeline",
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

            # Show symbolic formula for alpha using material properties.
            alpha_val = (
                material.thermal_conductivity_w_m_k
                / (material.density_kg_m3 * material.specific_heat_j_kg_k)
            )
            st.markdown("#### Symbolic thermal diffusivity")
            st.latex(
                r"\alpha = \frac{k}{\rho C_p} = %.3e \,\mathrm{m^2/s}" % alpha_val
            )

        with tab_pipeline:
            st.subheader("Intent → Pipeline Translation")
            if requirements_summary:
                st.markdown("#### Stage 0 · Requirements extraction")
                st.write(requirements_summary)

            if strategy_summary:
                st.markdown("#### Stage 1 · Simulation strategy")
                st.write(strategy_summary)

            if manifest.domain == "heat_diffusion":
                steps = [
                    "Parse natural language intent into a structured SimulationManifest.",
                    "Resolve material properties (k, ρ, Cₚ) from the materials database.",
                    "Run symbolic and unit-aware checks (melting point, geometry, diffusivity).",
                    "Discretize the domain into a 2D grid and apply boundary conditions.",
                    "Integrate the heat equation forward in time to steady state.",
                    "Post-process the field to extract max temperature and compare to LLM intuition.",
                ]
            elif manifest.domain == "conceptual_fluid":
                steps = [
                    "Parse natural language intent into a conceptual fluid SimulationManifest.",
                    "Identify relevant governing equations (e.g., incompressible Navier–Stokes).",
                    "Propose a mesh and boundary conditions (inlet, outlet, walls).",
                    "Select a numerical scheme (e.g., SIMPLE/PISO) and time-stepping strategy.",
                    "Define key QoIs (drag, lift, pressure drop) for downstream analysis.",
                ]
            else:
                steps = [
                    "Parse natural language intent into a conceptual structural SimulationManifest.",
                    "Identify material model and stress–strain behavior for the component.",
                    "Propose a finite-element mesh and loading/boundary conditions.",
                    "Outline solution steps (assembly, solve, post-process stresses).",
                    "Define safety factors and failure criteria for decision-making.",
                ]

            for i, s in enumerate(steps, start=1):
                st.markdown(f"**Stage {i}.** {s}")

        with tab_sim:
            if manifest.domain == "heat_diffusion" and field is not None:
                st.subheader("Numerical Simulation · 2D Heat Diffusion")
                fig_surface = _plot_temperature_surface(field)
                st.plotly_chart(fig_surface, width="stretch")

                # Additional views for richer understanding
                st.markdown("#### Alternative views")
                ny, nx = field.shape

                # 2D heatmap / contour view
                heatmap_fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=field,
                            colorscale="Viridis",
                            colorbar={"title": "Temperature (K)"},
                        )
                    ]
                )
                heatmap_fig.update_layout(
                    xaxis_title="X index",
                    yaxis_title="Y index",
                    title="2D Temperature Map (top view)",
                    template="plotly_dark",
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                st.plotly_chart(heatmap_fig, width="stretch")

                # 1D slices through the center
                center_y = ny // 2
                center_x = nx // 2
                slice_x = field[center_y, :]
                slice_y = field[:, center_x]

                line_fig = go.Figure()
                line_fig.add_trace(
                    go.Scatter(
                        y=slice_x,
                        mode="lines",
                        name="Centerline along X",
                    )
                )
                line_fig.add_trace(
                    go.Scatter(
                        y=slice_y,
                        mode="lines",
                        name="Centerline along Y",
                    )
                )
                line_fig.update_layout(
                    xaxis_title="Grid index",
                    yaxis_title="Temperature (K)",
                    title="Temperature profiles through the center",
                    template="plotly_dark",
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                st.plotly_chart(line_fig, width="stretch")

                st.markdown("#### Design exploration (what-if)")
                size_scale = st.slider(
                    "Geometry scale factor", min_value=0.5, max_value=2.0, value=1.0, step=0.25
                )
                bc_scale = st.slider(
                    "Boundary temperature scale factor",
                    min_value=0.8,
                    max_value=1.5,
                    value=1.0,
                    step=0.1,
                )
                if st.button("Run parameter sweep", key="parameter_sweep"):
                    scales = [0.5, 1.0, 1.5, 2.0]
                    results = []
                    for s in scales:
                        scaled_dims = {
                            "length": manifest.dimensions["length"] * s,
                            "width": manifest.dimensions["width"] * s,
                            "height": manifest.dimensions["height"] * s,
                        }
                        temp_manifest = manifest.model_copy(
                            update={"dimensions": scaled_dims}
                        )
                        temp_solver = HeatSolver(manifest=temp_manifest, material=material)
                        _, max_t = temp_solver.run()
                        results.append({"size_scale": s, "max_temp_K": max_t})
                    st.write("Max temperature vs geometry scaling:")
                    st.dataframe(results, use_container_width=True)
            else:
                st.subheader("No numeric backend for this domain yet.")
                st.info(
                    "For conceptual fluid and structural domains, Physinc currently "
                    "constructs the reasoning pipeline but does not run a PDE solver. "
                    "The same neuro-symbolic guardrails can be extended to those solvers."
                )

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

            st.markdown("#### Agent critique of pipeline (optional)")
            if st.button("Ask agent for pipeline critique", key="pipeline_critique"):
                try:
                    critique_prompt = (
                        "You are reviewing a Prompt-to-Sim pipeline that combines an "
                        "LLM with physics-based validation and simulation.\n\n"
                        f"User prompt:\n{user_prompt}\n\n"
                        f"Manifest:\n{manifest.model_dump()}\n\n"
                        f"Physics validation log:\n{validation_result.logs}\n\n"
                        f"Simulation max temperature: {sim_max_temp:.3f} K\n"
                        f"LLM prediction: {manifest.target_prediction:.3f} K\n"
                        f"Physics reliability score: {gap_result.physics_reliability_score:.1f}\n\n"
                        "Provide a short critique focusing on:\n"
                        "- Whether the chosen simulation strategy seems appropriate\n"
                        "- Any missing physics or boundary effects\n"
                        "- Suggestions to refine the pipeline for better fidelity or speed.\n"
                    )
                    critique = _call_gemini_raw(critique_prompt)
                    st.write(critique)
                except Exception as exc:  # pragma: no cover - UI/runtime path
                    st.error(f"Failed to get critique from agent: {exc}")
                    logger.exception("Critique error.")


if __name__ == "__main__":
    main()

