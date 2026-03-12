## PhysiSync · Prompt-to-Sim Physics Validator

PhysiSync is a neuro-symbolic "Prompt-to-Sim" system built for the IISc Artpark Prompt-to-Sim challenge. It translates natural language engineering prompts into physically validated heat diffusion simulations and quantifies the **LLM Physics Gap** between language-model intuition and numerical reality.

### Core pipeline

1. **LLM Parser (`agents/llm_parser.py`)**
   - Uses Gemini (`gemini-2.5-flash`) to map a prompt like  
     "Simulate a 10cm aluminum cube at 400 Kelvin"  
     into a structured `SimulationManifest` (material, temperature, geometry, intuitive target temperature).
   - Enforces supported materials using `core/constants.py`.
   - Ensures a meaningful `target_prediction`; falls back to `temp_k` if the model fails to provide one.

2. **Physics Validator (`core/physics_validator.py`) – Neuro-symbolic guardrail**
   - Uses **Pint** for unit-aware checks (Kelvin, meters, volume).
   - Uses **SymPy** for a symbolic check of thermal diffusivity  
     $\alpha = \frac{k}{\rho C_p}$ with material parameters from `constants.py`.
   - Verifies:
     - Temperature is below the material’s melting point.
     - All geometric dimensions are positive.
     - Diffusivity is positive and physically plausible.
   - Emits a stepwise log used as the "Thought Process" in the UI.

3. **Numerical Solver (`simulation/heat_solver.py`)**
   - 2D explicit finite-difference heat equation solver:
     \\( \partial_t u = \alpha \nabla^2 u \\).
   - Builds a grid from the manifest’s geometry, applies simple boundary conditions, and marches in time under a CFL stability constraint.
   - Returns the full temperature field and the simulated maximum temperature.

4. **Physics Gap Logic (`core/correction_loop.py`)**
   - Compares LLM-predicted max temperature vs. simulated max temperature.
   - Computes absolute and relative error and a **Physics Reliability Score** (0–100).
   - Generates a correction report explaining likely reasons for disagreement (e.g., missing material properties).

5. **Streamlit UI (`app.py`)**
   - Dark, IISc-style dashboard with:
     - Prompt input and material selection.
     - Stepwise "Physics Workings" log.
     - Plotly 3D surface of the temperature field.
     - Physics Gap metrics + gauge + qualitative interpretation (high / medium / low trust).

### Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Gemini key:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Running the app

From the project root:

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

### How this addresses the "LLM Physics Gap"

- **Neuro-symbolic validation**: Every LLM proposal passes through explicit symbolic and unit-aware checks before any simulation runs.
- **Ground truth via simulation**: A finite-difference solver provides a numerical reference, not just heuristic reasoning.
- **Gap scoring**: The Physics Reliability Score quantifies how much the LLM "hallucinated" relative to the PDE solution.
- **Transparent reasoning**: The UI surfaces the full physics reasoning chain (constraints, formulas, and checks) so judges can see exactly how language, symbols, and numbers interact.

