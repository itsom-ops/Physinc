"""
LLM-powered parser that converts natural language prompts into a structured
simulation manifest.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Literal

import requests
from pydantic import BaseModel, Field, ValidationError, validator

from core.constants import MATERIAL_DATABASE, MaterialProperties


logger = logging.getLogger(__name__)


class SimulationManifest(BaseModel):
    """
    Structured representation of a simulation request.
    """

    domain: Literal["heat_diffusion", "conceptual_fluid", "conceptual_structural"] = Field(
        "heat_diffusion",
        description="Simulation domain: heat_diffusion, conceptual_fluid, or conceptual_structural.",
    )
    material: str = Field(..., description="Name of the material, e.g. 'aluminum'.")
    temp_k: float = Field(
        ..., description="Initial or characteristic temperature in Kelvin."
    )
    dimensions: Dict[str, float] = Field(
        ...,
        description=(
            "Object dimensions in meters with keys: length, width, height. "
            "Example: {'length': 0.1, 'width': 0.1, 'height': 0.1}"
        ),
    )
    target_prediction: float = Field(
        ...,
        description=(
            "Intuitive LLM prediction for the final maximum temperature in Kelvin."
        ),
    )

    @validator("material")
    def validate_material_supported(cls, value: str) -> str:
        key = value.strip().lower()
        if key not in MATERIAL_DATABASE:
            raise ValueError(
                f"Unsupported material '{value}'. "
                f"Supported materials: {', '.join(MATERIAL_DATABASE.keys())}"
            )
        return key

    @validator("dimensions")
    def validate_dimensions_keys(cls, value: Dict[str, float]) -> Dict[str, float]:
        required_keys = {"length", "width", "height"}
        missing = required_keys - set(value.keys())
        if missing:
            raise ValueError(f"Missing dimension keys: {', '.join(sorted(missing))}")
        for k, v in value.items():
            if v <= 0:
                raise ValueError(f"Dimension '{k}' must be positive, got {v}.")
        return value


class LLMParser:
    """
    Parser that uses the Gemini HTTP API directly to map a natural language
    description to a validated `SimulationManifest`.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
    ) -> None:
        self._model_name = model_name
        self._api_key_env = api_key_env

    @staticmethod
    def resolve_material(manifest: SimulationManifest) -> MaterialProperties:
        """
        Look up material properties from the manifest.
        """
        return MATERIAL_DATABASE[manifest.material]

    def _build_prompt(self, user_prompt: str, domain: str) -> str:
        supported_materials = ", ".join(MATERIAL_DATABASE.keys())
        schema_description = json.dumps(
            {
                "domain": "string, one of: heat_diffusion, conceptual_fluid, conceptual_structural",
                "material": "string, one of: " + supported_materials,
                "temp_k": "float, temperature in Kelvin",
                "dimensions": {
                    "length": "float, meters",
                    "width": "float, meters",
                    "height": "float, meters",
                },
                "target_prediction": "float, predicted final maximum temperature in Kelvin",
            },
            indent=2,
        )
        return (
            "You are an expert computational physicist and simulation engineer. "
            "Your job is to translate a natural language engineering request into "
            "a structured simulation manifest.\n\n"
            "Task:\n"
            f"- Read the user's request and emit a SINGLE JSON object that conforms to "
            "the schema below.\n"
            "- Interpret any centimeter dimensions and convert them to meters.\n"
            f"- The `material` field MUST be one of: {supported_materials}.\n"
            "- The `domain` field MUST be one of: "
            "'heat_diffusion', 'conceptual_fluid', 'conceptual_structural'.\n"
            f"- Prefer the domain '{domain}' when it matches the user's intent.\n\n"
            "JSON schema (described in natural language):\n"
            f"{schema_description}\n\n"
            "User request:\n"
            f"\"\"\"{user_prompt}\"\"\"\n\n"
            "Respond with ONLY the JSON object, no markdown, no commentary."
        )

    def _call_gemini(self, prompt: str) -> str:
        api_key = os.getenv(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self._api_key_env} is not set. Please configure your Gemini API key."
            )

        # Use the stable v1 Gemini endpoint for the public developer API.
        url = (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"{self._model_name}:generateContent"
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

        response = requests.post(url, headers=headers, params={"key": api_key}, json=payload, timeout=30)
        if response.status_code != 200:
            logger.error(
                "Gemini API error: %s - %s",
                response.status_code,
                response.text,
            )
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected Gemini response format: %s", data)
            raise RuntimeError("Unexpected Gemini response format") from exc

        return text

    def parse(self, user_prompt: str, domain: str = "heat_diffusion") -> SimulationManifest:
        """
        Parse a natural-language prompt into a `SimulationManifest` using Gemini.
        """
        logger.info(
            "Parsing user prompt into SimulationManifest via Gemini HTTP API for domain '%s'.",
            domain,
        )
        prompt = self._build_prompt(user_prompt, domain=domain)

        raw_text = self._call_gemini(prompt)
        logger.debug("Raw Gemini response text: %s", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from Gemini: %s", exc)
            raise

        try:
            manifest = SimulationManifest.parse_obj(data)
        except ValidationError as exc:
            logger.error("Manifest failed validation: %s", exc)
            raise

        # Ensure we always have a meaningful LLM prediction for the physics gap.
        if manifest.target_prediction <= 0:
            logger.warning(
                "LLM returned non-positive target_prediction (%.3f). "
                "Falling back to temp_k as naive prediction.",
                manifest.target_prediction,
            )
            manifest.target_prediction = manifest.temp_k

        logger.info("Parsed manifest: %s", manifest.json())
        return manifest


