"""
IBM Watsonx.ai / Granite LLM Integration
Handles authentication and model inference via ibm-watsonx-ai SDK.

Environment variables (set in .env next to this project):
    WATSONX_API_KEY      — IBM Cloud API key
    WATSONX_PROJECT_ID   — watsonx.ai project UUID
    WATSONX_URL          — regional endpoint (default: us-south)
    GRANITE_MODEL_ID     — model to use (default: ibm/granite-3-8b-instruct)
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (two levels up from this file:
#   src/granite_llm.py  →  src/  →  project root/)
# This works regardless of which directory `streamlit run` was launched from.
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent          # .../src/
_PROJECT    = _HERE.parent                              # .../Interview_Agent/
_ENV_FILE   = _PROJECT / ".env"

load_dotenv(dotenv_path=_ENV_FILE, override=False)

# Also try cwd as a fallback (covers running directly from project root)
load_dotenv(override=False)

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams


# ---------------------------------------------------------------------------
# Placeholder sentinel — .env.example ships with these literal strings
# ---------------------------------------------------------------------------
_PLACEHOLDERS = {
    "your_ibm_cloud_api_key_here",
    "your_api_key",
    "your_watsonx_api_key_here",
    "YOUR_API_KEY",
}

DEFAULT_MODEL = "ibm/granite-3-8b-instruct"


def _mask(value: str) -> str:
    """Return first 4 chars + asterisks — safe for debug printing."""
    if not value:
        return "(empty)"
    return value[:4] + "*" * max(0, len(value) - 4)


class GraniteLLM:
    """
    Wrapper around IBM watsonx.ai ModelInference for Granite foundation models.

    Usage:
        llm = GraniteLLM()
        response = llm.generate("What is recursion?")
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ):
        self.api_key    = os.getenv("WATSONX_API_KEY", "").strip()
        self.project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
        self.url        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()
        self.model_id   = model_id or os.getenv("GRANITE_MODEL_ID", DEFAULT_MODEL).strip()

        # ── Debug output (masked) ──────────────────────────────────────────
        print("[GraniteLLM] Loading credentials …")
        print(f"  .env file searched : {_ENV_FILE}")
        print(f"  .env file exists   : {_ENV_FILE.exists()}")
        print(f"  WATSONX_API_KEY    : {_mask(self.api_key)}")
        print(f"  WATSONX_PROJECT_ID : {'SET (' + self.project_id[:8] + '…)' if self.project_id else '(empty)'}")
        print(f"  WATSONX_URL        : {self.url}")
        print(f"  GRANITE_MODEL_ID   : {self.model_id}")

        # ── Validate: missing ──────────────────────────────────────────────
        if not self.api_key:
            raise EnvironmentError(
                "WATSONX_API_KEY is not set.\n"
                f"Edit {_ENV_FILE} and add your IBM Cloud API key."
            )
        if not self.project_id:
            raise EnvironmentError(
                "WATSONX_PROJECT_ID is not set.\n"
                f"Edit {_ENV_FILE} and add your watsonx.ai project ID."
            )

        # ── Validate: still contains placeholder text ──────────────────────
        if self.api_key in _PLACEHOLDERS:
            raise EnvironmentError(
                "WATSONX_API_KEY still contains the placeholder value.\n"
                f"Open {_ENV_FILE} and replace it with your real IBM Cloud API key.\n\n"
                "Get your key at: https://cloud.ibm.com → Manage → Access (IAM) → API keys"
            )
        if self.project_id in _PLACEHOLDERS or "your_" in self.project_id.lower():
            raise EnvironmentError(
                "WATSONX_PROJECT_ID still contains the placeholder value.\n"
                f"Open {_ENV_FILE} and replace it with your real watsonx.ai project UUID.\n\n"
                "Find it at: https://dataplatform.cloud.ibm.com → your project → Manage → General"
            )

        self.parameters = {
            GenParams.MAX_NEW_TOKENS:     max_new_tokens,
            GenParams.TEMPERATURE:        temperature,
            GenParams.TOP_P:              top_p,
            GenParams.TOP_K:              top_k,
            GenParams.REPETITION_PENALTY: repetition_penalty,
            GenParams.STOP_SEQUENCES:     ["<|endoftext|>"],
        }

        self._client = None
        self._model  = None
        print("[GraniteLLM] Credentials validated. Model will be loaded on first call.")

    # -------------------------------------------------------------------------
    # Lazy initialisation — connects to IBM watsonx.ai on the first generate()
    # -------------------------------------------------------------------------
    def _get_model(self) -> ModelInference:
        if self._model is None:
            print("[GraniteLLM] Connecting to IBM watsonx.ai …")
            credentials = Credentials(
                api_key = self.api_key,
                url     = self.url,
            )
            self._client = APIClient(
                credentials = credentials,
                project_id  = self.project_id,
            )
            self._model = ModelInference(
                model_id    = self.model_id,
                credentials = credentials,
                project_id  = self.project_id,
                params      = self.parameters,
            )
            print("[GraniteLLM] Connected.")
        return self._model

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the generated text string."""
        model    = self._get_model()
        response = model.generate_text(prompt=prompt)
        return response.strip()

    def generate_with_metadata(self, prompt: str) -> dict:
        """Return full response dict including token usage."""
        model    = self._get_model()
        response = model.generate(prompt=prompt)
        result   = response["results"][0]
        return {
            "generated_text":    result["generated_text"].strip(),
            "input_token_count": result.get("input_token_count", 0),
            "generated_tokens":  result.get("generated_token_count", 0),
            "stop_reason":       result.get("stop_reason", ""),
        }

    def health_check(self) -> bool:
        """Verify connectivity to IBM watsonx.ai."""
        try:
            result = self.generate("Reply with the single word: OK")
            return bool(result)
        except Exception as exc:
            print(f"[GraniteLLM] Health check failed: {exc}")
            return False
