"""
Gemini Agent for generic health insights.
Uses the new Google GenAI SDK (google-genai) for personalized health coaching.
"""
from typing import Dict, List, Optional
from .base_agent import BaseAgent
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, CHAT_MODEL_INFO, CHAT_MODEL_GEMINI


class GeminiAgent(BaseAgent):
    """Agent that provides generic health insights using the new Google GenAI SDK."""

    def __init__(self):
        info = CHAT_MODEL_INFO[CHAT_MODEL_GEMINI]
        super().__init__(name=info["name"], description=info["description"])

        self.client = None
        self.api_key = self._get_api_key()
        self.model_name = GEMINI_MODEL
        self.capabilities = info["capabilities"]

    def _get_api_key(self):
        """Get API key from various sources."""
        import os
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
        return GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")

    def initialize(self, **kwargs) -> bool:
        """Initialize Gemini API using the new SDK."""
        try:
            from google import genai

            api_key = kwargs.get("api_key", self.api_key)
            if not api_key:
                print("[ERROR] Gemini API key missing.")
                return False

            self.client = genai.Client(api_key=api_key)
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize Gemini agent: {e}")
            self.is_initialized = False
            return False

    def generate_response(self, message: str, context: Dict, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response using the new SDK."""
        if not self.is_initialized:
            return self._generate_fallback_response(message, context)
        try:
            prompt = self._build_system_prompt(context)
            full_contents = prompt + "\nUser Question: " + message
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_contents
            )
            return response.text
        except Exception as e:
            print(f"[ERROR] Gemini response failure: {e}")
            return "[DEBUG] Error: " + str(e) + " | Model: " + str(self.model_name)
    def _build_system_prompt(self, context: Dict) -> str:
        prob = context.get("probability", 0)
        risk = context.get("risk_level", "Unknown")
        summary = context.get("profile_summary", "No data")
        prob_str = "{:.1f}".format(prob)
        return (
            "You are a compassionate health coach.\n"
            "CONTEXT: Risk Level " + risk + " (" + prob_str + "%), Metrics: " + summary + "\n"
            "YOUR ROLE: Provide empathetic, positive guidance on lifestyle and habits.\n"
            "SAFETY: Always clarify this is NOT a medical diagnosis. Include disclaimers."
        )

    def _generate_fallback_response(self, message: str, context: Dict) -> str:
        prob = context.get("probability", 0)
        risk = context.get("risk_level", "Unknown")
        prob_str = "{:.1f}".format(prob)
        return "Assessment: Risk probability " + prob_str + "% (" + risk + "). Please consult a doctor for a formal diagnosis."

    def get_capabilities(self) -> List[str]:
        return self.capabilities
