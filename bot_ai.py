# bot_ai.py
"""
AIBot: Handles all AI/LLM-based analysis, prompt generation, and response validation for the trading system.
- Generates prompts for LLMs (e.g., Gemini)
- Handles LLM response parsing and schema validation
- Provides both a class-based and standalone interface for AI analysis
"""

# === Imports ===
import re
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from config import GEMINI_API_KEY, GEMINI_MODEL, TEMPERATURE, MAX_TOKENS, ANALYSIS_SCHEMA, CRYPTO_ANALYSIS_TEMPLATE, STOCK_ANALYSIS_TEMPLATE

# === Standalone AI Analysis Function ===
def generate_ai_analysis(prompt_template, variables, model_name=None, api_key=None):
    """
    Generate AI-driven analysis using the configured LLM.
    Args:
        prompt_template (str): The prompt template string, with placeholders for variables.
        variables (dict): Dictionary of variables to fill into the prompt template.
        model_name (str, optional): Override the default model name.
        api_key (str, optional): Override the default API key.
    Returns:
        str: The LLM's response text, or an empty string on failure.
    """
    model = model_name or GEMINI_MODEL
    key = api_key or GEMINI_API_KEY
    try:
        genai.configure(api_key=key)
        llm = genai.GenerativeModel(model)
        prompt = prompt_template.format(**variables)
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return ""

# === AIBot Class ===
class AIBot:
    """
    AIBot encapsulates all AI analysis, prompt generation, and validation logic for the trading system.
    Usage:
        bot = AIBot()
        result = bot.generate_analysis(prompt_template, variables)
        is_valid, error = AIBot.validate_analysis(result)
    """
    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name or GEMINI_MODEL
        self.api_key = api_key or GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        self.llm = genai.GenerativeModel(self.model_name)

    def generate_analysis(self, prompt_template, variables):
        """
        Generate an AI analysis using the LLM and a prompt template.
        Returns the LLM's response text or an empty string on failure.
        """
        prompt = prompt_template.format(**variables)
        try:
            response = self.llm.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=TEMPERATURE,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=MAX_TOKENS
                )
            )
            return response.text.strip()
        except Exception:
            return ""

    @staticmethod
    def clean_json_response(text):
        """
        Clean and parse the AI's JSON response, handling common formatting issues.
        Returns a Python dict or None if parsing fails.
        """
        text = re.sub(r'```[a-zA-Z]*', '', text).strip()
        text = text.replace("'", '"')
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def validate_analysis(analysis_data):
        """
        Validate that the AI's analysis matches the required schema.
        Returns (True, "") if valid, otherwise (False, error_message).
        """
        if not analysis_data:
            return False, "No analysis data provided"
        required_keys = ['action', 'reasoning', 'confidence']
        if not all(key in analysis_data for key in required_keys):
            return False, "Missing required fields"
        valid_actions = ['buy', 'sell', 'hold']
        if analysis_data['action'] not in valid_actions:
            return False, f"Invalid action value: {analysis_data['action']}"
        if not isinstance(analysis_data['confidence'], (int, float)) or not 0 <= float(analysis_data['confidence']) <= 1:
            return False, "Confidence must be between 0.0 and 1.0"
        return True, ""

    def adapt_with_performance(self, trade_history, win_rate, min_confidence=0.7):
        """
        Adjust AI prompt or parameters based on win/loss rate and trade history.
        If win rate is low, increase required confidence or add caution to prompts.
        If win rate is high, allow more aggressive trading or lower confidence threshold.
        """
        # Example: Adjust temperature or min_confidence based on win_rate
        if win_rate < 0.4:
            # Too many losses, be more conservative
            new_confidence = min(0.9, min_confidence + 0.1)
            prompt_note = "(Caution: Recent win rate is low. Only recommend trades with very high confidence and clear reasoning.)"
        elif win_rate > 0.7:
            # High win rate, can be more aggressive
            new_confidence = max(0.5, min_confidence - 0.1)
            prompt_note = "(Performance is strong. You may consider more aggressive trades if justified.)"
        else:
            new_confidence = min_confidence
            prompt_note = ""
        return new_confidence, prompt_note

    def generate_embedding(self, text: str) -> list:
        """Generate an embedding for the given text using the LLM (stub for now)."""
        # TODO: Replace with real embedding model or API
        import numpy as np
        np.random.seed(abs(hash(text)) % (2**32))
        return (np.random.rand(384) - 0.5).tolist()  # Example: 384-dim random vector

    @staticmethod
    def parse_ai_analysis_response(response):
        # Ensure the response is a dict and contains 'action'
        import logging
        logger = logging.getLogger(__name__)
        if not isinstance(response, dict):
            logger.warning("AI analysis response is not a dict. Returning default hold action.")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': 'Malformed AI response.'}
        if 'action' not in response:
            logger.warning(f"AI analysis response missing 'action' key: {response}")
            response['action'] = 'hold'
        if 'confidence' not in response:
            response['confidence'] = 0.0
        if 'reasoning' not in response:
            response['reasoning'] = 'No reasoning provided.'
        return response

    def build_prompt(asset_analysis_input):
        prompt = f"""
        Asset: {asset_analysis_input.symbol}
        Reflection Insights:
        {asset_analysis_input.reflection_insights}

        Historical AI Context:
        {asset_analysis_input.historical_ai_context}

        # ...other prompt sections...
        """
        return prompt

# === End of bot_ai.py ===
