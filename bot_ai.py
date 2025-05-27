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
from config_system import GEMINI_API_KEY, GEMINI_MODEL, TEMPERATURE, MAX_TOKENS, ANALYSIS_SCHEMA, CRYPTO_ANALYSIS_TEMPLATE, STOCK_ANALYSIS_TEMPLATE

# === Standalone AI Analysis Function ===
def generate_ai_analysis(prompt_template, variables, model_name=None, api_key=None):
    """
    Generate AI-driven analysis using the configured LLM.
    Args:
        prompt_template (str): The prompt template string, with placeholders for variables.
        variables (dict or str): Variables to fill into the prompt template.
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
        # Fix: handle both dict and str for variables
        if variables is None:
            prompt = prompt_template
        elif isinstance(variables, dict):
            try:
                # Convert template to f-string style using locals()
                prompt = prompt_template
                for k, v in variables.items():
                    prompt = prompt.replace(f'{{{k}}}', str(v))
            except Exception as e:
                prompt = f"[Prompt Format Error: {e}]\n{prompt_template}"
        elif isinstance(variables, str):
            prompt = prompt_template
        else:
            prompt = prompt_template
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
        # Add prompt_templates mapping for correct operation
        self.prompt_templates = {
            'CRYPTO_ANALYSIS': CRYPTO_ANALYSIS_TEMPLATE,
            'STOCK_ANALYSIS': STOCK_ANALYSIS_TEMPLATE,
            'MARKET_OVERVIEW': """Market Overview:\n- S&P 500 30-day return: {spy_return:.2f}%\n- Current VIX: {current_vix:.2f}\n- Average VIX (30d): {avg_vix:.2f}\n- Market Sentiment: {market_sentiment}\n- Top Performing Sectors: {top_sectors}\n\nPlease provide:\n1. Key market themes and trends to focus on\n2. Asset types that might outperform in current conditions\n3. Risk factors to be aware of\n4. Specific sectors or themes to prioritize\nKeep response concise and actionable for trading decisions."""
        }

    def generate_analysis(self, prompt_type, variables=None):
        """
        Generate an AI analysis using the LLM and a prompt template.
        Returns the LLM's response text or an empty string on failure.
        """
        prompt_template = self.prompt_templates.get(prompt_type, "")
        # Fix: Ensure variables is a dict for .format(**variables)
        if variables is None:
            return prompt_template
        if isinstance(variables, dict):
            try:
                prompt = prompt_template
                for k, v in variables.items():
                    prompt = prompt.replace(f'{{{k}}}', str(v))
                return prompt
            except Exception as e:
                return f"[Prompt Format Error: {e}]\n{prompt_template}"
        if isinstance(variables, str):
            return prompt_template
        # Fallback for other types
        return prompt_template

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

    @staticmethod
    def build_prompt(asset_analysis_input):
        # Build a robust prompt for the LLM using asset_analysis_input fields
        symbol = getattr(asset_analysis_input, 'symbol', 'unknown')
        asset_type = getattr(asset_analysis_input, 'asset_type', 'unknown')
        market_data = getattr(asset_analysis_input, 'market_data', {})
        technical_indicators = getattr(asset_analysis_input, 'technical_indicators', {})
        news_sentiment = getattr(asset_analysis_input, 'news_sentiment', None)
        reflection_insights = getattr(asset_analysis_input, 'reflection_insights', None)
        historical_ai_context = getattr(asset_analysis_input, 'historical_ai_context', None)
        prompt = f"""
        Asset: {symbol}
        Asset Type: {asset_type}
        Market Data: {market_data}
        Technical Indicators: {technical_indicators}
        News Sentiment: {news_sentiment}
        Reflection Insights: {reflection_insights}
        Historical AI Context: {historical_ai_context}
        # ...add more sections as needed...
        """
        return prompt

# === End of bot_ai.py ===
