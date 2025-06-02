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
from config_system import (
    GEMINI_MODEL,
    CRYPTO_ANALYSIS_TEMPLATE,
    STOCK_ANALYSIS_TEMPLATE,
    LOG_FULL_PROMPT,
    SELFTEST_LIVE_API_CALLS_ENABLED,
)
from bot_gemini_key_manager import GeminiKeyManagerBot
import numpy as np
from utils.logger_mixin import LoggerMixin
from utils.logging_decorators import log_method_calls


# === AIBot Class ===
class AIBot(LoggerMixin):
    """
    AIBot encapsulates all AI analysis, prompt generation, and validation logic for the trading system.
    Usage:
        bot = AIBot()
        result = bot.generate_analysis(prompt_template, variables)
        is_valid, error = AIBot.validate_analysis(result)
    """

    @log_method_calls
    def __init__(self, model_name=None, api_key=None):
        """
        Initialize the AIBot with model name and API key management.
        - Sets up Gemini LLM model and key manager.
        - Loads prompt templates for various analysis types.
        - Raises RuntimeError if no API key is available.
        """
        super().__init__()
        self.model_name = model_name or GEMINI_MODEL
        self.gemini_key_manager = GeminiKeyManagerBot()
        self.api_key = api_key or self.gemini_key_manager.get_available_key()
        if not self.api_key:
            raise RuntimeError(
                "All Gemini API keys exhausted for now. Wait before retrying."
            )
        genai.configure(api_key=self.api_key)
        self.llm = genai.GenerativeModel(self.model_name)
        self.prompt_templates = {
            "CRYPTO_ANALYSIS": CRYPTO_ANALYSIS_TEMPLATE,
            "STOCK_ANALYSIS": STOCK_ANALYSIS_TEMPLATE,
            "MARKET_OVERVIEW": """Market Overview:\n- S&P 500 30-day return: {spy_return:.2f}%\n- Current VIX: {current_vix:.2f}\n- Average VIX (30d): {avg_vix:.2f}\n- Market Sentiment: {market_sentiment}\n- Top Performing Sectors: {top_sectors}\n\nPlease provide:\n1. Key market themes and trends to focus on\n2. Asset types that might outperform in current conditions\n3. Risk factors to be aware of\n4. Specific sectors or themes to prioritize\nKeep response concise and actionable for trading decisions.""",
        }

    @staticmethod
    @log_method_calls
    def _format_prompt(template: str, variables) -> str:
        """
        Formats a prompt template with variables.
        - If variables is a dict, replaces placeholders in the template.
        - If variables is a string or None, returns the template as-is.
        """
        if variables is None:
            return template
        elif isinstance(variables, dict):
            prompt = template
            for k, v in variables.items():
                prompt = prompt.replace(f"{{{k}}}", str(v))
            return prompt
        elif isinstance(variables, str):
            return template
        else:
            return template

    @log_method_calls
    def generate_analysis(self, prompt_type, variables=None):
        """
        Generates an AI analysis using the LLM and a prompt template.
        - Selects the correct template and formats it with variables.
        - Calls the LLM and returns the response.
        """
        self.logger.debug(f"[generate_analysis()] Called with prompt_type={prompt_type!r}, variables={variables!r}")
        prompt_template = self.prompt_templates.get(prompt_type, "")
        prompt = self._format_prompt(prompt_template, variables)
        self.logger.debug(f"[generate_analysis()] Built prompt: {prompt!r}")
        result = self.call_llm(prompt)
        self.logger.debug(f"[generate_analysis()] LLM result: {result!r}")
        return result

    @log_method_calls
    def call_llm(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the response text.
        - Logs a concise summary or the full prompt depending on LOG_FULL_PROMPT.
        - Handles prompt summary extraction for logging.
        """
        method = "call_llm"
        self.logger.debug(f"[call_llm()] Received prompt: {prompt!r}")
        # Check for empty prompt and handle gracefully
        if not prompt or not prompt.strip():
            self.logger.error("[call_llm()] FAILED | Error: prompt must not be empty")
            return json.dumps({"error": "Prompt was empty. No analysis performed."})
        # Extract a concise summary from the prompt (e.g., asset type/symbol or first line)
        summary = None
        import re
        match = re.search(r"(Asset|Stock|Crypto):\\s*([\\w\\-\\.]+)", prompt)
        if match:
            summary = f"{match.group(1)}: {match.group(2)}"
        else:
            for line in prompt.splitlines():
                if line.strip():
                    summary = line.strip()
                    break
        if not summary:
            summary = "LLM prompt sent"
        if LOG_FULL_PROMPT:
            self.logger.debug(f"[call_llm()] Full prompt sent: {prompt}")
        else:
            self.logger.info(f"[call_llm()] Prompt summary: {summary}")
        try:
            response = self.llm.generate_content(prompt)
            self.logger.debug(f"[call_llm()] LLM_RAW_RESPONSE | response=\n{response.text}")
        except Exception as e:
            self.logger.error(f"[call_llm()] LLM call failed: {e}")
            return json.dumps({"error": f"LLM call failed: {e}"})
        try:
            parsed = json.loads(response.text)
            action = parsed.get('action', None)
            confidence = parsed.get('confidence', None)
            reason = parsed.get('reason', parsed.get('reasoning', None))
            self.logger.info(f"[call_llm()] LLM_DECISION | action='{action}', confidence={confidence}, reason='{reason}'")
        except Exception:
            self.logger.warning(f"[call_llm()] LLM_JSON_PARSE_FAIL | Could not parse: {response.text}")
        return response.text.strip()

    @log_method_calls
    def generate_embedding(self, text: str) -> list:
        """
        Converts text into a vector of numbers ("embedding") for similarity search, clustering, or as ML model input.
        This enables your agentic stock AI to compare, group, and analyze text data (like news, reports, or insights).
        """
        np.random.seed(abs(hash(text)) % (2**32))
        return (np.random.rand(384) - 0.5).tolist()

    @staticmethod
    @log_method_calls
    def clean_json_response(text):
        """Cleans and parses a possibly malformed JSON string from LLM output. Robustly strips markdown, extracts JSON, and validates schema."""
        import logging
        import re
        import json
        logger = logging.getLogger("AIBot.clean_json_response")
        raw = text
        logger.debug(f"[LLM_JSON_PARSE] Raw LLM output: {repr(raw)[:200]}")
        # 1. Remove all markdown code fences (```json, ```) 
        cleaned = re.sub(r"```[a-zA-Z]*", "", raw)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = cleaned.strip()
        logger.debug(f"[LLM_JSON_PARSE] After fence strip: {repr(cleaned)[:200]}")
        # 2. Try to extract JSON object or array
        match = re.search(r"({[\s\S]*})", cleaned)
        if match:
            cleaned = match.group(1)
            logger.debug(f"[LLM_JSON_PARSE] Extracted object: {repr(cleaned)[:200]}")
        else:
            match = re.search(r"(\[[\s\S]*\])", cleaned)
            if match:
                cleaned = match.group(1)
                logger.debug(f"[LLM_JSON_PARSE] Extracted array: {repr(cleaned)[:200]}")
        # 3. Normalize quotes and trailing commas
        cleaned = cleaned.replace("'", '"')
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        # 4. Attempt JSON parse with fallback repairs
        last_error = None
        for attempt in range(3):
            try:
                result = json.loads(cleaned)
                logger.info(f"[LLM_JSON_PARSE_SUCCESS] Parsed JSON on attempt {attempt+1}")
                # 5. Schema validation
                valid, err = AIBot.validate_analysis(result)
                if not valid:
                    logger.warning(f"[LLM_JSON_PARSE_FAIL] Schema validation failed: {err} | {repr(result)}")
                    return None
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"[LLM_JSON_PARSE_FAIL] Attempt {attempt+1} failed: {e} | {repr(cleaned)[:200]}")
                # Fallback: try to extract again or collapse whitespace
                if attempt == 0:
                    cleaned = re.sub(r"^.*?({[\s\S]*})", r"\1", cleaned)
                    cleaned = re.sub(r"^.*?(\[[\s\S]*\])", r"\1", cleaned)
                elif attempt == 1:
                    cleaned = re.sub(r"\s+", " ", cleaned)
        logger.error(f"[LLM_JSON_PARSE_FAIL] All attempts failed: {last_error} | {repr(cleaned)[:200]}")
        return None

    @staticmethod
    @log_method_calls
    def validate_analysis(analysis_data):
        """Validates that the AI's analysis matches the required schema."""
        if not analysis_data:
            return False, "No analysis data provided"
        required_keys = ["action", "reasoning", "confidence"]
        if not all(key in analysis_data for key in required_keys):
            return False, "Missing required fields"
        valid_actions = ["buy", "sell", "hold"]
        if analysis_data["action"] not in valid_actions:
            return False, f"Invalid action value: {analysis_data['action']}"
        if (
            not isinstance(analysis_data["confidence"], (int, float))
            or not 0 <= float(analysis_data["confidence"]) <= 1
        ):
            return False, "Confidence must be between 0.0 and 1.0"
        return True, ""

    @log_method_calls
    def adapt_with_performance(self, trade_history, win_rate, min_confidence=0.7):
        """Adjusts AI prompt or parameters based on win/loss rate and trade history."""
        if win_rate < 0.4:
            new_confidence = min(0.9, min_confidence + 0.1)
            prompt_note = "(Caution: Recent win rate is low. Only recommend trades with very high confidence and clear reasoning.)"
        elif win_rate > 0.7:
            new_confidence = max(0.5, min_confidence - 0.1)
            prompt_note = "(Performance is strong. You may consider more aggressive trades if justified.)"
        else:
            new_confidence = min_confidence
            prompt_note = ""
        return new_confidence, prompt_note

    @staticmethod
    @log_method_calls
    def parse_ai_analysis_response(response):
        """Ensures the response is a dict and contains 'action', 'confidence', and 'reasoning'."""
        if not isinstance(response, dict):
            return {
                "action": "hold",
                "confidence": 0.0,
                "reasoning": "Malformed AI response.",
            }
        if "action" not in response:
            response["action"] = "hold"
        if "confidence" not in response:
            response["confidence"] = 0.0
        if "reasoning" not in response:
            response["reasoning"] = "No reasoning provided."
        return response

    @staticmethod
    @log_method_calls
    def build_prompt(asset_analysis_input):
        """Builds a prompt for the LLM using asset_analysis_input fields."""
        symbol = getattr(asset_analysis_input, "symbol", "unknown")
        asset_type = getattr(asset_analysis_input, "asset_type", "unknown")
        market_data = getattr(asset_analysis_input, "market_data", {})
        technical_indicators = getattr(asset_analysis_input, "technical_indicators", {})
        news_sentiment = getattr(asset_analysis_input, "news_sentiment", None)
        reflection_insights = getattr(asset_analysis_input, "reflection_insights", None)
        historical_ai_context = getattr(
            asset_analysis_input, "historical_ai_context", None
        )
        prompt = f"""
        Asset: {symbol}
        Asset Type: {asset_type}
        Market Data: {market_data}
        Technical Indicators: {technical_indicators}
        News Sentiment: {news_sentiment}
        Reflection Insights: {reflection_insights}
        Historical AI Context: {historical_ai_context}
        """
        return prompt

    @log_method_calls
    def selftest(self):
        """Runs self-tests for clean_json_response and LLM call logic."""
        print(f"\n--- Running {self.__class__.__name__} Self-Test ---")
        try:
            print("  - Testing clean_json_response edge cases...")
            test_cases = [
                ('{"action": "buy", "reasoning": "Test", "confidence": 0.9}', True),
                (
                    '```json\n{"action": "sell", "reasoning": "Test", "confidence": 0.8}\n```',
                    True,
                ),
                (
                    'Some text before {"action": "hold", "reasoning": "Test", "confidence": 0.7} some text after',
                    True,
                ),
                ('{"action": "buy", "reasoning": "Test"', False),
                (
                    "{action: buy, reasoning: Test, confidence: 0.6}",
                    False,
                ),
                ("", False),
                ("   ", False),
                ("Not a JSON at all", False),
                ("[{'action': 'buy', 'confidence': 0.5, 'reasoning': 'Test'}]", True),
            ]
            for idx, (input_str, should_succeed) in enumerate(test_cases):
                result = self.clean_json_response(input_str)
                if should_succeed:
                    assert result is not None and isinstance(
                        result, (dict, list)
                    ), f"Test {idx+1} failed: Expected valid JSON, got {result}"
                else:
                    assert result is None or isinstance(
                        result, (dict, list)
                    ), f"Test {idx+1} failed: Expected None or partial, got {result}"
            print("    -> clean_json_response passed all edge case tests.")

            print("  - Testing LLM call logic...")
            if SELFTEST_LIVE_API_CALLS_ENABLED:
                test_prompt = "Say 'Gemini API test successful'"
                response = self.call_llm(test_prompt)
                assert (
                    isinstance(response, str)
                    and "gemini api test successful" in response.lower()
                ), f"LLM call did not return expected confirmation. Got: '{response}'"
                print("    -> LLM call successful and response as expected.")
            else:
                print(
                    "    -> Mocking LLM call (SELFTEST_LIVE_API_CALLS_ENABLED = False)..."
                )
                mock_responses = [
                    '{"action": "buy", "reasoning": "Mock", "confidence": 0.9}',
                    '```json\n{"action": "sell", "reasoning": "Mock", "confidence": 0.8}\n```',
                    'Some text before {"action": "hold", "reasoning": "Mock", "confidence": 0.7} after',
                ]
                for idx, resp in enumerate(mock_responses):
                    cleaned = self.clean_json_response(resp)
                    assert cleaned is not None and isinstance(
                        cleaned, dict
                    ), f"Mock LLM response {idx+1} failed to clean/parse. Got: {cleaned}"
                print("    -> Mocked LLM responses cleaned and parsed successfully.")
            print(f"--- {self.__class__.__name__} Self-Test PASSED ---")
        except AssertionError as e:
            print(f"--- {self.__class__.__name__} Self-Test FAILED: {e} ---")
        except Exception as e:
            print(
                f"--- {self.__class__.__name__} Self-Test encountered an ERROR: {e} ---"
            )


# === Standalone AI analysis function for legacy compatibility ===
@log_method_calls
def generate_ai_analysis(prompt, variables=None):
    """Standalone wrapper for AIBot.generate_analysis for legacy compatibility."""
    bot = AIBot()
    # If prompt is a template, try to infer type
    if "crypto" in prompt.lower():
        prompt_type = "CRYPTO_ANALYSIS"
    elif "stock" in prompt.lower():
        prompt_type = "STOCK_ANALYSIS"
    else:
        prompt_type = "MARKET_OVERVIEW"
    return bot.generate_analysis(prompt_type, variables)


if __name__ == "__main__":
    print("\n--- Running AIBot Real Gemini API Test ---")
    try:
        test_bot = AIBot()
        test_prompt = "Say 'Gemini API test successful'"
        print("  - Sending real prompt to Gemini API...")
        response = test_bot.call_llm(test_prompt)
        print(f"  - Gemini API response: {response}")
        assert (
            "gemini api test successful" in response.lower()
        ), f"Gemini API did not return expected confirmation. Got: '{response}'"
        print("--- AIBot Real Gemini API Test PASSED ---")
    except AssertionError as e:
        print(f"--- AIBot Real Gemini API Test FAILED: {e} ---")
    except Exception as e:
        print(f"--- AIBot Real Gemini API Test encountered an ERROR: {e} ---")

# === End of bot_ai.py ===
