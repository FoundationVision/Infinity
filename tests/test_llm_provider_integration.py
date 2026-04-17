"""Integration tests for tools/llm_provider.py — require live API keys."""

import os
import unittest

# Skip all tests if no API key is available
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MESSAGES = [
    {"role": "system", "content": "You refine image prompts. Output only the refined prompt."},
    {"role": "user", "content": "a tree"},
]


@unittest.skipUnless(MINIMAX_API_KEY, "MINIMAX_API_KEY not set")
class TestMiniMaxIntegration(unittest.TestCase):
    """Live integration tests against MiniMax API."""

    def test_chat_completion(self):
        from tools.llm_provider import chat_completion
        result = chat_completion(
            messages=MESSAGES,
            provider="minimax",
            api_key=MINIMAX_API_KEY,
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

    def test_chat_completion_with_model(self):
        from tools.llm_provider import chat_completion
        result = chat_completion(
            messages=MESSAGES,
            provider="minimax",
            api_key=MINIMAX_API_KEY,
            model="MiniMax-M2.5-highspeed",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)

    def test_prompt_rewriter(self):
        from tools.llm_provider import chat_completion
        from tools.prompt_rewriter import SYSTEM, FEW_SHOT_HISTORY
        messages = (
            [{"role": "system", "content": SYSTEM}]
            + FEW_SHOT_HISTORY
            + [{"role": "user", "content": "a sunset over the ocean"}]
        )
        result = chat_completion(
            messages=messages,
            provider="minimax",
            api_key=MINIMAX_API_KEY,
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)


@unittest.skipUnless(OPENAI_API_KEY, "OPENAI_API_KEY not set")
class TestOpenAIIntegration(unittest.TestCase):
    """Live integration tests against OpenAI API."""

    def test_chat_completion(self):
        from tools.llm_provider import chat_completion
        result = chat_completion(
            messages=MESSAGES,
            provider="openai",
            api_key=OPENAI_API_KEY,
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)


if __name__ == "__main__":
    unittest.main()
