"""Tests for tools/llm_provider.py — configurable LLM provider."""

import os
import unittest
from unittest.mock import patch, MagicMock

import openai


class TestDetectProvider(unittest.TestCase):
    """Test automatic provider detection from environment variables."""

    def _detect(self):
        from tools.llm_provider import _detect_provider
        return _detect_provider()

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=False)
    def test_detect_minimax(self):
        self.assertEqual(self._detect(), "minimax")

    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "az-key"}, clear=False)
    def test_detect_azure(self):
        # Remove MINIMAX_API_KEY if present to isolate test
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["AZURE_OPENAI_API_KEY"] = "az-key"
            self.assertEqual(self._detect(), "azure")

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_default_openai(self):
        self.assertEqual(self._detect(), "openai")


class TestGetApiKey(unittest.TestCase):
    """Test API key resolution with fallback priorities."""

    def _get_key(self, provider):
        from tools.llm_provider import _get_api_key
        return _get_api_key(provider)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai"}, clear=True)
    def test_openai_key(self):
        self.assertEqual(self._get_key("openai"), "sk-openai")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=True)
    def test_minimax_key(self):
        self.assertEqual(self._get_key("minimax"), "mm-key")

    @patch.dict(os.environ, {"LLM_API_KEY": "generic-key"}, clear=True)
    def test_generic_key_fallback(self):
        self.assertEqual(self._get_key("openai"), "generic-key")

    @patch.dict(os.environ, {"LLM_API_KEY": "generic", "OPENAI_API_KEY": "specific"}, clear=True)
    def test_specific_overrides_generic(self):
        self.assertEqual(self._get_key("openai"), "specific")


class TestGetLlmClient(unittest.TestCase):
    """Test client creation for different providers."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_openai_client(self):
        from tools.llm_provider import get_llm_client
        client, model = get_llm_client(provider="openai")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(model, "gpt-4o")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=True)
    def test_minimax_client(self):
        from tools.llm_provider import get_llm_client
        client, model = get_llm_client(provider="minimax")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(model, "MiniMax-M2.7")
        self.assertIn("minimax", client.base_url.host)

    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "az-test"}, clear=True)
    def test_azure_client(self):
        from tools.llm_provider import get_llm_client
        client, model = get_llm_client(provider="azure")
        self.assertIsInstance(client, openai.AzureOpenAI)
        self.assertEqual(model, "gpt-4o")

    @patch.dict(os.environ, {"LLM_API_KEY": "custom-key"}, clear=True)
    def test_custom_base_url(self):
        from tools.llm_provider import get_llm_client
        client, model = get_llm_client(
            provider="openai",
            base_url="http://localhost:8000/v1",
        )
        self.assertIsInstance(client, openai.OpenAI)
        self.assertIn("localhost", str(client.base_url))

    @patch.dict(os.environ, {"LLM_MODEL": "my-model", "OPENAI_API_KEY": "k"}, clear=True)
    def test_env_model_override(self):
        from tools.llm_provider import get_llm_client
        _, model = get_llm_client(provider="openai")
        self.assertEqual(model, "my-model")

    def test_explicit_params_override(self):
        from tools.llm_provider import get_llm_client
        client, model = get_llm_client(
            provider="minimax",
            api_key="explicit-key",
            model="custom-model",
        )
        self.assertEqual(model, "custom-model")


class TestChatCompletion(unittest.TestCase):
    """Test chat_completion wrapper."""

    @patch("tools.llm_provider.get_llm_client")
    def test_basic_completion(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "refined prompt"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_get_client.return_value = (mock_client, "gpt-4o")

        from tools.llm_provider import chat_completion
        result = chat_completion(
            messages=[{"role": "user", "content": "a tree"}],
            provider="openai",
            api_key="test-key",
        )
        self.assertEqual(result, "refined prompt")
        mock_client.chat.completions.create.assert_called_once()

    @patch("tools.llm_provider.get_llm_client")
    def test_json_mode(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"key": "value"}'
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_get_client.return_value = (mock_client, "gpt-4o")

        from tools.llm_provider import chat_completion
        result = chat_completion(
            messages=[{"role": "user", "content": "test"}],
            provider="openai",
            api_key="test-key",
            return_json=True,
        )
        self.assertEqual(result, '{"key": "value"}')
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})

    @patch("tools.llm_provider.get_llm_client")
    @patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}, clear=True)
    def test_minimax_think_tag_stripping(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "<think>internal reasoning</think>\nactual response"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_get_client.return_value = (mock_client, "MiniMax-M2.7")

        from tools.llm_provider import chat_completion
        result = chat_completion(
            messages=[{"role": "user", "content": "test"}],
            provider="minimax",
            api_key="test-key",
        )
        self.assertEqual(result, "actual response")

    @patch("tools.llm_provider.get_llm_client")
    @patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}, clear=True)
    def test_minimax_temperature_clamping(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "result"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_get_client.return_value = (mock_client, "MiniMax-M2.7")

        from tools.llm_provider import chat_completion
        chat_completion(
            messages=[{"role": "user", "content": "test"}],
            provider="minimax",
            api_key="test-key",
            temperature=0.0,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # temperature=0.0 should be clamped to 0.01 for MiniMax
        self.assertGreaterEqual(call_kwargs["temperature"], 0.01)

    @patch("tools.llm_provider.get_llm_client")
    def test_openai_no_temperature_clamping(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "result"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_get_client.return_value = (mock_client, "gpt-4o")

        from tools.llm_provider import chat_completion
        chat_completion(
            messages=[{"role": "user", "content": "test"}],
            provider="openai",
            api_key="test-key",
            temperature=0.0,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.0)


class TestPromptRewriter(unittest.TestCase):
    """Test the PromptRewriter integration with llm_provider."""

    @patch("tools.llm_provider.get_llm_client")
    def test_rewriter_uses_llm_provider(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "<prompt:A detailed tree in sunlight><cfg:3>"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_get_client.return_value = (mock_client, "gpt-4o")

        from tools.prompt_rewriter import PromptRewriter
        rewriter = PromptRewriter(system="", few_shot_history=[])
        result = rewriter.rewrite("a tree")
        self.assertIn("<prompt:", result)
        self.assertIn("<cfg:", result)

    @patch("tools.llm_provider.get_llm_client")
    def test_get_gpt_result_backward_compat(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_get_client.return_value = (mock_client, "gpt-4o")

        from tools.prompt_rewriter import get_gpt_result
        result, err = get_gpt_result(
            messages=[{"role": "user", "content": "test"}],
            retry=1,
        )
        self.assertEqual(result, "response")
        self.assertIsNone(err)


class TestDefaultModels(unittest.TestCase):
    """Test default model selection per provider."""

    def test_default_models_defined(self):
        from tools.llm_provider import _DEFAULT_MODELS
        self.assertIn("openai", _DEFAULT_MODELS)
        self.assertIn("minimax", _DEFAULT_MODELS)
        self.assertIn("azure", _DEFAULT_MODELS)
        self.assertEqual(_DEFAULT_MODELS["minimax"], "MiniMax-M2.7")

    def test_default_base_urls(self):
        from tools.llm_provider import _DEFAULT_BASE_URLS
        self.assertIn("minimax", _DEFAULT_BASE_URLS)
        self.assertIn("minimax.io", _DEFAULT_BASE_URLS["minimax"])


if __name__ == "__main__":
    unittest.main()
