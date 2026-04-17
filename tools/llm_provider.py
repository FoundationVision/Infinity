"""
Configurable LLM provider for prompt rewriting and other LLM-powered tools.

Supports multiple providers via environment variables:
  - LLM_PROVIDER: "openai" (default), "azure", "minimax"
  - LLM_API_KEY: API key for the selected provider
  - LLM_BASE_URL: Custom base URL (optional, auto-detected per provider)
  - LLM_MODEL: Model name (optional, defaults per provider)

Provider-specific env vars (take precedence over generic ones):
  - OPENAI_API_KEY / OPENAI_BASE_URL
  - AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_API_VERSION
  - MINIMAX_API_KEY

Examples:
  # Standard OpenAI
  export LLM_PROVIDER=openai
  export OPENAI_API_KEY=sk-...

  # MiniMax (OpenAI-compatible)
  export LLM_PROVIDER=minimax
  export MINIMAX_API_KEY=...

  # Custom OpenAI-compatible endpoint
  export LLM_PROVIDER=openai
  export LLM_BASE_URL=http://localhost:8000/v1
  export LLM_API_KEY=...
"""

import os
import openai


# Default models per provider
_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "azure": "gpt-4o",
    "minimax": "MiniMax-M2.7",
}

# Default base URLs per provider
_DEFAULT_BASE_URLS = {
    "openai": None,  # uses openai default
    "minimax": "https://api.minimax.io/v1",
}


def _detect_provider():
    """Auto-detect provider from available environment variables."""
    if os.environ.get("MINIMAX_API_KEY"):
        return "minimax"
    if os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    return "openai"


def _get_api_key(provider):
    """Get API key for the given provider."""
    # Generic key takes lowest precedence
    key = os.environ.get("LLM_API_KEY", "")

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", key)
    elif provider == "azure":
        key = os.environ.get("AZURE_OPENAI_API_KEY", key)
    elif provider == "minimax":
        key = os.environ.get("MINIMAX_API_KEY", key)

    # Fall back to conf.py GPT_AK if no env var is set
    if not key:
        try:
            from conf import GPT_AK
            if GPT_AK and GPT_AK != "[YOUR GPT_AK]":
                key = GPT_AK
        except ImportError:
            pass

    return key


def get_llm_client(provider=None, api_key=None, base_url=None, model=None):
    """
    Create an OpenAI-compatible client for the specified provider.

    Args:
        provider: LLM provider name ("openai", "azure", "minimax").
                  Auto-detected from env vars if not specified.
        api_key: API key. Read from env vars if not specified.
        base_url: Base URL. Uses provider default if not specified.
        model: Model name. Uses provider default if not specified.

    Returns:
        tuple: (client, model_name)
    """
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "").lower() or _detect_provider()

    if api_key is None:
        api_key = _get_api_key(provider)

    if model is None:
        model = os.environ.get("LLM_MODEL", "") or _DEFAULT_MODELS.get(provider, "gpt-4o")

    if base_url is None:
        base_url = os.environ.get("LLM_BASE_URL", "") or _DEFAULT_BASE_URLS.get(provider)

    if provider == "azure":
        endpoint = base_url or os.environ.get(
            "AZURE_OPENAI_ENDPOINT",
            "https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl",
        )
        api_version = os.environ.get("AZURE_API_VERSION", "2023-07-01-preview")
        client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        )
    else:
        # OpenAI, MiniMax, and any OpenAI-compatible provider
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.OpenAI(**kwargs)

    return client, model


def chat_completion(messages, provider=None, api_key=None, base_url=None,
                    model=None, return_json=False, temperature=None, **kwargs):
    """
    Run a chat completion using the configured LLM provider.

    Args:
        messages: List of message dicts with "role" and "content" keys.
        provider: LLM provider name. Auto-detected if not specified.
        api_key: API key. Read from env vars if not specified.
        base_url: Base URL. Uses provider default if not specified.
        model: Model name. Uses provider default if not specified.
        return_json: Whether to request JSON output format.
        temperature: Sampling temperature (optional).

    Returns:
        str: The assistant's response content.
    """
    client, model_name = get_llm_client(provider, api_key, base_url, model)

    create_kwargs = {
        "model": model_name,
        "messages": messages,
    }
    if return_json:
        create_kwargs["response_format"] = {"type": "json_object"}

    # MiniMax temperature must be in (0.0, 1.0]
    resolved_provider = provider or os.environ.get("LLM_PROVIDER", "").lower() or _detect_provider()
    if temperature is not None:
        if resolved_provider == "minimax":
            temperature = max(0.01, min(temperature, 1.0))
        create_kwargs["temperature"] = temperature

    completion = client.chat.completions.create(**create_kwargs)
    content = completion.choices[0].message.content

    # Strip MiniMax thinking tags if present
    if resolved_provider == "minimax" and content and "<think>" in content:
        import re
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()

    return content
