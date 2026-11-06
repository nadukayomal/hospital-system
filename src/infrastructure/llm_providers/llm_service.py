import sys
import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from typing import Optional, Any

ROOT_DIR = Path(__file__).resolve().parents[3]
print(ROOT_DIR)

load_dotenv()

# Import configuration constants and helper functions from utils.
from utils import *

def get_chat_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
    streaming: bool = False,
    tier: Optional[str] = None,
) -> ChatOpenAI:
    """
    Initialize and return a ChatOpenAI model instance for OpenAI or OpenRouter.

    Args:
        model (str, optional): Name of the language model to instantiate.
        provider (str, optional): Service provider name ("openai" or "openrouter").
        temperature (float, optional): Sampling temperature. Defaults to 0.0.
        max_tokens (int, optional): Maximum tokens to generate. Defaults to 500.
        streaming (bool, optional): Whether to stream responses. Defaults to False.
        tier (str, optional): Model performance/tier classification ("general", "strong", "reason").

    Returns:
        ChatOpenAI: Configured LangChain ChatOpenAI object.
    """
    if not provider:
        config_provider = get_provider().get("default", "openrouter")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openrouter_key:
            provider = "openrouter"
        elif openai_key:
            provider = "openai"
        else:
            provider = config_provider

    provider = provider.lower()
    tier = tier or get_provider().get("tier", "general")

    if provider == "openrouter":
        api_key = load_api_keys(provider="openrouter") or os.getenv("OPENROUTER_API_KEY")
        open_router_url = get_provider().get("openrouter_base_url", "https://openrouter.ai/api/v1")
        if not model:
            model = get_openrouter().get("chat", {}).get(tier, "openai/gpt-4o-mini")

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            openai_api_key=api_key,
            openai_api_base=open_router_url
        )
    else:
        api_key = load_api_keys(provider="openai") or os.getenv("OPENAI_API_KEY")
        if not model:
            model = get_openai().get("chat", {}).get(tier, "gpt-4o-mini")

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            openai_api_key=api_key
        )

def get_reasoning_llm(**kwargs: Any) -> ChatOpenAI:
    """Instantiate a reasoning-optimized LLM (e.g., o3-mini, DeepSeek R1).

    Args:
        **kwargs: Additional model overrides passed to `get_chat_llm`.

    Returns:
        ChatOpenAI: Configured reasoning model instance.
    """
    return get_chat_llm(tier="reason", **kwargs)


def get_strong_llm(**kwargs: Any) -> ChatOpenAI:
    """Instantiate a high-capability LLM (e.g., GPT-4o, Claude 3.5 Sonnet).

    Args:
        **kwargs: Additional model overrides passed to `get_chat_llm`.

    Returns:
        ChatOpenAI: Configured general-purpose strong model instance.
    """
    return get_chat_llm(tier="strong", **kwargs)
    
if __name__ == "__main__":
    pass
