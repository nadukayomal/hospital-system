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
                model: str,
                provider: str,
                temperature: float = 0.7,
                max_tokens: int = 500,
                streaming: bool = False,
                tier: str = None,
                )-> ChatOpenAI:
    """
    Initialize and return a ChatOpenAI model instance for OpenAI or OpenRouter.

    Args:
        model (str): Name of the language model to instantiate.
        provider (str): Service provider name ("openai" or "openrouter").
        tier (str): Model performance/tier classification.
        temperature (float, optional): Sampling temperature for output randomness between 0.0 and 2.0. Defaults to 0.7.
        max_tokens (int, optional): Maximum number of tokens to generate in the completion. Defaults to 500.
        streaming (bool, optional): Whether to stream responses token-by-token. Defaults to False.

    Returns:
        ChatOpenAI: Configured LangChain ChatOpenAI object.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")

    if provider == "openrouter":
        open_router_url = "https://openrouter.ai/api/v1"

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            openai_api_key=api_key,
            openai_api_base=open_router_url
        )
    else:
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
