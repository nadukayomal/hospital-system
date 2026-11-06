import os
import sys
import logging

from typing import Optional, Any
from langchain_openai import OpenAIEmbeddings

from utils import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



def get_default_embeddings(
    model: Optional[str] = None, 
    provider: Optional[str] = None, 
    batch_size: Optional[int] = None, 
    tier: Optional[str] = None, 
    show_progress: Optional[bool] = None
) -> OpenAIEmbeddings:
    """
    Factory function to create an embedding model instance.
    
    Supports multiple providers via OpenRouter unified API,
    or direct OpenAI access if configured.
    
    Args:
        model: Model name (e.g., "openai/text-embedding-3-small"). If None, resolved from config.
        provider: Provider name ("openrouter", "openai"). If None, resolved from config or env.
        tier: Embedding tier ("default", "small")
        batch_size: Number of texts to embed in parallel
        show_progress: Display progress bar for large batch operations
    
    Returns:
        OpenAIEmbeddings: An embedding model instance ready for vectorization
    """
    # Resolve provider
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
    tier = tier or get_embedding().get("tier", "default")
    batch_size = batch_size or get_embedding().get("batch_size", 100)

    if provider in ["local", "huggingface", "sentence-transformers"]:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = model or "all-MiniLM-L6-v2"
        logger.info(f"Using local HuggingFace embeddings: model={model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)

    if provider == "openrouter":
        api_key = load_api_keys(provider="openrouter") or os.getenv("OPENROUTER_API_KEY")
        open_router_url = get_provider().get("openrouter_base_url", "https://openrouter.ai/api/v1")
        if not model:
            model = get_openrouter().get("embedding", {}).get(tier, "openai/text-embedding-3-small")

        logger.info(f"Using OpenRouter embeddings: model={model}")

        return OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            openai_api_base=open_router_url,
            show_progress_bar=show_progress or False,
            check_embedding_ctx_length=False,
            chunk_size=batch_size,
        )

    else:
        api_key = load_api_keys(provider="openai") or os.getenv("OPENAI_API_KEY")
        if not model:
            model = get_openai().get("embedding", {}).get(tier, "text-embedding-3-small")

        logger.info(f"Using direct OpenAI embeddings: model={model}")

        return OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            show_progress_bar=show_progress or False,
            chunk_size=batch_size,
        )



