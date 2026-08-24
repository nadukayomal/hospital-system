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
                            model: str, 
                            provider: str, 
                            batch_size: int = None, 
                            tier: str = None, 
                            show_progress: bool = None
                            ) -> OpenAIEmbeddings:
    """
    Factory function to create an embedding model instance.
    
    Supports multiple providers via OpenRouter unified API,
    or direct OpenAI access if configured.
    
    Args:
        model: Model name (e.g., "text-embedding-3-large"). If None, uses config.
        provider: Override provider ("openrouter", "openai")
        tier: Embedding tier ("default", "small")
        batch_size: Number of texts to embed in parallel
        show_progress: Display progress bar for large batch operations
    
    Returns:
        OpenAIEmbeddings: An embedding model instance ready for vectorization
    """
    if provider.lower() == "openrouter":

        api_key = load_api_keys(provider = "openai")
        open_router_url = "https://openrouter.ai/api/v1"

        logger.info(f"Openrouter support embedding")

        try:
            return OpenAIEmbeddings(
                        model=model,
                        openai_api_key=api_key,
                        openai_api_base =open_router_url,
                        show_progress_bar=show_progress
                    )
        except Exception as e:
            logger.error(f"Occur error returning default embedding openrouter: {e}")

    else:
        logger.info(f"Direct openai embedding")

        api_key = load_api_keys(provider = "openai")

        try:
            return OpenAIEmbeddings(
                        model=model,
                        openai_api_key=api_key,
                        show_progress_bar=show_progress
                    )
        except Exception as e:
            logger.error(f"Occur error returnning direct openai embedding : {e}")



