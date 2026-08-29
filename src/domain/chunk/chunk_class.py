import os
import sys
import logging

from typing import List, Dict, Any, Optional, Tuple
from .chunker import (
                        fixed_chunk,
                        semantic_chunk,
                        sliding_chunk,
                        parent_child_chunk,
                        late_chunk_index,
                        late_chunk_split,
                        )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Unified service for all chunking strategy
    """
    def __init__(self):
        self.strategies = {
                            "semantic": semantic_chunk,
                            "fixed": fixed_chunk,
                            "sliding": sliding_chunk,
                            "parent_child": parent_child_chunk,
                            "late_chunk": late_chunk_index
                            }
        
    def chunk(self, documents: List, strategy: str):
        """
        Chunk document based on provided strategy.
        Args:
            documents: list of document dicts.
            strategy: one strategy has defined self.strategy.
        Return:
            list of chunks.
        """

        if strategy not in self.strategies:
            logger.error("Not available provided chunking strategy check -> chunker.py")
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(self.strategies.keys())}")

        logger.info(f"Initiate chunking with : {strategy}")
        return self.strategies[strategy](documents)

    def available_strategies(self) -> List:
        """ Return list of available chunking strategies """

        return list(self.strategies.keys())