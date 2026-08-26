"""
Cache-Augmented Generation (CAG) cache implementation.

Provides:
    - CAGCache: Semantic similarity-based cache
    - Static FAQs: Predefined questions always in cache
    - Dynamic History: User queries with 24-hour TTL

How it works:
    1. Stores embeddings alongside cached responses.
    2. New queries are embedded once and compared against all cached
       embeddings.
    3. Cached queries use pre-computed embeddings, so no re-embedding
       is needed.
    4. Matching is done via cosine similarity (simple dot product).

Benefits:
    - Near-zero latency for cache hits.
    - Zero API cost for cached queries.
    - Catches paraphrased questions via semantic matching.
    - Lightweight: only new queries require embedding.
    - Two-tier design: Static FAQs + Dynamic History (24h TTL).
"""

import hashlib
import pickle
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CAGCache:
    """
    Semantic similarity-based cache with FAQ and history support.

    Two-tier caching:
        1. Static FAQs: predefined questions that never expire.
        2. Dynamic History: user queries with a configurable TTL(default 24h).

    All lookups use cosine similarity between query embeddings.
    """
    def __init__(
                    self, 
                    cache_dir: Path, 
                    embedder: Any, 
                    similarity_threshold: float=0.90, 
                    max_cache_size: int=1000, 
                    history_ttl_hours: float=24.0
                    ):
        
        self.cache_dir = cache_dir
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.history_ttl_hours = history_ttl_hours

        # Cache files
        self.faq_cache_file = cache_dir / "cag_faqs.pkl"
        self.history_cache_file = cache_dir / "cag_history.pkl"

        # Two-tier cache structure
        self.faq_cache: Dict[str, Any] = self._load_cache(self.faq_cache_file)
        self.history_cache: Dict[str, Any] = self._load_cache(self.history_cache_file)

        # Clean expired history on load
        self._cleanup_expired_history()

        # Build embedding matrices for fast lookup
        self._update_faq_embedding_matrix()
        self._update_history_embedding_matrix()

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load cache from the disk if exist""" 

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_file}: {e}")
                return {}
        return {}

    def _save_faq_cache(self) -> None:
        """Persist FAQ cache to disk."""
        try:
            with open(self.faq_cache_file, 'wb') as f:
                pickle.dump(self.faq_cache, f)
            logger.info(f"Saved {len(self.faq_cache)} items to FAQ cache on disk.")
        except Exception as e:
            logger.error(f"Failed to save FAQ cache to disk: {e}")

    def _save_history_cache(self) -> None:
        """Persist history cache to disk."""
        try:
            with open(self.history_cache_file, 'wb') as f:
                pickle.dump(self.history_cache, f)
            logger.info(f"Saved {len(self.history_cache)} items to History cache on disk.")
        except Exception as e:
            logger.error(f"Failed to save History cache to disk: {e}")

    def _cleanup_expired_history(self) -> None:
        """
        Remove entries older than TTL from history
        """

        # Calculate the cutoff timestamp.
        # Any entry older than (now - TTL hours) is considered expired.
        # Example: if TTL = 24 hours → cutoff = current_time - 86400 seconds
        cutoff_time = time.time() - (self.history_ttl_hours * 3600)

        # Collect all keys whose timestamp is older than the cutoff.
        # .get('timestamp', 0) safely handles missing timestamps
        # (treats them as very old → they will be removed).
        expired_keys = [
                        key for key, entry in self.history_cache.items()
                        if entry.get('timestamp', 0) < cutoff_time
                        ]
        
        if expired_keys:
            logger.info(f"Found {len(expired_keys)} expired entries in history cache. Purging...")
            for key in expired_keys:
                # Remove from in-memory cache
                del self.history_cache[key]
            self._save_history_cache()
        else:
            logger.debug("No expired history entries found during cleanup.")

    def _generate_key(self, query: str) -> str:
        """
        Generate a unique key for a query.
        """
        # Combine the query text with the current timestamp.
        raw_string = f"{query}_{time.time()}"

        # Convert the string to bytes and create an MD5 hash.
        # .hexdigest() turns the hash into a readable 32-character hex string.
        key = hashlib.md5(raw_string.encode()).hexdigest()
        logger.debug(f"Generated hash key '{key}' for query: '{query}'")

        return key

    """ Starting embedding operation methods. """

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query (only call needed for new queries)."""
        embedding = self.embedder.embed_query(query)
        return np.array(embedding)

    def _update_faq_embedding_matrix(self) -> None:
        """Build FAQ embedding matrix for fast batch similarity with logging."""

        logger.debug("Rebuilding FAQ embedding matrix...")
        
        # Filter for valid FAQs containing completed responses
        valid_faqs = {k: v for k, v in self.faq_cache.items() if v.get('has_response')}
        
        if not valid_faqs:
            logger.warning("No valid FAQs with responses found. Resetting FAQ matrix to empty.")
            self._faq_embedding_matrix = None
            self._faq_cache_ids = []
            return
        
        # Synchronize keys and matrix rows
        self._faq_cache_ids = list(valid_faqs.keys())
        embeddings = [valid_faqs[cid]['embedding'] for cid in self._faq_cache_ids]
        
        try:
            # Stack 1D array list into a single 2D NumPy array
            self._faq_embedding_matrix = np.vstack(embeddings)
            
            matrix_shape = self._faq_embedding_matrix.shape
            logger.info(
                        f"Successfully updated FAQ embedding matrix. "
                        f"Total FAQs: {matrix_shape[0]} | Embedding Dimensions: {matrix_shape[1]}"
                        )
        except Exception as e:
            logger.error(f"Failed to create FAQ embedding matrix: {e}")
            self._faq_embedding_matrix = None
            self._faq_cache_ids = []

    def _update_history_embedding_matrix(self) -> None:
        """Build history embedding matrix with active TTL tracking and logging."""

        logger.debug("Running history cleanup prior to updating matrix...")
        self._cleanup_expired_history()
        
        if not self.history_cache:
            logger.debug("History cache is currently empty. Resetting matrix attributes.")
            self._history_embedding_matrix = None
            self._history_cache_ids = []
            return
        
        # Track keys and stack active vectors
        self._history_cache_ids = list(self.history_cache.keys())
        embeddings = [self.history_cache[cid]['embedding'] for cid in self._history_cache_ids]
        
        try:
            self._history_embedding_matrix = np.vstack(embeddings)
            matrix_shape = self._history_embedding_matrix.shape
            logger.info(
                        f"Updated History embedding matrix: {matrix_shape[0]} active entries "
                        f"| Dimensions: {matrix_shape[1]}"
                        )
        except Exception as e:
            logger.error(f"Error stacking history embeddings into matrix: {e}")
            self._history_embedding_matrix = None
            self._history_cache_ids = [] 

    def _find_similar(self, query_embedding, embedding_matrix, cache_ids):
        """
        Find most similar entry using cosine similarity.
        
        This is the core "lightweight semantic check" - just a dot product.
        """
        if embedding_matrix is None or len(cache_ids) == 0:
            logger.debug("Similarity lookup skipped: matrix or cache IDs list is empty.")
            return None

        # Vector normalization with zero-division safeguard
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        cache_norms = embedding_matrix / (
            np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-10
        )

        # Vectorized dot product for batch cosine similarity
        similarities = np.dot(cache_norms, query_norm)

        # Find top match index and score
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        matched_id = cache_ids[best_idx]
        logger.debug(f"Top cosine match candidate ID: '{matched_id}' with score: {best_similarity:.4f}")

        # Check against similarity threshold
        if best_similarity >= self.similarity_threshold:
            logger.info(
                        f"Cache HIT! Score {best_similarity:.4f} >= Threshold {self.similarity_threshold:.4f} "
                        f"(ID: {matched_id})"
                        )
            return (matched_id, best_similarity)

        logger.info(
                    f"Cache MISS. Best similarity score {best_similarity:.4f} "
                    f"was below required threshold {self.similarity_threshold:.4f}"
                    )

        return None


    """ FAQ Management methods """

    def load_faqs(self, faq_queries: List[str], responses: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Load static FAQs into cache.
        
        If responses not provided, call warm_faqs() via CAGService later.
        
        Args:
            faq_queries: List of FAQ questions
            responses: Optional list of pre-computed responses
        
        Returns:
            Number of new FAQs loaded
        """
        loaded = 0
        for i, query in enumerate(faq_queries):
            # Check if a very similar FAQ already exists
            query_embedding = self._embed_query(query)
            existing = self._find_similar(
                                            query_embedding, 
                                            self._faq_embedding_matrix, 
                                            self._faq_cache_ids
                                            )
            
            # If we already have something > 95% similar → skip (avoid duplicates)
            if existing and existing[1] > 0.95:
                logger.debug(f"Skipping duplicate FAQ (similarity={existing[1]:.3f}): '{query}'")
                continue

            # Create a new unique key and entry
            key = self._generate_key(query)
            entry = {
                        'query': query,
                        'embedding': query_embedding,
                        'is_faq': True,
                        'timestamp': time.time()
                        }

            # Attach response if it was provided
            if responses and i < len(responses):
                entry['answer'] = responses[i].get('answer', '')
                entry['evidence_urls'] = responses[i].get('evidence_urls', [])
                entry['has_response'] = True
                logger.debug(f"Loaded FAQ with response: '{query}'")
            else:
                entry['has_response'] = False
                logger.debug(f"Loaded FAQ without response (pending): '{query}'")