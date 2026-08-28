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

            # Store it
            self.faq_cache[key] = entry
            loaded += 1

        # Persist + rebuild matrix only if something changed    
        if loaded > 0:
            self._save_faq_cache()
            self._update_faq_embedding_matrix()
            logger.info(f"Loaded {loaded} new FAQ(s). Total FAQs now: {len(self.faq_cache)}")
        else:
            logger.debug("No new FAQs were loaded (all were duplicates).")

        return loaded

    def get_pending_faqs(self) -> List[str]:
        """Get FAQ queries that don't have responses yet."""
    
        pending = [
                    entry['query'] 
                    for entry in self.faq_cache.values()
                    if not entry.get('has_response', False)
                    ]

        logger.debug(f"Found {len(pending)} pending FAQ(s) without responses")
        return pending 

    def update_faq_response(self, query: str, response: Dict[str, Any]) -> bool:
        """
        Update response for an FAQ entry.

        Args:
            query: The FAQ question
            response: Dict with 'answer' and 'evidence_urls'

        Returns:
            True if updated, False if FAQ not found
        """
        logger.debug(f"Trying to update FAQ response for: '{query}'")

        # Exact text match (fastest, for pending FAQs)
        for key, entry in self.faq_cache.items():
            if entry['query'].lower().strip() == query.lower().strip():
                self.faq_cache[key]['answer'] = response['answer']
                self.faq_cache[key]['evidence_urls'] = response.get('evidence_urls', [])
                self.faq_cache[key]['has_response'] = True
                self.faq_cache[key]['timestamp'] = time.time()

                self._save_faq_cache()
                self._update_faq_embedding_matrix()

                logger.info(f"Updated FAQ (exact match): '{query}'")
                return True

        # Semantic match (in case the wording is slightly different)
        query_embedding = self._embed_query(query)
        match = self._find_similar(
                                    query_embedding,
                                    self._faq_embedding_matrix if self._faq_embedding_matrix is not None else None,
                                    self._faq_cache_ids
                                    )
        
        if match:
            key = match[0]
            self.faq_cache[key]['answer'] = response['answer']
            self.faq_cache[key]['evidence_urls'] = response.get('evidence_urls', [])
            self.faq_cache[key]['has_response'] = True
            self.faq_cache[key]['timestamp'] = time.time()

            self._save_faq_cache()
            self._update_faq_embedding_matrix()

            logger.info(f"Updated FAQ (semantic match, score={match[1]:.3f}): '{query}'")
            return True

        logger.warning(f"Could not find FAQ to update: '{query}'")
        return False

    def list_faqs(self) -> List[Dict[str, Any]]:
        """List all FAQ entries with their status."""

        result = [
                    {
                        'query': entry['query'],
                        'has_response': entry.get('has_response', False),
                        'timestamp': datetime.fromtimestamp(entry['timestamp']).isoformat()
                    }
                    for entry in self.faq_cache.values()
                ]

        logger.debug(f"Listing {len(result)} FAQ entries")
        return result

    """ Public Interface """

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response using semantic similarity.
        
        Lookup order: FAQs first -> History second
        
        Args:
            query: User query
        
        Returns:
            Cached response with 'source', 'similarity_score', 'matched_query'
            or None if no match
        """
        logger.info(f"Initiating cache lookup for query: '{query}'")
        
        # Clean expired dynamic history prior to lookup
        self._cleanup_expired_history()
        
        # Embed user query vector
        query_embedding = self._embed_query(query)
        
        # Search: FAQ Cache
        faq_match = self._find_similar(
                                        query_embedding,
                                        self._faq_embedding_matrix,
                                        self._faq_cache_ids
                                        )
        
        if faq_match:
            cache_id, similarity = faq_match
            cached = self.faq_cache[cache_id].copy()
            cached.pop('embedding', None)
            cached['similarity_score'] = similarity
            cached['matched_query'] = cached['query']
            cached['source'] = 'faq'
            logger.info(f"HIT [FAQ Tier] | Score: {similarity:.4f} | Matched: '{cached['matched_query']}'")

            return cached
            
        # Search: History Cache
        self._update_history_embedding_matrix()
        
        history_match = self._find_similar(
                                            query_embedding,
                                            self._history_embedding_matrix,
                                            self._history_cache_ids
                                            )
        
        if history_match:
            cache_id, similarity = history_match
            entry = self.history_cache[cache_id]
            
            # Verify TTL window
            if time.time() - entry['timestamp'] < self.history_ttl_hours * 3600:
                cached = entry.copy()
                cached.pop('embedding', None)
                cached['similarity_score'] = similarity
                cached['matched_query'] = cached['query']
                cached['source'] = 'history'
                logger.info(f"HIT [History Tier] | Score: {similarity:.4f} | Matched: '{cached['matched_query']}'")
                return cached
            else:
                logger.warning(f"Matched history entry '{cache_id}' expired during lookup check.")

        logger.info(f"MISS [All Tiers] | No matching cached entries found for query: '{query}'")
        return None

    def set(self, query: str, response: Dict[str, Any]) -> None:
        """
        Cache a response to history.
        
        Args:
            query: User query
            response: Dict with 'answer' and optionally 'evidence_urls'
        """

        key = self._generate_key(query)
        embedding = self._embed_query(query)
        
        self.history_cache[key] = {
                                    'query': query,
                                    'embedding': embedding,
                                    'answer': response['answer'],
                                    'evidence_urls': response.get('evidence_urls', []),
                                    'timestamp': time.time(),
                                    'is_faq': False
                                    }
        logger.info(f"Cached new query to history. Generated key: '{key}'")
        
        # FIFO Eviction Check
        if len(self.history_cache) > self.max_cache_size:
            oldest_key = min(
                self.history_cache.keys(),
                key=lambda k: self.history_cache[k]['timestamp']
            )
            del self.history_cache[oldest_key]
            logger.info(f"Cache limit exceeded ({self.max_cache_size}). Evicted oldest key: '{oldest_key}'")
            
        self._update_history_embedding_matrix()
        self._save_history_cache()

    def clear(self, clear_faqs: bool = False) -> None:
        """
        Clear cache.
        
        Args:
            clear_faqs: If True, also clear FAQ cache (default False)
        """
        history_count = len(self.history_cache)
        self.history_cache = {}
        self._history_embedding_matrix = None
        self._history_cache_ids = []
        self._save_history_cache()
        logger.info(f"Cleared {history_count} dynamic history entries.")
        
        if clear_faqs:
            faq_count = len(self.faq_cache)
            self.faq_cache = {}
            self._faq_embedding_matrix = None
            self._faq_cache_ids = []
            self._save_faq_cache()
            logger.warning(f"Cleared {faq_count} static FAQ entries from cache!")

    def stats(self) -> Dict[str, Any]:
        """
        Get cache performance and storage metrics.
        """

        faq_size = self.faq_cache_file.stat().st_size if self.faq_cache_file.exists() else 0
        history_size = self.history_cache_file.stat().st_size if self.history_cache_file.exists() else 0
        
        faqs_ready = sum(1 for e in self.faq_cache.values() if e.get('has_response'))
        faqs_pending = len(self.faq_cache) - faqs_ready
        
        self._cleanup_expired_history()
        
        stat_data = {
                        'total_cached': len(self.faq_cache) + len(self.history_cache),
                        'faq_count': len(self.faq_cache),
                        'faq_ready': faqs_ready,
                        'faq_pending': faqs_pending,
                        'history_count': len(self.history_cache),
                        'history_ttl_hours': self.history_ttl_hours,
                        'similarity_threshold': self.similarity_threshold,
                        'cache_size_kb': (faq_size + history_size) / 1024
                        }
        
        logger.debug(f"Generated cache statistics: {stat_data}")
        return stat_data

    def get_history_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent queries from history (within TTL).
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of query info sorted by recency
        """

        self._cleanup_expired_history()
        
        entries = [
            {
                'query': entry['query'],
                'timestamp': datetime.fromtimestamp(entry['timestamp']).isoformat(),
                'age_hours': (time.time() - entry['timestamp']) / 3600
            }
            for entry in self.history_cache.values()
        ]
        
        entries.sort(key=lambda x: x['age_hours'])
        logger.debug(f"Retrieved {len(entries[:limit])} history query items.")
        return entries[:limit]

    def __len__(self) -> int:
        return len(self.faq_cache) + len(self.history_cache)

    def __contains__(self, query: str) -> bool:
        return self.get(query) is not None