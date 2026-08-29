import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from .rag_service import RAGService
from .cag_cache import CAGCache


class CAGService:
    """
    Manages high-throughput response lookup using a static FAQ cache (permanent)
    and a dynamic history cache (24-hour TTL), reducing LLM generation latency
    and API costs for semantically similar user queries.
    """
    def __init__(self, rag_service, cache):
        """
        Initialize CAG service.
        
        Args:
            rag_service: RAGService instance for generation
            cache: CAGCache instance
        """
        self.rag_service = rag_service
        self.cache = cache

        # Hit rate tracking
        self._hits = 0
        self._misses = 0
        self._faq_hits = 0
        self._history_hits = 0

    @property
    def hit_rate(self) -> float:
        """Calculates the current session cache hit rate ratio.

        Returns:
            float: Hit rate ratio between 0.0 and 1.0 (returns 0.0 if no queries processed).
        """
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    """ FAQ Management """

    def load_faqs(self, faq_queries: List[str]) -> int:
        """Loads static FAQ queries into the cache index without generating responses.

        Args:
            faq_queries (List[str]): List of question strings to index into the static FAQ store.

        Returns:
            int: Number of new FAQ questions successfully loaded into the cache store.
        """
        return self.cache.load_faqs(faq_queries)

    def warm_faqs(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Generates RAG responses for all pending FAQ questions missing cached answers.

        Args:
            verbose (bool, optional): If True, prints execution logs and timing progress. Defaults to True.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - warmed (int): Count of FAQ entries processed and cached.
                - total_time (float): Execution duration in seconds.
        """
        pending = self.cache.get_pending_faqs()

        if not pending:
            logger.info("All FAQs already have responses.")
            return {'warmed': 0, 'total_time': 0.0}

        warmed = 0
        start_time = time.perf_counter()

        # Iterate over each pending question string
        for i, query in enumerate(pending):
            if verbose:
                logger.info(f"Warming FAQ [{i + 1}/{len(pending)}]: '{query[:50]}...'")

            rag_result = self.rag_service.generate(query)
            # Save generated response payload back into FAQ cache store
            self.cache.update_faq_response(query, {
                                                    'answer': rag_result['answer'],
                                                    'evidence_urls': rag_result['evidence_urls']}
                                                )

            warmed += 1
        # Compute total elapsed warming time in seconds
        total_time = time.perf_counter() - start_time

        if verbose:
            logger.info(f"FAQ warming complete! Warmed: {warmed} queries in {total_time:.2f}s.")

        return {'warmed': warmed, 'total_time': total_time}

    def list_faqs(self) -> List[Dict[str, Any]]:
        """Retrieves all indexed FAQ entries alongside their current completion status.

        Returns:
            List[Dict[str, Any]]: List of dictionary records containing FAQ metadata and answers.
        """
        logger.debug("Fetching all FAQ records from cache backend.")
        
        # Return complete list of FAQ dict objects from cache manager
        return self.cache.list_faqs()

    """ Generation """

    def generate(self, query, use_cache=True, verbose=True) -> Dict:
        """
        Processes a query using cache lookups first, falling back to full RAG execution on a miss.

        Args:
            query (str): The raw text question submitted by the user.
            use_cache (bool, optional): Controls whether cache lookups and storage are active. Defaults to True.
            verbose (bool, optional): If True, outputs diagnostic logs during lookup steps. Defaults to True.

        Returns:
            Dict[str, Any]: Execution result dictionary containing:
                - answer (str): Generated or retrieved response text.
                - evidence_urls (List[str]): Source attribution links used for answer compilation.
                - cache_hit (bool): True if answer was retrieved from cache, False if generated via RAG.
                - cache_source (Optional[str]): Source of hit ('faq', 'history', or None).
                - generation_time (float): Time taken to fulfill the request in seconds.
                - similarity_score (float, optional): Cosine similarity score of matched entry (cache hits).
                - matched_query (str, optional): Original cached query string matched (cache hits).
                - num_docs (int, optional): Number of context documents retrieved (RAG misses).
        """
        if verbose:
            logger.info(f"Processing query: '{query}' | Target Similarity Threshold: {self.cache.similarity_threshold}")

        # Execute cache lookup sequence if caching parameter is enabled
        if use_cache:
            # Capture start time of cache vector search
            lookup_start = time.perf_counter()

            # Perform semantic cosine similarity lookup against indexed vectors
            cached = self.cache.get(query)

            # Measure exact lookup duration
            lookup_time = time.perf_counter() - lookup_start

            if cached:
                self._hits += 1

                # Extract hit metadata attributes from returned payload
                source = cached.get('source', 'unknown')
                similarity = cached.get('similarity_score', 1.0)
                matched_query = cached.get('matched_query', query)

                # Categorize hit counts by tier type
                if source == 'faq':
                    self._faq_hits += 1
                else:
                    self._history_hits += 1

                if verbose:
                    logger.info(
                        f"CACHE HIT [{source.upper()}] | Similarity: {similarity:.3f} | "
                        f"Lookup Time: {lookup_time * 1000:.1f}ms"
                    )

                    if matched_query.lower().strip() != query.lower().strip():
                        display_q = matched_query[:60] + "..." if len(matched_query) > 60 else matched_query
                        logger.info(f"Matched Cached Entry: \"{display_q}\"")

                # Return immediate structured response from cache payload
                return {
                        'answer': cached['answer'],
                        'evidence_urls': cached.get('evidence_urls', []),
                        'cache_hit': True,
                        'cache_source': source,
                        'generation_time': lookup_time,
                        'lookup_time': lookup_time,
                        'similarity_score': similarity,
                        'matched_query': matched_query
                    }

            # IF match falls below threshold (CACHE MISS)
            self._misses += 1
            if verbose:
                logger.info(f"CACHE MISS | Lookup Time: {lookup_time * 1000:.1f}ms | Invoking RAG pipeline...")

        # Execute fall-back RAG generation pipeline
        rag_result = self.rag_service.generate(query)

        result = {
                    'answer': rag_result['answer'],
                    'evidence_urls': rag_result['evidence_urls'],
                    'cache_hit': False,
                    'cache_source': None,
                    'generation_time': rag_result['generation_time'],
                    'num_docs': rag_result['num_docs']
                    }

        # Store generated RAG output to dynamic history cache (24h TTL)
        if use_cache:
            cache_start = time.perf_counter()
            self.cache.set(query, result)
            cache_time = time.perf_counter() - cache_start

            if verbose:
                logger.debug(f"Saved query output to dynamic history in {cache_time * 1000:.1f}ms.")

        return result

    """ History Management """

    def get_recent_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recently executed user queries stored within the history cache.

        Args:
            limit (int, optional): Maximum number of query records to return. Defaults to 50.

        Returns:
            List[Dict[str, Any]]: List of query metadata records ordered by recency.
        """

        logger.debug(f"Retrieving last {limit} queries from history store.")
        
        # Delegate retrieval to underlying cache store
        return self.cache.get_history_queries(limit)

    def warm_cache(self, queries: List[str], verbose: bool = True) -> Dict[str, Any]:
        """
        Pre-populates the dynamic history cache by executing RAG over a batch of queries.

        Args:
            queries (List[str]): List of candidate user questions to pre-cache.
            verbose (bool, optional): If True, displays generation progress. Defaults to True.

        Returns:
            Dict[str, Any]: Metrics payload detailing:
                - cached (int): Count of newly cached queries.
                - skipped (int): Count of queries skipped (already present in cache).
                - total_time (float): Aggregate execution runtime in seconds.
        """
        cached_count = 0
        skipped_count = 0
        start_time = time.perf_counter()

        # Iterate over provided list of candidate queries
        for i, query in enumerate(queries):
            if verbose:
                logger.info(f"Warming history cache [{i + 1}/{len(queries)}]: '{query[:50]}...'")

            # Check if query vector already exists in cache index
            if query not in self.cache:
                # Run generate() without recursive logging to execute RAG and populate cache
                self.generate(query, use_cache=True, verbose=False)
                cached_count += 1
            else:
                skipped_count += 1
                if verbose:
                    logger.info(f"Query already cached, skipping execution: '{query[:30]}...'")

        total_time = time.perf_counter() - start_time

        if verbose:
            logger.info(
                f"History warming complete! Cached: {cached_count}, Skipped: {skipped_count}, "
                f"Total Time: {total_time:.2f}s."
            )

        return {
            'cached': cached_count,
            'skipped': skipped_count,
            'total_time': total_time
        }

    """ Statistics & Management """

    def cache_stats(self) -> Dict[str, Any]:
        """
        Gathers unified performance metrics covering overall cache health and session hits.

        Returns:
            Dict[str, Any]: Performance metrics dictionary containing cache storage counts,
            session hit/miss statistics, and calculated session hit rate percentages.
        """
        # Fetch base storage statistics dictionary from underlying CAGCache
        stats = self.cache.stats()

        # Inject current runtime session metric attributes into result dictionary
        stats['session_hits'] = self._hits
        stats['session_misses'] = self._misses
        stats['session_faq_hits'] = self._faq_hits
        stats['session_history_hits'] = self._history_hits
        stats['session_hit_rate'] = f"{self.hit_rate:.1%}"

        logger.debug(f"Retrieved cache stats summary. Session Hit Rate: {stats['session_hit_rate']}")
        return stats

    def clear_cache(self, clear_faqs: bool = False) -> None:
        """
        Flushes dynamic entries from the cache index.

        Args:
            clear_faqs (bool, optional): If True, removes static FAQ entries alongside dynamic history.
                Defaults to False (preserving static FAQ entries).
        """
        logger.warning(f"Clearing cache index entries (clear_faqs={clear_faqs}).")

        self.cache.clear(clear_faqs=clear_faqs)

    def reset_stats(self) -> None:
        """Resets all internal session counters for hits, misses, and hit breakdowns back to zero."""
        
        logger.info("Resetting session hit/miss counter metrics back to zero.")
        
        # Re-initialize integer metric counters to 0
        self._hits = 0
        self._misses = 0
        self._faq_hits = 0
        self._history_hits = 0   