"""
CRAG (Corrective RAG) service with self-correcting retrieval.

Provides:
- CRAGService: Self-correcting RAG with confidence scoring
- Automatic query expansion on low confidence
- Better grounding and reduced hallucinations

Workflow:
    1. Initial retrieval (k=4)
    2. Calculate confidence score
    3. If low: Corrective retrieval (k=8, expanded)
    4. Generate with best evidence

Benefits:
    - Better accuracy for complex queries
    - Reduces hallucinations
    - Automatic self-correction
"""

import os
import sys
import logging
import time
from datetime import time
from typing import Any, Dict, List
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

ROOT_DIR = ROOT_DIR = Path(__file__).resolve().parents[3]

from utils import calculate_confidence, format_docs, load_config
from domain import RAG_TEMPLATE

# Setting up configs
crag_confidence_threshold = load_config().get("crag", {}).get("confidence_threshold", {})
crag_expanded_k = load_config().get("crag", {}).get("expanded_k", {})
top_k_result = load_config().get("retrieval", {}).get("top_k", {})


class CRAGService:
    """
    Corrective RAG service with automatic self-correction.

    Features:
        - Initial retrieval with confidence scoring
        - Automatic corrective retrieval if confidence low
        - Query expansion strategies
        - Detailed metrics for debugging

    Usage:
        service = CRAGService(retriever, llm)
        result = service.generate(query, confidence_threshold=0.6)

        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Confidence: {result['confidence_final']}")
        logger.info(f"Correction applied: {result['correction_applied']}")
    """
    def __init__(
                    self,
                    retriever: VectorStoreRetriever,
                    llm: Any,
                    initial_k: int = top_k_result,
                    expanded_k: int = crag_expanded_k,
                    ) -> None:
        """
        Initialize CRAG service.

        Args:
            retriever: Vector store retriever instance.
            llm: LangChain LLM instance.
            initial_k: Number of docs for initial retrieval.
            expanded_k: Number of docs for corrective retrieval.
        """
        self.retriever = retriever
        self.llm = llm
        self.initial_k = initial_k
        self.expanded_k = expanded_k
        self.prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        logger.info(
            f"Initialized CRAGService with initial_k={self.initial_k} and expanded_k={self.expanded_k}"
        )

    def generate(
                    self,
                    query: str,
                    confidence_threshold: float = crag_confidence_threshold,
                    verbose: bool = True,
                    ) -> Dict[str, Any]:
        """
        Generates answer with CRAG (Corrective RAG).

        Args:
            query (str): User question.
            confidence_threshold (float, optional): Minimum confidence score (0-1). Defaults to CRAG_CONFIDENCE_THRESHOLD.
            verbose (bool, optional): If True, prints execution logs and timing progress. Defaults to True.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - answer (str): Generated answer text.
                - confidence_initial (float): Initial retrieval confidence score.
                - confidence_final (float): Final confidence score post-correction.
                - correction_applied (bool): Whether corrective retrieval was triggered.
                - docs_used (int): Count of documents fed to prompt context.
                - generation_time (float): Elapsed time for generation stage in seconds.
                - evidence_urls (List[str]): Extracted source attribution URLs.
                - evidence (List[Any]): List of raw Document objects retrieved.
        """

        if verbose:
            logger.info(f"Processing Query: '{query}'")
            logger.info(f"Target Confidence Threshold: {confidence_threshold}")

        # STEP 1: INITIAL RETRIEVAL
        if verbose:
            logger.info(f"Executing Initial Retrieval (k={self.initial_k})...")

        # Mutate the retriever search parameter dynamically for initial retrieval
        self.retriever.search_kwargs["k"] = self.initial_k

        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)

        if verbose:
            logger.info(f"Initial Confidence Score: {confidence_initial:.2f}")

        # STEP 2: CHECK CONFIDENCE & EVALUATE CORRECTION

        # Compare initial score against target threshold
        # Decides whether to use initial results directly or trigger extended retrieval.
        if confidence_initial >= confidence_threshold:
            if verbose:
                logger.info("Confidence score is sufficient. Proceeding with initial documents.")

            # Assign initial retrieval output directly to final variables
            final_docs = docs_initial
            confidence_final = confidence_initial
            correction_applied = False
        else:
            if verbose:
                logger.warning(
                    f"Low confidence score ({confidence_initial:.2f} < {confidence_threshold}). "
                    f"Applying Corrective Retrieval..."
                )

            # STEP 3: CORRECTIVE RETRIEVAL (EXPANDED SEARCH)
            if verbose:
                logger.info(f"Executing Corrective Retrieval (k={self.expanded_k}, expanded search)...")

            # Expand the document collection limit on the retriever
            self.retriever.search_kwargs["k"] = self.expanded_k
            # Perform second, broader document fetch operation
            docs_corrected = self.retriever.invoke(query)
            # Recalculate context quality confidence on expanded doc set
            confidence_final = calculate_confidence(docs_corrected, query)

            if verbose:
                logger.info(f"Corrected Confidence Score: {confidence_final:.2f}")
                improvement = (confidence_final - confidence_initial) * 100
                logger.info(f"Confidence improved by {improvement:+.1f}%")

            final_docs = docs_corrected
            correction_applied = True

        # STEP 4: ANSWER GENERATION

        if verbose:
            logger.info("Generating response using final evidence context...")

        start = time.perf_counter()
        context = format_docs(final_docs)

        prompt_input = {"context": context, "question": query}

        # Construct chain pipeline and execute LLM generation
        # Pipes formatted prompt -> LLM inference -> Output string parsing in a single operational step.
        answer = (self.prompt | self.llm | StrOutputParser()).invoke(prompt_input)

        elapsed = time.perf_counter() - start

        if verbose:
            logger.info(f"Response generated in {elapsed:.2f} seconds.")

        # Extract unique source URLs from retrieved document metadata
        # Derives clean source list to provide users clear data provenance and citation links.
        evidence_urls = list(
            set([doc.metadata["url"] for doc in final_docs if "url" in doc.metadata])
        )

        return {
                "answer": answer,
                "confidence_initial": confidence_initial,
                "confidence_final": confidence_final,
                "correction_applied": correction_applied,
                "docs_used": len(final_docs),
                "generation_time": elapsed,
                "evidence_urls": evidence_urls,
                "evidence": final_docs,
                }

    def batch_generate(
                        self,
                        queries: List[str],
                        confidence_threshold: float = crag_confidence_threshold,
                        ) -> List[Dict[str, Any]]:
        """
        Generates RAG responses for multiple queries sequentially.

        Args:
            queries (List[str]): List of user question strings to process.
            confidence_threshold (float, optional): Minimum confidence score threshold. Defaults to crag_confidence_threshold.

        Returns:
            List[Dict[str, Any]]: List of result dictionaries containing answers and metadata for each query.
        """
        logger.info(f"Starting batch generation for {len(queries)} queries.")

        results = []

        for i, query in enumerate(queries, start=1):
            logger.info(f"Batch Item [{i}/{len(queries)}]: Processing query '{query[:40]}...'")

            result = self.generate(query, confidence_threshold, verbose=False)
            results.append(result)

        logger.info(f"Successfully finished batch generation for {len(queries)} queries.")
        return results

    def analyze_confidence(self, query: str) -> Dict[str, Any]:
        """
        Analyzes initial vs expanded confidence metrics without invoking LLM answer generation.

        Args:
            query (str): User question string to analyze.

        Returns:
            Dict[str, Any]: Diagnostic metrics showing confidence performance comparisons.
        """
        logger.info(f"Analyzing retrieval confidence metrics for query: '{query}'")

        # Gathers baseline metrics using standard document retrieval count.
        self.retriever.search_kwargs["k"] = self.initial_k
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)

        # Gathers comparison metrics using expanded document retrieval count.
        self.retriever.search_kwargs["k"] = self.expanded_k
        docs_expanded = self.retriever.invoke(query)
        confidence_expanded = calculate_confidence(docs_expanded, query)

        improvement = confidence_expanded - confidence_initial

        return {
                "query": query,
                "confidence_initial": confidence_initial,
                "confidence_expanded": confidence_expanded,
                "improvement": improvement,
                "docs_initial": len(docs_initial),
                "docs_expanded": len(docs_expanded),
                }
