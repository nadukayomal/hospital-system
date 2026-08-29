import os
import sys
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, Runnable
from langchain_core.vectorstores import VectorStoreRetriever

ROOT_DIR = ROOT_DIR = Path(__file__).resolve().parents[3]

from domain import RAG_TEMPLATE
from utils import  format_docs, load_config

# Initiate configs.
top_k_result = load_config().get("retrieval", {}).get("top_k", {})


def build_rag_chain(
                    retriever: VectorStoreRetriever, 
                    llm: Any, 
                    k: int=top_k_result, 
                    template: str=RAG_TEMPLATE
                    ) -> Runnable:
    """
    Build modern RAG chain using LangChain Expression Language (LCEL) with logging.

    Chain structure:
        1. RunnableParallel: Retrieves docs + passes question through
        2. format_docs: Converts docs to context string
        3. Prompt: Fills template with context + question
        4. LLM: Generates answer
        5. StrOutputParser: Extracts string from LLM response
    
    Args:
        - retriever: VectorStore retriever (from vectorstore.as_retriever())
        - llm: LangChain LLM instance (ChatOpenAI, etc.)
        - k: Number of docs to retrieve (default from config)
        - template: Prompt template string
    
    Returns:
        - Runnable chain that can be invoked with query strin

    """
    logger.info("Constructing LCEL RAG Chain...")

    if k != top_k_result:
        logger.info(f"Overriding retriever search parameter 'k' to: {k}")
        retriever.search_kwargs["k"] = k

    rag_prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        RunnableParallel(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    logger.info("LCEL RAG Chain constructed successfully.")
    return rag_chain


class RAGService:
    """High-level RAG service for question answering with embedded logging."""
    def __init__(
                    self,
                    retriever: VectorStoreRetriever,
                    llm: Any,
                    k: int = top_k_result
                    ):
        """
        Initialize RAG service.
        
        Args:
            retriever: Vector store retriever
            llm: LangChain LLM instance
            k: Number of documents to retrieve
        """

        self.retriever = retriever
        self.llm = llm
        self.k = k
        logger.info(f"Initializing RAGService instance with top_k={k}")
        self.chain = build_rag_chain(retriever, llm, k)

    def generate(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for query using RAG and record performance metrics.
        
        Args:
            query: User question
        
        Returns:
            Dict with:
            - answer: Generated answer string
            - evidence: List of retrieved documents
            - evidence_urls: List of unique source URLs
            - generation_time: Seconds taken
        """
        logger.info(f"Processing generation request for query: '{query}'")
        start = time.time()

        try:
            # Retrieve context evidence
            logger.debug("Executing retriever invoke...")
            evidence = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(evidence)} evidence documents.")

            # Run answer generation
            logger.debug("Invoking LCEL chain generation...")
            answer = self.chain.invoke(query)

            elapsed = time.time() - start

            # Extract unique URLs safely
            evidence_urls = list(set([
                doc.metadata.get('url', 'N/A') for doc in evidence if hasattr(doc, 'metadata')
            ]))

            logger.info(
                f"Generation complete in {elapsed:.2f}s | "
                f"Docs: {len(evidence)} | Unique URLs: {len(evidence_urls)}"
            )

            return {
                    'answer': answer,
                    'evidence': evidence,
                    'evidence_urls': evidence_urls,
                    'generation_time': elapsed,
                    'num_docs': len(evidence)
                    }
        except Exception as e:
            logger.error(f"Failed during RAG generation for query '{query}': {e}", exc_info=True)
            raise

    def stream(self, query: str):
        """
        Stream answer generation token-by-token.

        Args:
            query: User question
        
        Yields:
            String chunks as they're generated

        """

        logger.info(f"Initiating response stream for query: '{query}'")
        chunk_count = 0
        try:
            for chunk in self.chain.stream(query):
                chunk_count += 1
                yield chunk
            logger.debug(f"Streaming completed successfully. Total chunks emitted: {chunk_count}")
        except Exception as e:
            logger.error(f"Error during streaming execution: {e}")
            raise

    def batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries in batch.

        Args:
            queries: List of user questions
        
        Returns:
            List of result dicts (same format as generate())

        """
        logger.info(f"Starting batch generation for {len(queries)} queries.")
        results = []
        for idx, query in enumerate(queries, 1):
            logger.info(f"Executing batch item [{idx}/{len(queries)}]")
            results.append(self.generate(query))
        logger.info("Completed all batch generation items.")
        return results