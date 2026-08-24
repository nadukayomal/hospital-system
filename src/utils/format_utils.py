"""
This is include helper function for RAG
    - Document formatting
    - Confidence scoring
    - Citation utilities
"""

import os
import sys
import re
import logging

from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



def format_docs(docs: list) -> str:
    """
    Format list of Documents into a single context string.
    
    Args:
        docs: List of LangChain Document objects
    
    Returns:
        Formatted context string with source URLs
    """
    logger.info(f"Formatting {len(docs)} documents into context.")

    formatted = []
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get('url', 'N/A')
        title = doc.metadata.get('title', 'N/A')
        content = doc.page_content[:500]  # First 500 chars

        formatted.append(
                        f"[Source {i}: {url}]\n"
                        f"Title: {title}\n"
                        f"Content: {content}\n"
                        )
      
    context = "\n---\n".join(formatted)
    logger.debug(f"Formatted context length: {len(context)} characters")
    return context


def calculate_confidence(docs: list, query: str) -> float:
    """
    Calculate confidence score for retrieved documents.
    
    Multi-factor heuristic:
    1. Keyword overlap (query ∩ docs)
    2. Content richness (avg doc length)
    3. Strategy diversity (multiple chunking strategies)
    
    Args:
        docs: List of retrieved documents
        query: User query string
    
    Returns:
        Confidence score 0.0 to 1.0
    """
    if not docs:
        logger.warning("calculate_confidence called with empty document list")
        return 0.0
    
    # Extract quary keywords
    query_words = set(query.lower().split())
    overlaps = []

    # Measures how many words from the user's query appear in the documents.
    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
        overlaps.append(overlap)
    keyword_score = sum(overlaps) / len(overlaps)

    # Checks the average length of the retrieved documents.
    avg_length = sum(len(doc.page_content) for doc in docs) / len(docs)
    length_score = min(avg_length / 500, 1.0)

    # Checks whether the documents were created using different chunking strategies.
    strategies = {doc.metadata.get('strategy', 'unknown') for doc in docs}
    diversity_score = min(len(strategies) / 3.0, 1.0)  # max 3 strategies

    # Final Confidence Score
    # Give more importance to keyword relevance (50%),
    # then content length (30%), and strategy diversity (20%).
    confidence = (
                    0.5 * keyword_score +
                    0.3 * length_score +
                    0.2 * diversity_score
                )
    
    logger.info(
                f"Confidence Breakdown -> "
                f"Keyword: {keyword_score:.3f} | "
                f"Length: {length_score:.3f} | "
                f"Diversity: {diversity_score:.3f} | "
                f"Final: {confidence:.3f}"
                )

    return confidence


def extract_citations(text: str) -> List:
    """
        - Extract all cited URLs from the generated answer.
        - The LLM is expected to include sources in square brackets like
        - This function finds those citations and returns only valid URLs.
    """
    if not text:
        logger.warning("There are no texts to citations.")
        return []

    # Find every piece of text that appears inside square brackets
    citations = re.findall(r'\[([^\]]+)\]', text)

    urls = [
            citation.strip()
            for citation in citations
            if 'http' in citation or '.com' in citation or '.org' in citation
            ]

    # Remove duplicates while preserving order
    unique_urls = list(dict.fromkeys(urls))

    if unique_urls:
        logger.info(f"Extracted {len(unique_urls)} unique citation(s) from the answer")
        logger.debug(f"Citations found: {unique_urls}")
    else:
        logger.info("No valid URL citations found in the generated text")

    return unique_urls



def truncate_text(text: str, max_length: int = 400) -> str:
    """
    Cleanly truncate long text for previews or display.
    
    If the text is longer than max_length, it cuts at the nearest
    word boundary and adds "..." at the end. This avoids breaking
    words in the middle.
    """

    if not text:
        return ""

    if len(text) <= max_length:
        return text

    # Cut the text and try to end at the last complete word
    truncated = text[:max_length].rsplit(' ', 1)[0] + "..."
    logger.debug(f"Text truncated from {len(text)} to {len(truncated)} characters")

    return truncated



    




