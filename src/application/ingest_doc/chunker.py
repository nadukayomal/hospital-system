import os
import sys
import logging
import re

from utils import *

from typing import List, Dict, Any, Optional, Tuple
from langchain_text_splitters import (
                                        MarkdownHeaderTextSplitter,
                                        RecursiveCharacterTextSplitter
                                        )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)




def sliding_chunk(documents: List, sliding_window_size: int, sliding_stride_size: int) -> List:
    """
    Getting document return the chunkpeas with add metadata
    Args: 
        document: chunking document 
        sliding_window_size: chunk size 
        sliding_stride_size: overlap size

    Return:
        return list of chunks
    """
    chunks = []
    chunk_idx = []

    window_size = sliding_window_size * 4
    stride_size = sliding_stride_size * 4

    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']

        # Simple sliding window over content 
        # The pos is: current positions of character where sliding window start
        # The window_idx: is counting inside the document
        pos = 0
        window_idx = 0
        content_len = len(content)

        # Cut one document content in to chunk until reach content length
        while pos < content_len:
            # Getting the index where the end chunk from content
            end = min(pos + window_size, content_len)
            window_text = content[pos:end]

            if window_text.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": window_text.strip(),
                    "strategy": "sliding",
                    "chunk_index": chunk_idx,
                    "window_index": window_idx,
                    "overlap_tokens": stride_size if window_idx > 0 else 0
                })
                chunk_idx += 1
                window_idx += 1

            # Increate pos valuve from which index to start to chunk next
            pos += stride_size
            if pos >= content_len:
                break

    return chunks 


def parent_child_chunk(
                        documents: List, 
                        parent_chunk_size: int, 
                        child_chunk_size: int, 
                        child_overlap: int
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    """
    Create parent-child chunk from the given document
    Steps:
        - Loading document
        - Create parent chunk
        - Each parent create many child chunk
        - Retrivel : fetch children -> return parent context

    Return:
        Tuple of (children_chunks, parent_chunks)
        Children have 'parent_id' field linking to parent
    """

    parent_chunks = []
    child_chunks = []
    parent_idx = 0
    child_idx = 0

    parent_size = parent_chunk_size * 4
    child_size = child_chunk_size * 4
    child_overlap = child_overlap * 4

    # Split parent chunk
    parent_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size = parent_size,
                                                    chunk_overlap = 200,
                                                    length_function = len
                                                    )

    # Child splitter
    child_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size = child_size,
                                                    chunk_overlap = child_overlap,
                                                    length_function = len
                                                    )
    
    for doc in documents:
        content = doc["content"]
        url = doc["url"]
        title = doc["title"]

        # Create parent chunk
        parent_texts = parent_splitter.split_text(content)

        for parent_text in parent_texts:
            if not parent_texts.strip():
                continue
            parent_id = f"{url}::parent::{parent_idx}"

            # Store parent
            parent_chunks.append({
                                    "parent_id": parent_id,
                                    "url": url,
                                    "title": title,
                                    "text": parent_text.strip(),
                                    "strategy": "parent",
                                    "chunk_index": parent_idx,
                                    "token_count": count_tokens(parent_text)
                                    })

            # Create child chunk for this parent
            child_texts = child_splitter.split_text(parent_texts)

            for child_text in child_texts:
                if child_text.strip():
                    child_chunks.append({
                                        "child_id": f"{parent_id}::child::{child_idx}",
                                        "parent_id": parent_id,
                                        "url": url,
                                        "title": title,
                                        "text": child_text.strip(),
                                        "strategy": "child",
                                        "chunk_index": child_idx,
                                        "token_count": count_tokens(child_text)
                                        })
                    child_idx += 1
            parent_idx += 1

    return child_chunks, parent_chunks


def late_chunk_split(passage: str, query: str) -> List[Dict[str, Any]]:
    """
    Get user query and relevant passage
    Return most appropriate chunks.

    Args:
        passage: The base passage text
        query: User query
    Returns:
        List of smaller chunks around query matches
        
    """

    # Find query term positions
    query_terms = query.lower().split()
    passage_lower = passage.lower()

    # Find all match positions
    match_positions = []

    # Search one term are in the passeage
    # Find every place it appears in the passage and store the character index
    for term in query_terms:
        pos = 0
        while True:
            # Find matching terms until end of the passage
            pos = passage_lower.find(term, pos)
            # If position -1  last word in passage. then break
            if pos == -1:
                break
            match_positions.append(pos)
            pos += len(term)

    if not match_positions:
        logger.info("No matching words found.")
        # Return full passage as one chunk 
        return [{"text": passage, "score": 0.0}]

    # Create chunk around matching
    chunks = []
    context_length = (get_chunking().get("late", {}).get("context_window", {})) * 4
    chunk_size = (get_chunking().get("late", {}).get("split_size", {})) * 4
    logger.info(f"Initialize late chunking config -> context window : {context_length} , chunk size : {chunk_size}")

    for match_pos in match_positions:
        # Extract context around match
        start = max(0, match_pos - context_length)
        end = min(len(passage), match_pos + chunk_size)

        # Slice the range as one small chunk 
        chunk_text = passage[start:end].strip()

        # Calculate relevance score
        score = 1.0 if match_pos in match_positions else 0.5

        chunks.append({
                        "text": chunk_text,
                        "match_position": match_pos,
                        "score": score
                        })
    
    # Dedublicate overlapping chunk 
    unique_chunks = []
    seen_texts = set()
    for chunk in sorted(chunks, key=lambda x: x['score'], reverse=True):
        if chunk["text"] not in seen_texts:
            unique_chunks.append(chunk)
            seen_texts.add(chunk['text'])

    return unique_chunks[:5]







