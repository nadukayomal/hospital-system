"""
Module: document_chunking.py
Description: Document chunking strategies for RAG & text-processing pipelines.
              Provides semantic, fixed-size, sliding-window, parent-child, 
              and late-chunking implementations.
"""

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



def semantic_chunk(documents: List) -> List:
    """
    Split document by markdown heading structure.
    """
    logger.info(f"Starting semantic chunk with : {documents}")

    # Initialize configs
    semantic_min_chunk_size = get_chunking().get("semantic", {}).get("min_chunk_size", {})
    semantic_max_chunk_size = get_chunking().get("semantic", {}).get("max_chunk_size", {})

    chunks = []
    chunk_idx = 0

    # Define heading hierarchy to filter content
    headers_to_split = [("#", "h1"),
                        ("##", "h2"),
                        ("###", "h3")]

    splitter = MarkdownHeaderTextSplitter(
                                            headers_to_split_on=headers_to_split, 
                                            strip_headers=False
                                            )
    
    # Loop each document in documents
    for doc_idx, doc in enumerate(documents):
        # Extract followings in each document
        content = doc['content']
        url = doc['url']
        title = doc['title']

        try:
            # Split content by headings
            sections = splitter.split_text(content)
            if not sections:
                logger.debug(f"No headings found in document {doc_idx} using full content")
                # Create empty content with empty metadata : create fake object. 
                # Maintain process untill over the number of documents.
                sections = [type('obj', (object,), {'page_content': content, 'metadata': {}})()]
            else:
                logger.debug(f"Document {doc_idx} split into {sections} heading sections")

            for section in sections:
                # Remove whitespace in each section
                text = section.page_content.strip()

                if not text or len(text) < semantic_min_chunk_size:
                    logger.debug(f"Skipping short/empty section {len(text)}")
                    continue

                # Check text is too large or not than max chunk size
                token_count = count_tokens(text)
                if token_count > semantic_max_chunk_size:
                    # Use recursive text aplitter
                    logger.debug("Section is too large apply recursive text splitter")

                    character_size = semantic_max_chunk_size * 4
                    sub_splitter = RecursiveCharacterTextSplitter(
                                                                    chunk_size = character_size,
                                                                    chunk_overlap = 100,
                                                                    length_function = len
                                                                    )
                    sub_chunks = sub_splitter.split_text(text)

                    for sub_text in sub_chunks:
                        # Split oversize ection onto small pieces
                        # Create new chunk dictionary for each-sub piece
                        if sub_text.strip():
                            chunks.append({
                                            "url": url,
                                            "title": title,
                                            "text": sub_text.strip(),
                                            "strategy": "semantic",
                                            "chunk_index": chunk_idx,
                                            "heading": section.metadata.get('h1', '') or section.metadata.get('h2', '')
                                            })
                            chunk_idx += 1
                else:
                    heading = section.metadata.get('h1', '') or section.metadata.get('h2', '')

                    # Creates one clean chunk from the whole section.
                    chunks.append({
                                    "url": url,
                                    "title": title,
                                    "text": text,
                                    "strategy": "semantic",
                                    "chunk_index": chunk_idx,
                                    "heading": section.metadata.get('h1', '') or section.metadata.get('h2', '')
                                    })
                    logger.debug(f"Created semantic Chunk : {chunk_idx} token: {token_count}")
                    chunk_idx += 1
                    
        except Exception as e:
            logger.warning(
                f"Error processing document {doc_idx} title : {title}: {str(e)} - falling back to single chunk"
            )

            # Most probably run this :
            # Markdown broken
            # Splitter crashes
            # Unexpected data type
            # Any other runtime error
            if content.strip():
                chunks.append({
                                "url": url,
                                "title": title,
                                "text": content.strip(),
                                "strategy": "semantic",
                                "chunk_index": chunk_idx,
                                "heading": ""
                                })
                chunk_idx += 1

    logger.info(f"semantic_chunk finished. Created len{chunks} chunks from {len(documents)} documents")
    return chunks


def fixed_chunk(documents: List) -> List:
    """
    Split document into fixed chunk with chunk overlap
    """
    logger.info(f"Starting Fixed chunk with : {documents}")

    # Initialize configs
    chunk_size = get_chunking().get("fixed", {}).get("chunk_size", {})
    overlap_size = get_chunking().get("fixed", {}).get("chunk_overlap", {})
    chunk_size_char = chunk_size * 4
    overlap_char = overlap_size * 4

    chunks = []
    chunk_idx = 0

    splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=chunk_size_char,
                                                chunk_overlap=overlap_char,
                                                length_function=len,
                                                separators=["\n\n", "\n", ". ", " ", ""]
                                                )

    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        # Split content
        doc_chunks = splitter.split_text(content)
        
        for text in doc_chunks:
            if text.strip():
                token_count = count_tokens(text)
                chunks.append({
                                "url": url,
                                "title": title,
                                "text": text.strip(),
                                "strategy": "fixed",
                                "chunk_index": chunk_idx,
                                "token_count": token_count,
                                "overlap_tokens": overlap_size
                                })
                chunk_idx += 1
                
    return chunks


def sliding_chunk(documents: List) -> List:
    """
    Getting document return the chunkpeas with add metadata
    Args: 
        document: chunking document 
        sliding_window_size: chunk size 
        sliding_stride_size: overlap size

    Return:
        return list of chunks
    """
    # Initialize config
    sliding_window_size = get_chunking().get("sliding", {}).get("window_size", {})
    sliding_stride_size = get_chunking().get("sliding", {}).get("stride_size", {})

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


def parent_child_chunk(documents: List) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

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
    # Initialize config
    parent_chunk_size = get_chunking().get("parent_child", {}).get("parent_size", {})
    child_chunk_size = get_chunking().get("parent_child", {}).get("child_size", {})
    child_overlap = get_chunking().get("parent_child", {}).get("child_overlap", {})

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


def late_chunk_index(documents: List) -> List:
    """
    Create large base passage for index
    working steps:
        1. index large passage by late_chunk_base_size token.
        2. on retrieval, split near query matches into similer chunk.
        3. provides tighter quotes without exploding index size.
    Usage:
        Need precision without pre-micro chunking everything.
    """

    logger.info(f"Starting late chunk indexing on : {documents}")

    chunks = []
    chunk_idx = 0

    # Large base passage
    # Convert token target to approximate characters
    base_size = (get_chunking().get("late", {}).get("base_size", {})) * 4
    logger.info(f"Initiate base size in late index : {base_size}")

    splitter = RecursiveCharacterTextSplitter(
                                                chunk_size = base_size,
                                                chunk_overlap = 100,
                                                length_function = len
                                                )

    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']

        # Create base passage
        passages = splitter.split_text(content)

        for passage in passages:
            if passage.strip():
                chunks.append({
                                "url": url,
                                "title": title,
                                "text": passage.strip(),
                                "strategy": "late_chunk_base",
                                "chunk_index": chunk_idx,
                                "token_count": count_tokens(passage),
                                "splittable": True
                                })
                chunk_idx += 1

    return chunks

    
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







