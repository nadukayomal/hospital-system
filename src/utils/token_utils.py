import os
import sys
import logging
import re
import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def count_tokens(text: str, model: str="gpt-4") -> int:
    """
    This function execute how many tokens are in the text.
    Return the int value (token length) 
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        logger.info(f"encoding by mode : {model}")
    except Exception as e:
        encoding = tiktoken.get_encoding("cl100k_base")
        logger.info(f"encoding by mode : cl100k_base ")

    num_of_token = len(encoding.encode(text))
    logger.info(f"There are {num_of_token} tokens.")
    return num_of_token

