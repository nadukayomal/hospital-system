"""
JavaScript-aware Web Crawler (BFS + Playwright)

A polite BFS web crawler that renders JavaScript-heavy pages (React/SPA)
using Playwright and extracts clean structured content.

Main Features:
    - Full JS rendering with headless Chromium
    - BFS crawling with max depth control
    - URL filtering (domain, exclude patterns, media files)
    - Extracts: title, headings, clean markdown, internal links
    - Supports both async (crawl_async) and sync (crawl) usage

Core Methods:
    - should_crawl()     : URL filtering rules
    - extract_content()  : Clean content extraction
    - crawl_async()      : Main async crawler
    - crawl()            : Sync wrapper (Jupyter-friendly)

Usage:
    crawler = WebCrawler(base_url="https://example.com", max_depth=2)
    docs = crawler.crawl(["https://example.com/docs"])
"""

import os
import sys
import logging
import re
import asyncio

from typing import List, Dict, Any, Set
from collections import deque
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from markdownify import markdownify as md

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



class WebCrawler:
    """
    Async web crawler using Playwright for JavaScript-rendered content.
    
    Features:
        - Respects depth limits and exclude patterns
        - Handles SPA/React apps with proper JS rendering waits
        - Extracts clean markdown content
        - Discovers internal links for BFS traversal
        - Polite crawling with configurable delays
    """
    def __init__(self, base_url: str, max_depth: int, exclude_patterns: List):
        self.base_url = base_url
        self.max_depth = max_depth
        self.exclude_patterns = exclude_patterns
        self.visited = set()
        self.documents = []

    def should_crawl(self, url: str) -> bool:
        """
        Decide whether a URL should be crawled or skipped.
        based on the condition mention here 'if' statement
        Returns:
            True : Crawl this URL
            False: Skip this URL
        """
        # Skip if already visited the site
        if url in self.visited:
            logger.debug(f"Skipping already visited URL: {url}")
            return False
        
        # Only crawl URLs that belong to the same website
        if not url.startswith(self.base_url):
            logger.debug(f"Skipping external URL: {url}")
            return False
        
        # Skip URLs that match any exclude pattern
        for pattern in self.exclude_patterns:
            if pattern in url:
                logger.debug(f"Skipping excluded pattern '{pattern}' in URL: {url}")
                return False
            
        # Skip media files and downloadable documents
        media_extensions = r'\.(jpg|jpeg|png|gif|pdf|zip|exe)$'
        if re.search(media_extensions, url, re.IGNORECASE):
            logger.debug(f"Skipping media/file URL: {url}")
            return False

        # All checks passed → this URL is good to crawl
        logger.debug(f"URL approved for crawling: {url}")
        return True

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract clean, structured content from a webpage.

        Returns a dictionary with:
            - title     : Page title
            - headings  : List of h1-h4 texts
            - content   : Clean markdown version of the main content
            - links     : List of unique internal links (for further crawling)
        """
        logger.info(f"Starting extracting .........")
        logger.info(f"step 1 : Remove unessential tags \n")

        # Remove unwanted / noisy elements tags
        # Inside this mentioned tag that content is doed not need. messsy data
        noise_tags = ["script", "style", "nav", "footer", "aside", "noscript", "iframe"]
        for element in soup(noise_tags):
            element.decompose()
        logger.debug(f"Removed noise elements from {url}")


        logger.info(f"step 2 : Extract title from the page \n")
        # Extract title : try to get content inside title tag
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        else:
            # Fallback: use the last part of the URL
            title = url.rstrip("/").split("/")[-1] or "Untitled"
        logger.debug(f"Extracted title: {title}")


        logger.info(f"step 3 : Extract all content inside the h tags \n")
        # Extract all headings (h1 to h4)
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4'])]


        logger.info(f"step 4 : Extract all content inside the h tags \n")
        # Extract and clean internal links
        links = set() 

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href:
                continue

            # Convert relative URLs → absolute URLs
            if href.startswith("/"):
                href = self.base_url.rstrip("/") + href
            elif not href.startswith(("http://", "https://")):
                href = urljoin(url, href)

            # Keep only internal links (same domain)
            if not href.startswith(self.base_url):
                continue

            # Remove fragment (#...) and query parameters (?...)
            clean_href = href.split("#")[0].split("?")[0]

            # Skip empty links and the current page itself
            if clean_href and clean_href != url:
                links.add(clean_href)
        logger.debug(f"Found {len(links)} unique internal links")

        # Locate the main content area
        main_content = (
                        soup.find("div", {"id": "root"}) or                     # React apps
                        soup.find("main") or                                    # Semantic HTML
                        soup.find("article") or                                 # Blog / article pages
                        soup.find("div", class_=re.compile(r"content|main|container", re.I)) or
                        soup.body                                               # Last resort
                        )

        logger.info(f"step 5 : Convert html to markdown \n")
        # Convert HTML → Markdown
        if main_content:
            content_md = md(str(main_content), heading_style="ATX")
        else:
            content_md = md(str(soup), heading_style="ATX")
            logger.warning(f"Could not find main content container for {url}")


        logger.info(f"step 6 : cleanning markdown content \n")
        # Final cleanup of the markdown
        # Remove common JavaScript warning messages
        content_md = re.sub(
                            r"You need to enable JavaScript.*?\.",
                            "",
                            content_md,
                            flags=re.IGNORECASE
                            )
        
        # Collapse 3 or more newlines into just 2
        content_md = re.sub(r"\n{3,}", "\n\n", content_md)
        content_md = content_md.strip()
        result = {
                    "title": title,
                    "headings": headings,
                    "content": content_md,
                    "links": list(links)
                    }

        logger.info(f"Successfully extracted content from {url} "
                f"({len(content_md)} chars, {len(links)} links)")

        return result

    async def crawl_async(
                            self, 
                            start_urls: List[str], 
                            request_delay: float = 2.0
                            ) -> List[Dict[str, Any]]:
        """
        Perform a Breadth-First Search (BFS) crawl using Playwright for JavaScript rendering.

        Args:
            start_urls: List of seed URLs to begin crawling from.
            request_delay: Seconds to wait between requests (politeness delay).

        Returns:
            List of document dictionaries containing url, title, content, links, and depth_level.
        """
        logger.info(f"Initialize the BFS queue")

        # Each item in the queue is a tuple: (url, depth)
        queue = deque([(url, 0) for url in start_urls])
        logger.info(f"Starting crawl with {len(start_urls)} seed URLs")

        async with async_playwright() as p:
            logger.info(f"Launching browser......")

            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            # 30 seconds default timeout
            page.set_default_timeout(30000)
            logger.info("Browser launched successfully")

            while queue:
                url, depth = queue.popleft()

                # Skip if depth exceeded or URL should not be crawled
                if depth > self.max_depth or not self.should_crawl(url):
                    logger.debug(f"Skipping URL (depth={depth}): {url}")
                    continue

                try:
                    logger.info(f"[{depth}] Crawling: {url}")
                    self.visited.add(url)

                    # Navigate to the page
                    await page.goto(url, wait_until="domcontentloaded", timeout=60000)

                    # Wait for JavaScript / React content to load
                    try:
                        await page.wait_for_selector("body", timeout=10000)

                        # Extra wait for SPA rendering
                        await page.wait_for_timeout(3000)

                        # Scroll to bottom to trigger lazy-loaded content
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await page.wait_for_timeout(1000)
                    except Exception:

                        # Fallback if selectors fail
                        logger.warning(f"Selector wait failed, using fallback wait for {url}")
                        await page.wait_for_timeout(5000)

                    # Get fully rendered HTML and parse it
                    html = await page.content()
                    soup = BeautifulSoup(html, "html.parser")

                    # Extract structured content
                    doc_data = self.extract_content(soup, url)
                    doc_data["url"] = url
                    doc_data["depth_level"] = depth

                    # Save only if content is substantial
                    content_length = len(doc_data["content"])
                    if content_length >= 100:
                        self.documents.append(doc_data)
                        logger.info(f"Saved document | chars={content_length} | links={len(doc_data['links'])} | {url}")
                    else:
                        logger.warning(f"Skipped (content too short: {content_length} chars) | {url}")

                    # Add new links to the queue (BFS)
                    if depth < self.max_depth:
                        links_added = 0
                        for link in doc_data["links"]:
                            # Avoid adding already visited or already queued URLs
                            already_in_queue = any(item[0] == link for item in queue)
                            if link not in self.visited and not already_in_queue:
                                queue.append((link, depth + 1))
                                links_added += 1

                        if links_added > 0:
                            logger.debug(f"Added {links_added} new URLs to queue (next depth={depth + 1})")

                    logger.info(
                                f"Progress | saved={len(self.documents)} | "
                                f"visited={len(self.visited)} | queue={len(queue)}"
                                )
                    
                    # Polite delay between requests
                    await asyncio.sleep(request_delay)
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg or "net::ERR_" in error_msg:
                        logger.warning(f"Page not found or network error - skipping: {url}")
                    else:
                        logger.error(f"Error while crawling {url}: {error_msg[:150]}")
                    continue

            await browser.close()
            logger.info("Browser closed. Crawl finished.")

        return self.documents

def crawl(
            self, 
            start_urls: List[str], 
            request_delay: float = 2.0
            ) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for async crawl (for Jupyter compatibility).
    
    Args:
        start_urls: List of seed URLs
        request_delay: Seconds between requests
    
    Returns:
        List of crawled documents
    """
    return asyncio.run(self.crawl_async(start_urls, request_delay))