# Hospital System: Adaptive RAG

An advanced, adaptive Retrieval-Augmented Generation (RAG) system built for hospital and clinical knowledge retrieval. This project combines standard RAG with Cache-Augmented Generation (CAG) and Corrective RAG (CRAG), augmented by live medical web scraping.

## 🚀 Features & Implementation Details

The project is actively under development, and several core components have now been built and integrated into the architecture.

### 1. Adaptive RAG Patterns
We employ a multi-tiered pattern approach to retrieval to maximize accuracy and minimize latency:
- **Standard RAG (`rag_service.py`)**: Built using LangChain Expression Language (LCEL) for seamless parallel retrieval, context formatting, and LLM generation. 
- **CAG - Cache-Augmented Generation (`cag_cache.py`, `cag_service.py`)**: A semantic, cosine similarity-based cache that stores embeddings. It features two tiers:
  - *Static FAQs*: Predefined queries that never expire.
  - *Dynamic History*: User queries stored with a configurable TTL (e.g., 24 hours).
  Provides near-zero latency and zero API cost for repeated or semantically similar queries.
- **CRAG - Corrective RAG (`crag_service.py`)**: Implements self-correcting retrieval. It performs an initial retrieval (k=4) and calculates a heuristic confidence score based on keyword overlap, content richness, and chunk diversity. If the score is low, it triggers a broader, corrective retrieval (k=8) to prevent hallucinations and improve grounding.

### 2. Document Processing & Chunking Methods (`chunker.py`)
To handle complex medical documents effectively, multiple chunking strategies are implemented:
- **Sliding Window Chunking**: Simple chunks with overlapping strides to retain context boundaries.
- **Parent-Child Chunking**: Creates broad "parent" chunks and detailed "child" chunks. Retrieval fetches children but provides the parent context to the LLM.
- **Late Chunking**: Indexes large base passages, and splits them into smaller, highly relevant chunks dynamically at retrieval time based on query term matching.

### 3. Web Scraping Technology (`web_crawler.py`)
- **JavaScript-Aware Crawling**: Uses **Playwright** (headless Chromium) to fully render React/SPA pages before extraction.
- **Content Extraction**: Employs **BeautifulSoup4** to strip noise tags (scripts, nav, footers) and **Markdownify** to convert clean HTML structures into structured Markdown content. Includes BFS (Breadth-First Search) for polite, depth-controlled site traversal.

### 4. LLM & Embeddings Infrastructure
- **LLM Providers (`llm_service.py`)**: Factory implementations for chat models supporting multiple performance tiers (`general`, `strong`, `reasoning`). Configured to route queries via **OpenAI** or unified routing via **OpenRouter**.
- **Embeddings (`embeddings.py`)**: Centralized embedding initialization configured for OpenAI text-embedding models.
- **Vector Database**: Designed to integrate with **ChromaDB** for fast, local vector storage and retrieval.

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **LLM Framework**: LangChain Ecosystem (LCEL, Core)
- **Vector Store**: ChromaDB
- **Web Scraping**: Playwright, BeautifulSoup4, Markdownify
- **Data Handling**: Pandas, Numpy, Pydantic, Tiktoken
- **Development Tools**: `uv` (dependency locking), Black, Ruff, MyPy, Pytest

## 📁 Folder Structure

```text
hospital-system/
├── config/
│   ├── config.yaml          # Core hyperparameters and settings
│   ├── faq.yaml             # Static pre-defined questions
│   └── model.yaml           # Provider, model tiers, and definitions
├── data/                    # Data storage (vectorstore, markdown, cache)
├── notebooks/               # Jupyter notebooks for experimentation
├── src/
│   ├── application/
│   │   ├── __init__.py
│   │   ├── chat_service/    # AI response generation services
│   │   │   ├── __init__.py
│   │   │   ├── cag_cache.py
│   │   │   ├── cag_service.py
│   │   │   ├── crag_service.py
│   │   │   └── rag_service.py
│   │   └── ingest_doc/      # Web scraping and data extraction
│   │       ├── __init__.py
│   │       ├── chunker.py
│   │       └── web_crawler.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── chunk/           # Advanced chunking implementations
│   │   │   ├── __init__.py
│   │   │   ├── chunk_class.py
│   │   │   └── chunker.py
│   │   ├── prompt/          # Standardized RAG prompts
│   │   │   ├── __init__.py
│   │   │   └── rag_templates.py
│   │   └── schema/          # Data models and structures
│   │       ├── __init__.py
│   │       └── models.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── api/             # API Endpoints (FastAPI)
│   │   ├── db/              # Database connections
│   │   ├── llm_providers/   # Vendor API integrations
│   │   │   ├── __init__.py
│   │   │   ├── embeddings.py
│   │   │   └── llm_service.py
│   │   └── monitoring/      # Observability and logging
│   └── utils/
│       ├── __init__.py
│       ├── config_utils.py  # Helpers to read YAML configs
│       ├── format_utils.py  # Helpers for citations and formatting
│       └── token_utils.py   # Helpers for Tiktoken counting
├── .env                     # Environment variables (API keys)
├── README.md                # Project documentation
├── pyproject.toml           # Dependencies and build settings
└── uv.lock                  # UV lock file for dependency locking
```

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nadukayomal/hospital-system.git
   cd hospital-system
   ```

2. **Set up the virtual environment:**
   We recommend using `uv` or standard `venv`.
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -e .[all]
   # Install playwright browsers
   playwright install chromium
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key
   OPENROUTER_API_KEY=your_openrouter_key
   ```

## 🗺️ Roadmap

- [x] Implement advanced chunking strategy and document processing pipeline.
- [x] Build the web scraping pipeline for live medical data integration.
- [x] Develop the Core RAG retrieval logic via LCEL.
- [x] Implement CRAG (Corrective RAG) for hallucination prevention and self-correction.
- [x] Implement CAG (Cache-Augmented Generation) for faster response times on FAQs.
- [ ] Integrate ChromaDB vector database for embeddings storage.
- [ ] Build the API layer (FastAPI) and Monitoring dashboards.

## 📄 License

This project is licensed under the MIT License.
