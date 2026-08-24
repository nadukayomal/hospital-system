# Hospital System: Adaptive RAG

An advanced, adaptive Retrieval-Augmented Generation (RAG) system built for hospital and clinical knowledge retrieval. This project combines standard RAG with Cache-Augmented Generation (CAG) and Corrective RAG (CRAG), augmented by live medical web scraping.

## 🚀 Features Built So Far

The project is actively under development. Currently, the foundational architecture, configuration management, and LLM integrations are complete.

### 1. Modular Architecture Setup
The codebase is structured following Domain-Driven Design (DDD) principles:
- **`src/application/`**: Use cases and application logic.
- **`src/domain/`**: Core business entities and logic.
- **`src/infrastructure/`**: External services, API integrations, and database access.
- **`src/utils/`**: Shared utilities, formatters, and configuration handlers.

### 2. Configuration Management
- **Centralized Settings**: Robust configuration loading via `utils/config_utils.py`.
- **YAML Driven**: Configured using modular YAML files (`config.yaml`, `model.yaml`, `faq.yaml`) to handle system behavior, model routing, and hyperparameter tuning (chunking, retrieval, CAG, CRAG).
- **Environment Management**: Secure API key loading from `.env`.

### 3. LLM & Embeddings Infrastructure
- **LLM Service (`llm_service.py`)**: Factory implementations for `ChatOpenAI` integrated with LangChain. Supports multiple performance tiers (`general`, `strong`, `reasoning`) to intelligently route queries to the right model (e.g., GPT-4o-mini vs. GPT-4o vs. o3-mini/DeepSeek).
- **Embeddings (`embeddings.py`)**: Centralized embedding initialization configured to use OpenAI embeddings either directly or via OpenRouter.
- **Provider Agnostic**: Configured to seamlessly switch between direct APIs (OpenAI, Anthropic, Google, DeepSeek) and unified routing via **OpenRouter**.

### 4. Dependency & Environment Management
- **Modern Packaging**: Managed via `pyproject.toml` and `uv` lockfiles for deterministic builds.
- **Rich Tech Stack Pre-configured**: LangChain, ChromaDB, Sentence Transformers, Playwright, BeautifulSoup, and Pydantic are successfully integrated.
- **Code Quality Tools**: Configured `ruff`, `black`, `mypy`, and `pytest` for rigorous code linting, formatting, and testing.

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **LLM Framework**: LangChain Ecosystem
- **Vector Store**: ChromaDB
- **Web Scraping**: Playwright, BeautifulSoup4, Markdownify
- **Data Handling**: Pandas, Numpy, Pydantic
- **Development Tools**: `uv`, Black, Ruff, MyPy, Pytest

## 📁 Project Structure

```text
hospital-system/
├── config/              # YAML configuration files (config, model, faq)
├── data/                # Data storage (vectorstore, markdown, cache)
├── notebooks/           # Jupyter notebooks for experimentation
├── src/                 # Main source code
│   ├── application/     # Application logic (pending)
│   ├── domain/          # Core domain models (pending)
│   ├── infrastructure/  # External integrations (LLM Providers completed)
│   └── utils/           # Helper scripts (Config, Formatting)
├── pyproject.toml       # Dependencies and build settings
└── uv.lock              # UV lock file for dependency locking
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
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key
   OPENROUTER_API_KEY=your_openrouter_key
   ```

## 🗺️ Roadmap

- [ ] Implement chunking strategy and document processing pipeline.
- [ ] Integrate ChromaDB vector database for embeddings storage.
- [ ] Build the web scraping pipeline for live medical data integration.
- [ ] Develop the Core RAG retrieval logic.
- [ ] Implement CRAG (Corrective RAG) for hallucination prevention and self-correction.
- [ ] Implement CAG (Cache-Augmented Generation) for faster response times on FAQs.
- [ ] Build the API layer (FastAPI) and Monitoring dashboards.

## 📄 License

This project is licensed under the MIT License.
