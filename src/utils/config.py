import os
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
CONFIG_FILE = os.path.join(BASE_DIR, "config.yaml")
MODEL_FILE = os.path.join(BASE_DIR, "model.yaml")


def load_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return {}


def load_config() -> dict:
    return load_yaml(CONFIG_FILE)


def load_models() -> dict:
    return load_yaml(MODEL_FILE)


# Config.yaml getters

def get_provider():
    return load_config().get("provider", {})


def get_llm():
    return load_config().get("llm", {})


def get_embedding():
    return load_config().get("embedding", {})


def get_chunking():
    return load_config().get("chunking", {})


def get_retrieval():
    return load_config().get("retrieval", {})


def get_cag():
    return load_config().get("cag", {})


def get_crag():
    return load_config().get("crag", {})


def get_crawling():
    return load_config().get("crawling", {})


def get_paths():
    return load_config().get("paths", {})


def get_logging():
    return load_config().get("logging", {})


# Model.yaml getters

def get_openrouter():
    return load_models().get("openrouter", {})


def get_openai():
    return load_models().get("openai", {})


def get_anthropic():
    return load_models().get("anthropic", {})


def get_google():
    return load_models().get("google", {})


def get_groq():
    return load_models().get("groq", {})


def get_deepseek():
    return load_models().get("deepseek", {})