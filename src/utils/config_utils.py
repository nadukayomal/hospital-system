import os
import logging
import yaml
import sys

from pathlib import Path
from dotenv import load_dotenv, dotenv_values

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Current file → go up two levels → enter "config"
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "config"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CONFIG_FILE = BASE_DIR/"config.yaml"
MODEL_FILE = BASE_DIR/"model.yaml"


def load_yaml(file_path: str) -> dict:
    logger.error(f"Started loading {file_path}")
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

def load_api_keys(provider):
    """
    Args: get provider or any key names

    Return: return API key
    """

    use_provider = provider.lower()

    api_key_dict = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "hugging_face": "HF_TOKEN",
    }

    for key, val in api_key_dict.items():
        if key == use_provider:
            logger.info(f"Loaded {use_provider} API key")
            return os.getenv(val)



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

if __name__ == "__main__":
    load_api_keys()