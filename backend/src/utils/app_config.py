import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Base directory for the backend
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

class AppConfig:
    @staticmethod
    def load_yaml(file_path):
        full_path = BASE_DIR / "config" / file_path
        if not full_path.exists():
            return {}
        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def get_llm_config(cls):
        return cls.load_yaml("llms.yaml")

    @classmethod
    def get_agents_config(cls):
        return cls.load_yaml("agents.yaml")

    # Access environment variables easily
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
