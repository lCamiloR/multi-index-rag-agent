from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AgentConfig(BaseSettings):
    # Automatically load environment variables from the project root `.env` file
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
    )

    ANTHROPIC_API_KEY: str = Field(..., description="Anthropic API key.")
    ANTHROPIC_MODEL_VERSION: str = Field(..., description="Model version to use for the agent.")
    EMBEDDING_MODEL: str = Field(..., description="Model to be used for embeddings.")
    FAISS_INDEXING_PATH: Path = PROJECT_ROOT / "faiss_index"
    ASSETS_PATH: Path = PROJECT_ROOT / "assets"

AGENT_CONFIG = AgentConfig()