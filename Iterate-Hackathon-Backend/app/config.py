# app/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")

    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_db_name: str = Field("chat_history", env="MONGODB_DB_NAME")
    mongodb_collection_name: str = Field("message_store", env="MONGODB_COLLECTION_NAME")

    claude_model: str = Field(
        "claude-haiku-4-5-20251001",
        env="CLAUDE_MODEL",
    )
    
    claude_code_exec_model: str = Field(
        "claude-sonnet-4-5-20250929",
        env="CLAUDE_CODE_EXEC_MODEL",
        description="Model to use for code execution analysis (Sonnet 4.5 recommended)",
    )

    # Agent execution settings
    agent_timeout_seconds: float = Field(
        30.0,
        env="AGENT_TIMEOUT_SECONDS",
        description="Timeout for individual agent calls in seconds",
    )
    agent_max_retries: int = Field(
        2,
        env="AGENT_MAX_RETRIES",
        description="Maximum retry attempts for failed agent calls",
    )
    agent_max_dataset_rows: int = Field(
        100000,
        env="AGENT_MAX_DATASET_ROWS",
        description="Maximum dataset size (rows) for agent processing",
    )
    agent_sample_rows: int = Field(
        1000,
        env="AGENT_SAMPLE_ROWS",
        description="Maximum rows to send to Claude for code generation (larger datasets will be truncated)",
    )
    agent_enabled: bool = Field(
        True,
        env="AGENT_ENABLED",
        description="Feature flag to enable/disable agent (fallback to heuristics)",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
