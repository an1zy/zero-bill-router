import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, Field


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    host: str = "127.0.0.1"
    port: int = 8787


class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen2.5-coder:7b"
    enabled: bool = True


class RoutingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    local_max_chars: int = 1200
    local_keywords: list[str] = Field(
        default_factory=lambda: [
            "commit message",
            "summarize",
            "rename",
            "format",
            "typo",
            "docstring",
        ]
    )


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    base_url: str
    api_key_env: str
    model: str
    priority: int = 10
    cooldown_seconds: int = 60
    cost_per_1m_tokens: float = 0.0
    api_key: str | None = None
    active: bool = True
    is_active: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    server: ServerConfig = Field(default_factory=ServerConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    providers: list[ProviderConfig] = Field(default_factory=list)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    config = AppConfig.model_validate(data)

    for provider in config.providers:
        env_val = os.getenv(provider.api_key_env, "").strip()
        if env_val:
            provider.api_key = env_val
            if not provider.active or not provider.is_active:
                provider.active = False
                provider.is_active = False
            else:
                provider.active = True
                provider.is_active = True
        else:
            provider.api_key = None
            provider.active = False
            provider.is_active = False

    return config
