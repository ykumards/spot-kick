"""Where things live and what the knobs are. ~/.spotkick/config.toml overrides the defaults."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path(os.environ.get("SPOTKICK_HOME", Path.home() / ".spotkick"))


@dataclass
class Config:
    home: Path = HOME
    llm_backend: str = "codex"          # codex | openai | local
    llm_model: str = "gpt-5.6-terra"
    llm_reasoning: str = "low"
    local_base_url: str = "http://127.0.0.1:8080/v1"
    dig: int = 1                        # 0 any · 1 hits off · 2 deep
    n_candidates: int = 6
    alpha: float = 0.7                  # listener-state EWMA
    knobs: dict = field(default_factory=lambda: {"home_pull": 1.0, "wander": 0.15})

    @property
    def db_path(self) -> Path:
        return self.home / "spotkick.db"


def load(path: Path | None = None) -> Config:
    cfg = Config()
    path = path or cfg.home / "config.toml"
    if path.exists():
        data = tomllib.loads(path.read_text())
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    cfg.home = Path(cfg.home)
    return cfg
