"""Configuration: defaults overridden by ~/.spotkick/config.toml."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

HOME = Path(os.environ.get("SPOTKICK_HOME", Path.home() / ".spotkick"))
CONFIG_FILENAME = "config.toml"
DB_FILENAME = "spotkick.db"


@dataclass
class Config:
    home: Path = HOME
    llm_backend: str = "codex"          # codex · claude — which logged-in CLI names the songs
    llm_model: str = "gpt-5.6-terra"    # Codex's model
    llm_reasoning: str = "low"          # Codex's reasoning effort
    claude_model: str = "sonnet"        # Claude Code's model, when llm_backend is claude
    spotify_client_id: str = ""         # the developer's Spotify app: resolves names to track ids
                                        # (its secret lives in the Keychain, or SPOTKICK_SPOTIFY_CLIENT_SECRET)
    lean: str = ""                      # free text every pick stays inside (a mood, a language, an era); empty = none
    minime: bool = True                 # let Mini-Me (the taste model) choose among bench picks near the target
    n_candidates: int = 6
    alpha: float = 0.7                  # listener-state EWMA

    @property
    def db_path(self) -> Path:
        return self.home / DB_FILENAME


def load(path: Path | None = None) -> Config:
    """Load the configuration, applying any keys set in the TOML file. Unknown keys are ignored."""
    config = Config()
    config_path = path or config.home / CONFIG_FILENAME
    if config_path.exists():
        overrides = tomllib.loads(config_path.read_text())
        for name, value in overrides.items():
            if hasattr(config, name):
                setattr(config, name, value)
    # A TOML override arrives as a string; every consumer expects a Path.
    config.home = Path(config.home)
    return config


def save_setting(name: str, value: str | float | bool, path: Path | None = None) -> None:
    """Write one top-level key to the TOML file, leaving the rest of the file untouched.

    The stdlib reads TOML but does not write it, and re-serialising would drop comments, so the single
    ``name = value`` line is replaced or appended.
    """
    config_path = path or HOME / CONFIG_FILENAME
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = config_path.read_text().splitlines() if config_path.exists() else []
    new_line = f"{name} = {toml_literal(value)}"
    replaced = False
    kept_lines = []
    for line in existing_lines:
        key = line.split("=", 1)[0].strip()
        if key == name and not replaced:
            kept_lines.append(new_line)
            replaced = True
        else:
            kept_lines.append(line)
    if not replaced:
        kept_lines.append(new_line)
    config_path.write_text("\n".join(kept_lines) + "\n")


def toml_literal(value: str | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return str(value)
