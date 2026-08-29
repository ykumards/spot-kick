"""Codex CLI backend: `codex exec --output-schema`, using whatever login the CLI has. No SDK, no key."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from .llm import BrainError

URI_RE = re.compile(r"spotify:track:[A-Za-z0-9]{22}")


class Codex:
    name = "codex"

    def __init__(self, model: str = "gpt-5.6-terra", reasoning: str = "low", binary: str = "codex"):
        self.model, self.reasoning, self.binary = model, reasoning, binary

    def _cmd(self, *extra: str) -> list[str]:
        return [self.binary, *extra, "exec", "--skip-git-repo-check", "-s", "read-only", "-m", self.model,
                "-c", f'model_reasoning_effort="{self.reasoning}"']

    def complete_json(self, prompt: str, schema: dict, *, timeout: int = 240) -> dict:
        with tempfile.TemporaryDirectory() as d:
            sf, out = Path(d) / "schema.json", Path(d) / "out.json"
            sf.write_text(json.dumps(schema))
            cmd = self._cmd() + ["--output-schema", str(sf), "-o", str(out), prompt]
            try:
                r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False)
            except FileNotFoundError as e:
                raise BrainError("codex CLI not found; install it or switch llm_backend") from e
            except subprocess.TimeoutExpired as e:
                raise BrainError(f"codex timed out after {timeout}s") from e
            if r.returncode != 0 or not out.exists():
                raise BrainError(f"codex failed: {r.stderr.strip()[-400:]}")
            try:
                return json.loads(out.read_text())
            except json.JSONDecodeError as e:
                raise BrainError(f"codex returned non-JSON: {out.read_text()[:200]}") from e

    def search_uri(self, artist: str, title: str, *, timeout: int = 150) -> str | None:
        prompt = (f"Find the Spotify track URI for the studio recording of '{title}' by {artist}. Use web search on open.spotify.com "
                  "to confirm the 22-character id. Reply with spotify:track:<id> only.")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.txt"
            cmd = self._cmd("--search") + ["-o", str(out), prompt]
            try:
                subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None
            m = URI_RE.search(out.read_text() if out.exists() else "")
            return m.group(0) if m else None
