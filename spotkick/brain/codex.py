"""Codex CLI backend: `codex exec --output-schema`, using whatever login the CLI has. No SDK, no key."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import cli
from .llm import BrainError

DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING = "low"
DEFAULT_BINARY = "codex"
SCHEMA_FILENAME = "schema.json"
JSON_OUTPUT_FILENAME = "out.json"
OUTPUT_PREVIEW_CHARS = 200


class Codex:
    """The default brain: OpenAI's models through the Codex CLI."""

    name = "codex"

    def __init__(self, model: str = DEFAULT_MODEL, reasoning: str = DEFAULT_REASONING, binary: str = DEFAULT_BINARY):
        self.model = model
        self.reasoning = reasoning
        self.binary = binary

    def base_command(self) -> list[str]:
        """The `codex exec` invocation: sandboxed read-only, our model, our reasoning effort."""
        return [
            self.binary,
            "exec",
            "--skip-git-repo-check",
            "-s", "read-only",
            "-m", self.model,
            "-c", f'model_reasoning_effort="{self.reasoning}"',
        ]

    def complete_json(self, prompt: str, schema: dict, *, timeout: int = 240) -> dict:
        with tempfile.TemporaryDirectory() as workdir:
            schema_path = Path(workdir) / SCHEMA_FILENAME
            output_path = Path(workdir) / JSON_OUTPUT_FILENAME
            schema_path.write_text(json.dumps(schema))
            command = self.base_command() + ["--output-schema", str(schema_path), "-o", str(output_path), prompt]
            result = cli.run(command, timeout=timeout, tool="codex")
            if result.returncode != 0 or not output_path.exists():
                raise BrainError(f"codex failed: {cli.last_line(result.stderr)}")
            try:
                return json.loads(output_path.read_text())
            except json.JSONDecodeError as error:
                output_preview = output_path.read_text()[:OUTPUT_PREVIEW_CHARS]
                raise BrainError(f"codex returned non-JSON: {output_preview}") from error

