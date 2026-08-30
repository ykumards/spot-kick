"""Claude Code CLI backend: ``claude -p --json-schema``, using the CLI's own login.

``--json-schema`` makes the CLI validate the answer against the schema and return it as ``structured_output``;
``--tools ""`` gives it no tools. Every call runs from an empty directory so that no project settings or CLAUDE.md
reach the prompt.
"""
from __future__ import annotations

import json
import tempfile

from . import cli
from .llm import BrainError

DEFAULT_MODEL = "sonnet"
DEFAULT_BINARY = "claude"
# Structured output is an internal tool call, so a proposal takes two turns.
PROPOSE_MAX_TURNS = 4
NO_TOOLS = ""
# Claude Code thinks by default, and naming six songs made it think for ~11k tokens: two minutes per call against
# ten seconds without. The env var is honoured by every CLI version; the setting is the documented switch.
NO_THINKING_ENV = {"MAX_THINKING_TOKENS": "0"}
NO_THINKING_SETTINGS = '{"alwaysThinkingEnabled": false}'
OUTPUT_PREVIEW_CHARS = 200


class ClaudeCode:
    """Brain backend using the Claude Code CLI."""

    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, binary: str = DEFAULT_BINARY):
        self.model = model
        self.binary = binary

    def base_command(self) -> list[str]:
        """Return the ``claude -p`` argv: one JSON result, no saved session, no tools.

        ``--tools`` is variadic and would consume the prompt, so a boolean flag is placed last.
        """
        command = [self.binary, "-p", "--model", self.model, "--output-format", "json", "--tools", NO_TOOLS]
        command += ["--max-turns", str(PROPOSE_MAX_TURNS), "--settings", NO_THINKING_SETTINGS]
        command += ["--no-session-persistence"]
        return command

    def complete_json(self, prompt: str, schema: dict, *, timeout: int = 240) -> dict:
        command = self.base_command() + ["--json-schema", json.dumps(schema), prompt]
        with tempfile.TemporaryDirectory() as workdir:
            result = cli.run(command, timeout=timeout, tool="claude", cwd=workdir, extra_env=NO_THINKING_ENV)
        if result.returncode != 0:
            raise BrainError(f"claude failed: {cli.last_line(result.stderr or result.stdout)}")
        envelope = parse_envelope(result.stdout)
        if envelope.get("is_error"):
            errors = envelope.get("errors") or [envelope.get("result") or "unknown error"]
            raise BrainError(f"claude failed: {errors[0]}")
        structured = envelope.get("structured_output")
        if not isinstance(structured, dict):
            raise BrainError("claude returned no structured output")
        return structured



def parse_envelope(stdout: str) -> dict:
    """Parse the JSON object that ``--output-format json`` prints."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise BrainError(f"claude returned non-JSON: {stdout[:OUTPUT_PREVIEW_CHARS]}") from error
    if not isinstance(envelope, dict):
        raise BrainError("claude returned an unexpected shape")
    return envelope
