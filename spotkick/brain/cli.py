"""Subprocess helpers shared by the CLI backends.

Both backends are command-line tools that hold their own login, so the app never handles an API key. A call
builds an argv, runs it with no stdin and a timeout, and reports failure as ``BrainError``.
"""
from __future__ import annotations

import os
import subprocess

from .llm import BrainError

STDERR_TAIL_CHARS = 400


def last_line(text: str) -> str:
    """Return the last non-empty line of ``text``, truncated; CLIs put the actual error there."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "no error output"
    return lines[-1][:STDERR_TAIL_CHARS]


def run(
    command: list[str], *, timeout: int, tool: str, cwd: str | None = None, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` to completion.

    Raises BrainError when the binary is missing or the call times out. A non-zero exit is returned for the caller
    to interpret, because its meaning differs between CLIs.
    """
    environment = None
    if extra_env:
        environment = {**os.environ, **extra_env}
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
            env=environment,
        )
    except FileNotFoundError as error:
        raise BrainError(f"{tool} CLI not found; install it and log in first") from error
    except subprocess.TimeoutExpired as error:
        raise BrainError(f"{tool} timed out after {timeout}s") from error

