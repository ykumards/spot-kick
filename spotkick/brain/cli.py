"""Running a coding-agent CLI as a subprocess, the way every backend does it.

Both brains are command-line tools that already hold a login, so the app never touches a key: it builds an argv,
runs it with no stdin and a timeout, and turns the three ways that can fail into one `BrainError`.
"""
from __future__ import annotations

import os
import subprocess

from .llm import BrainError

STDERR_TAIL_CHARS = 400


def last_line(text: str) -> str:
    """The last non-empty line of stderr — where a CLI puts the actual error — cut to a readable length."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "no error output"
    return lines[-1][:STDERR_TAIL_CHARS]


def run(
    command: list[str], *, timeout: int, tool: str, cwd: str | None = None, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `command` to completion. Raises BrainError when the binary is missing or the call times out; a non-zero
    exit is returned for the caller to interpret, because the CLIs differ in what that means."""
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

