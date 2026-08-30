"""The Spotify client secret in the macOS Keychain, through the ``security`` tool.

One generic-password item under service ``spotkick`` holds the secret. Reading it back needs no prompt for the
user who stored it.
"""
from __future__ import annotations

import subprocess

SERVICE = "spotkick"
ACCOUNT = "spotify-client-secret"
NOT_FOUND_EXIT_CODE = 44
TIMEOUT_S = 10


class KeychainError(RuntimeError):
    pass


def run_security(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["security", *arguments], capture_output=True, text=True, timeout=TIMEOUT_S, check=False)


def get_secret(account: str = ACCOUNT, *, service: str = SERVICE) -> str | None:
    """Return the stored secret, or None."""
    completed = run_security("find-generic-password", "-a", account, "-s", service, "-w")
    if completed.returncode == NOT_FOUND_EXIT_CODE:
        return None
    if completed.returncode != 0:
        raise KeychainError(completed.stderr.strip() or "keychain read failed")
    return completed.stdout.strip()


def set_secret(secret: str, account: str = ACCOUNT, *, service: str = SERVICE) -> None:
    """Store or replace the secret."""
    completed = run_security("add-generic-password", "-U", "-a", account, "-s", service, "-w", secret)
    if completed.returncode != 0:
        raise KeychainError(completed.stderr.strip() or "keychain write failed")


def delete_secret(account: str = ACCOUNT, *, service: str = SERVICE) -> None:
    completed = run_security("delete-generic-password", "-a", account, "-s", service)
    if completed.returncode not in (0, NOT_FOUND_EXIT_CODE):
        raise KeychainError(completed.stderr.strip() or "keychain delete failed")
