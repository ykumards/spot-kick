"""One spelling for a song name, so the same song is the same key wherever it is looked up."""
from __future__ import annotations

import re

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(text: str) -> str:
    """Lower-case words with every run of punctuation or space collapsed to one space."""
    return NON_ALNUM_RE.sub(" ", text.lower()).strip()
