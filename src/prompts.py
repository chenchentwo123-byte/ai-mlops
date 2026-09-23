"""Normalize Grounding DINO text prompts.

Grounding DINO expects lowercase noun phrases joined by periods:
``"cat. dog. person."``. Users often paste ``cat.dog.person`` or
comma-separated lists; both are accepted.
"""

from __future__ import annotations

import re
from typing import Iterable


# Comma / semicolon / newline / pipe / period all separate classes.
# Periods are the official Grounding DINO delimiter.
_SPLIT_RE = re.compile(r"[,;\n|]+|\.(?:\s+|$)|\.(?=[A-Za-z0-9])")


def parse_classes(prompt: str) -> list[str]:
    """Split a user prompt into unique class names (order preserved)."""
    if not prompt:
        return []
    seen: set[str] = set()
    classes: list[str] = []
    for raw in _SPLIT_RE.split(prompt):
        name = raw.strip().lower().strip(".")
        name = re.sub(r"\s+", " ", name)
        if not name or name in seen:
            continue
        seen.add(name)
        classes.append(name)
    return classes


def to_caption(classes: Iterable[str]) -> str:
    """Join classes into the caption Grounding DINO expects."""
    names = [c.strip().lower().strip(".") for c in classes if c and str(c).strip()]
    if not names:
        return ""
    return ". ".join(names) + "."


def normalize_prompt(prompt: str) -> tuple[str, list[str]]:
    """Return ``(caption, classes)`` from free-form user text."""
    classes = parse_classes(prompt)
    return to_caption(classes), classes
