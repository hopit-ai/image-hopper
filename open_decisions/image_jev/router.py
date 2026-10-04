"""Deterministic text-only router registered as ``router-rules/v1``.

Only the question and visible option text are inspected.  Source metadata,
board categories, images, and conversation state are intentionally absent
from this API.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


RULES_VERSION = "router-rules/v1"
SCREEN_GEOMETRY = "screen_geometry"
DEFAULT = "default"
ROUTES = (SCREEN_GEOMETRY, DEFAULT)

GUI_TARGET_CUES = (
    "marker", "click", "tap", "button", "icon", "menu", "toolbar", "tab",
    "window", "dialog", "element", "on screen", "screenshot", "cursor", "link",
    "field",
)
GEOMETRY_CUES = (
    "angle", "triangle", "circle", "radius", "diameter", "perimeter", "area of",
    "length of", "parallel", "perpendicular", "polygon", "quadrilateral", "chord",
    "tangent", "arc", "degrees", "°", "find x", "find y", r"\triangle",
)
LITERAL_CUES = ("°", r"\triangle")

RULE_DEFINITION = {
    "version": RULES_VERSION,
    "input": "question+option-text-only",
    "case_sensitive": False,
    "matching": {
        "word_and_phrase_cues": "unicode-word-boundaries",
        "literal_cues": list(LITERAL_CUES),
    },
    "screen_geometry": {
        "gui_target_cues": list(GUI_TARGET_CUES),
        "geometry_cues": list(GEOMETRY_CUES),
    },
    "fallback": DEFAULT,
    "ambiguous": DEFAULT,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


RULES_SHA256 = hashlib.sha256(_canonical_bytes(RULE_DEFINITION)).hexdigest()


def _pattern(cue: str) -> re.Pattern[str]:
    # Every non-literal registered cue consists of words separated by spaces.
    # Lookarounds make phrases boundary-aware at both ends without ASCII-only
    # ``\b`` behaviour ("tablet" must not match "tab", for example).
    return re.compile(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", re.IGNORECASE)


_WORD_PATTERNS = tuple(
    _pattern(cue)
    for cue in (*GUI_TARGET_CUES, *GEOMETRY_CUES)
    if cue not in LITERAL_CUES
)


def _option_text(option: Any) -> str | None:
    if isinstance(option, str):
        return option
    if isinstance(option, Mapping):
        # These are the two visible fields accepted by the benchmark and
        # /v1/systemone contracts.  Metadata fields are deliberately ignored.
        values = []
        for key in ("name", "description", "text"):
            value = option.get(key)
            if value is not None:
                if not isinstance(value, str):
                    return None
                values.append(value)
        return " ".join(values) if values else None
    return None


def route(question: str, options: Sequence[Any]) -> str:
    """Return the registered route, failing ambiguous/malformed inputs closed.

    Matching is case-insensitive.  Word cues and multi-word phrases require
    word boundaries; ``°`` and ``\\triangle`` are matched literally.
    """
    if not isinstance(question, str) or isinstance(options, (str, bytes)):
        return DEFAULT
    try:
        option_values = list(options)
    except TypeError:
        return DEFAULT
    visible = []
    for option in option_values:
        value = _option_text(option)
        if value is None:
            return DEFAULT
        visible.append(value)
    text = "\n".join((question, *visible)).casefold()
    if any(cue.casefold() in text for cue in LITERAL_CUES):
        return SCREEN_GEOMETRY
    if any(pattern.search(text) for pattern in _WORD_PATTERNS):
        return SCREEN_GEOMETRY
    return DEFAULT


__all__ = [
    "DEFAULT", "GEOMETRY_CUES", "GUI_TARGET_CUES", "LITERAL_CUES", "ROUTES",
    "RULES_SHA256", "RULES_VERSION", "RULE_DEFINITION", "SCREEN_GEOMETRY", "route",
]
