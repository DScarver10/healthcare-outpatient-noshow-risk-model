"""Small shared helpers used across notebooks and modules."""

from __future__ import annotations

import json
from pathlib import Path


def print_section(title: str) -> None:
    """Print a lightweight notebook section header."""
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def save_json(payload: dict, path: Path) -> None:
    """Save metadata or parameters in a human-readable JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
