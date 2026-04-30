"""Common helpers for lab3 scripts."""

from __future__ import annotations

from pathlib import Path


def project_lab_root(current_file: str | Path) -> Path:
    """Return the lab3 root directory for a script inside src/."""
    return Path(current_file).resolve().parent.parent


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
