"""Read .env into the environment.

Keys (Vast.ai) and machine-specific URLs live in .env, which is gitignored; .env.example
documents the shape. Real environment variables always win, so the same code runs unchanged
on the rented GPU, where the values come from the container instead of a file.
"""

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Set every KEY=value from `path` that is not already in the environment."""
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if not key or not value:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded
