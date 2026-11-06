"""Load machine-specific settings (paths) from a YAML file kept out of git.

Convention: each project/examples folder has a tracked ``config.example.yaml``
and a gitignored ``config.yaml`` with real paths for the current machine.
"""
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    """Return the YAML at ``path`` as a dict; clear error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.resolve()} not found. Copy config.example.yaml to config.yaml "
            "in the same folder and fill in paths for your machine.")
    with open(path) as f:
        return yaml.safe_load(f) or {}
