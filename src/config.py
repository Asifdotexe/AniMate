"""
Configuration file parser for AniMate.
"""

from pathlib import Path

import yaml
from box import Box

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config.yaml") -> Box:
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")

    with open(full_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None or not isinstance(config_dict, dict):
        raise ValueError(f"Config file is empty or invalid (not a mapping): {full_path}")

    _resolve_paths(config_dict, PROJECT_ROOT)
    return Box(config_dict)


def _resolve_paths(config: dict, root: Path):
    if "paths" in config:
        for key, value in config["paths"].items():
            config["paths"][key] = str(root / value)


config = load_config()
