"""Configuration and JSON helpers for the H local-refinement experiment."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "ultralytics-main"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "data-v2-abl-d1h-localrefine-rb128-s42.yaml"
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
EXPECTED_REFINER_PARAMETERS = 1_224_515


def resolve_existing_path(value: str | Path) -> Path:
    """Resolve a path from the current directory or repository root."""
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"file not found: {value}; checked: {searched}")


def load_runtime(cli_args: Any) -> dict[str, Any]:
    """Load, validate, and resolve the H experiment configuration."""
    config_path = resolve_existing_path(cli_args.config)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("experiment YAML must contain a mapping")
    for key in ("experiment", "model", "data", "refiner", "train"):
        if key not in config:
            raise ValueError(f"experiment YAML is missing {key!r}")
    if not isinstance(config["refiner"], dict) or not isinstance(config["train"], dict):
        raise ValueError("refiner and train must be mappings")

    experiment = str(config["experiment"])
    requested_name = cli_args.run_name
    if cli_args.preflight1 and requested_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        requested_name = f"{experiment}-preflight1-{timestamp}"
    run_name = requested_name or experiment
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("run-name may contain only letters, numbers, dots, underscores, and hyphens")

    model_path = resolve_existing_path(cli_args.model or config["model"])
    data_path = resolve_existing_path(cli_args.data or config["data"])
    train = dict(config["train"])
    refiner = dict(config["refiner"])
    if cli_args.preflight1:
        train["epochs"] = 1
    project = Path(str(train.get("project", "runs"))).expanduser()
    if not project.is_absolute():
        project = REPO_ROOT / project
    run_dir = project.resolve() / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")

    return {
        "config_path": config_path,
        "experiment": experiment,
        "run_name": run_name,
        "run_dir": run_dir,
        "model": model_path,
        "data": data_path,
        "refiner": refiner,
        "train": train,
        "preflight": bool(cli_args.preflight1),
    }


def json_value(value: Any) -> Any:
    """Convert metric containers to JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    """Write an inspectable UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_value(value), file, ensure_ascii=False, indent=2)
