"""Tiện ích chung: seed, device, logging, đọc config."""

import os
import random
import yaml
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_dir(data_config: dict, project_root: Path) -> Path:
    """Resolve the data directory from the project root and check its dataset label."""
    data_dir = Path(data_config["dir"])
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    data_dir = data_dir.resolve()

    dataset = data_config.get("dataset")
    if dataset is not None:
        if dataset not in {"amazon", "yelp"}:
            raise ValueError(f"Dataset không hỗ trợ trong config: {dataset}")
        if data_dir.name.casefold() != dataset.casefold():
            raise ValueError(
                f"Dataset config là '{dataset}' nhưng data.dir trỏ tới "
                f"'{data_dir}'. Hãy dùng thư mục processed/{dataset}."
            )
    return data_dir


def get_num_workers(cfg_val: int = 0) -> int:
    """Trên Windows luôn đặt 0 để tránh lỗi chia sẻ bộ nhớ."""
    if os.name == "nt":
        return 0
    return cfg_val
