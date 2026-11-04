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


def get_num_workers(cfg_val: int = 0) -> int:
    """Trên Windows luôn đặt 0 để tránh lỗi chia sẻ bộ nhớ."""
    if os.name == "nt":
        return 0
    return cfg_val
