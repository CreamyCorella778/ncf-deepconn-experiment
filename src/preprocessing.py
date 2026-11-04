"""Đọc dữ liệu đã làm sạch và suy ra số user/item, xây vocab cho DeepCoNN."""

import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


def load_ratings(data_dir: str, filename: str) -> pd.DataFrame:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["user_idx"] = df["user_idx"].astype(np.int64)
    df["item_idx"] = df["item_idx"].astype(np.int64)
    df["rating"] = df["rating"].astype(np.float32)
    return df


def load_pickle_dict(data_dir: str, filename: str) -> dict:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_num_ids(train_df, val_df, test_df, user_dict=None, item_dict=None):
    n_users = (
        int(max(train_df.user_idx.max(), val_df.user_idx.max(), test_df.user_idx.max()))
        + 1
    )
    n_items = (
        int(max(train_df.item_idx.max(), val_df.item_idx.max(), test_df.item_idx.max()))
        + 1
    )
    if user_dict:
        n_users = max(n_users, max(user_dict.keys()) + 1)
    if item_dict:
        n_items = max(n_items, max(item_dict.keys()) + 1)
    return n_users, n_items


def build_vocab(
    user_docs: dict, item_docs: dict, min_freq: int = 2, max_size: int = 50000
) -> dict:
    """Vocab đơn giản: 0=<pad>, 1=<unk>, còn lại theo tần suất."""
    counter = Counter()
    for docs in (user_docs, item_docs):
        for text in docs.values():
            counter.update(str(text).lower().split())
    most_common = [w for w, c in counter.most_common(max_size) if c >= min_freq]
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, w in enumerate(most_common, start=2):
        vocab[w] = i
    return vocab
