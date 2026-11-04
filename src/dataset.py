"""Dataset cho NCF và DeepCoNN."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users = torch.from_numpy(df["user_idx"].values)
        self.items = torch.from_numpy(df["item_idx"].values)
        self.ratings = torch.from_numpy(df["rating"].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class DeepCoNNDataset(Dataset):
    """Mỗi mẫu: (user_ids[max_len], item_ids[max_len], rating)."""

    def __init__(
        self,
        df: pd.DataFrame,
        user_encoded: np.ndarray,
        item_encoded: np.ndarray,
        max_len: int = 200,
    ):
        self.users = df["user_idx"].values
        self.items = df["item_idx"].values
        self.ratings = df["rating"].values.astype(np.float32)
        self.max_len = max_len
        self.user_encoded = user_encoded
        self.item_encoded = item_encoded

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, i):
        u = int(self.users[i])
        it = int(self.items[i])
        return (
            torch.from_numpy(self.user_encoded[u].astype(np.int64)),
            torch.from_numpy(self.item_encoded[it].astype(np.int64)),
            torch.tensor(self.ratings[i], dtype=torch.float32),
        )


def encode_review_docs(
    docs: dict, vocab: dict, num_docs: int, max_len: int
) -> np.ndarray:
    """Encode review documents into a compact, ID-indexed array."""
    pad_id = vocab["<pad>"]
    unk_id = vocab["<unk>"]
    max_token_id = max(vocab.values())
    dtype = np.uint16 if max_token_id <= np.iinfo(np.uint16).max else np.uint32
    encoded = np.full((num_docs, max_len), pad_id, dtype=dtype)

    for doc_id, text in docs.items():
        doc_id = int(doc_id)
        if not 0 <= doc_id < num_docs:
            raise ValueError(f"Review ID ngoài phạm vi: {doc_id}")
        tokens = str(text).lower().split()[:max_len]
        encoded[doc_id, : len(tokens)] = [
            vocab.get(token, unk_id) for token in tokens
        ]
    return encoded
