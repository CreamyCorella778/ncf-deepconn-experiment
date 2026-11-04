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
        user_docs: dict,
        item_docs: dict,
        vocab: dict,
        max_len: int = 200,
    ):
        self.users = df["user_idx"].values
        self.items = df["item_idx"].values
        self.ratings = df["rating"].values.astype(np.float32)
        self.max_len = max_len
        self.vocab = vocab
        self.unk_id = vocab.get("<unk>", 0)
        self.pad_id = vocab.get("<pad>", 1)

        # Mã hóa sẵn toàn bộ document
        self.user_encoded = self._encode_docs(user_docs)
        self.item_encoded = self._encode_docs(item_docs)

    def _encode_docs(self, docs: dict) -> dict:
        encoded = {}
        for idx, text in docs.items():
            tokens = str(text).lower().split()[: self.max_len]
            ids = [self.vocab.get(t, self.unk_id) for t in tokens]
            if len(ids) < self.max_len:
                ids += [self.pad_id] * (self.max_len - len(ids))
            encoded[idx] = np.asarray(ids, dtype=np.int32)
        return encoded

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, i):
        u = int(self.users[i])
        it = int(self.items[i])
        u_ids = self.user_encoded.get(
            u, np.full(self.max_len, self.pad_id, dtype=np.int32)
        )
        i_ids = self.item_encoded.get(
            it, np.full(self.max_len, self.pad_id, dtype=np.int32)
        )
        return (
            torch.from_numpy(u_ids),
            torch.from_numpy(i_ids),
            torch.tensor(self.ratings[i], dtype=torch.float32),
        )
