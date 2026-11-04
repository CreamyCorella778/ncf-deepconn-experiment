"""Các độ đo đánh giá."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch


@dataclass
class SeenItemsIndex:
    offsets: np.ndarray
    item_ids: np.ndarray
    num_items: int

    def for_user(self, user_id: int) -> np.ndarray:
        start, end = self.offsets[user_id : user_id + 2]
        return self.item_ids[start:end]


def build_seen_items_index(
    interactions: Sequence[pd.DataFrame], num_users: int, num_items: int
) -> SeenItemsIndex:
    """Build a compact CSR-style index of each user's known positive items."""
    users = np.concatenate(
        [df["user_idx"].to_numpy(dtype=np.int32, copy=False) for df in interactions]
    )
    items = np.concatenate(
        [df["item_idx"].to_numpy(dtype=np.int32, copy=False) for df in interactions]
    )
    if (
        np.any(users < 0)
        or np.any(users >= num_users)
        or np.any(items < 0)
        or np.any(items >= num_items)
    ):
        raise ValueError("Interaction contains a user/item ID outside the model range.")

    counts = np.bincount(users, minlength=num_users)
    offsets = np.empty(num_users + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(users, kind="stable")
    sorted_items = items[order]
    del users, items, counts, order
    return SeenItemsIndex(offsets, sorted_items, num_items)


def _sample_negatives(
    rng: np.random.Generator,
    seen_items: np.ndarray,
    num_items: int,
    count: int,
    user_id: int,
) -> np.ndarray:
    seen_items = np.unique(seen_items)
    available = num_items - len(seen_items)
    if available < count:
        raise ValueError(
            f"User {user_id} has only {available} unseen items; "
            f"cannot sample {count} negatives."
        )

    negatives = np.empty(count, dtype=np.int64)
    filled = 0
    while filled < count:
        draws = rng.integers(
            0, num_items, size=max(16, 2 * (count - filled)), dtype=np.int64
        )
        candidates = draws[~np.isin(draws, seen_items)]
        if filled:
            candidates = candidates[~np.isin(candidates, negatives[:filled])]
        _, first_occurrences = np.unique(candidates, return_index=True)
        candidates = candidates[np.sort(first_occurrences)]
        take = min(len(candidates), count - filled)
        negatives[filled : filled + take] = candidates[:take]
        filled += take
    return negatives


@torch.no_grad()
def evaluate_sampled_ranking(
    model,
    positives: pd.DataFrame,
    seen_index: SeenItemsIndex,
    device,
    k: int = 10,
    negatives_per_positive: int = 99,
    seed: int = 42,
    batch_users: int = 256,
    user_encoded: np.ndarray = None,
    item_encoded: np.ndarray = None,
) -> dict:
    """Evaluate HR@K/NDCG@K; ties break by ascending item ID."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1.")
    if batch_users < 1:
        raise ValueError("batch_users must be at least 1.")
    if (user_encoded is None) != (item_encoded is None):
        raise ValueError("Both encoded review arrays must be supplied together.")
    if positives.empty:
        raise ValueError("Cannot calculate ranking metrics for an empty split.")

    users = positives["user_idx"].to_numpy(dtype=np.int64, copy=False)
    positive_items = positives["item_idx"].to_numpy(dtype=np.int64, copy=False)
    if (
        np.any(users < 0)
        or np.any(users >= len(seen_index.offsets) - 1)
        or np.any(positive_items < 0)
        or np.any(positive_items >= seen_index.num_items)
    ):
        raise ValueError("Evaluation positives contain an ID outside the model range.")

    model.eval()
    rng = np.random.default_rng(seed)
    candidate_count = negatives_per_positive + 1
    hits = 0
    ndcg = 0.0
    n = len(users)

    for start in range(0, n, batch_users):
        batch_end = min(start + batch_users, n)
        batch_users_ids = users[start:batch_end]
        batch_positive_items = positive_items[start:batch_end]
        candidates = np.empty((len(batch_users_ids), candidate_count), dtype=np.int64)
        candidates[:, 0] = batch_positive_items
        for row, user_id in enumerate(batch_users_ids):
            candidates[row, 1:] = _sample_negatives(
                rng,
                seen_index.for_user(int(user_id)),
                seen_index.num_items,
                negatives_per_positive,
                int(user_id),
            )

        if user_encoded is None:
            expanded_users = np.repeat(batch_users_ids, candidate_count)
            user_tensor = torch.as_tensor(expanded_users, dtype=torch.long, device=device)
            item_tensor = torch.as_tensor(
                candidates.reshape(-1), dtype=torch.long, device=device
            )
            scores = model(user_tensor, item_tensor)
        else:
            user_reviews = torch.as_tensor(
                user_encoded[batch_users_ids].astype(np.int64),
                dtype=torch.long,
                device=device,
            )
            item_reviews = torch.as_tensor(
                item_encoded[candidates.reshape(-1)].astype(np.int64),
                dtype=torch.long,
                device=device,
            ).reshape(len(batch_users_ids), candidate_count, -1)
            scores = model.score_candidates(user_reviews, item_reviews)

        scores = scores.reshape(len(batch_users_ids), candidate_count)
        negative_items = torch.as_tensor(
            candidates[:, 1:], dtype=torch.long, device=device
        )
        positive_items_tensor = torch.tensor(
            batch_positive_items, dtype=torch.long, device=device
        ).unsqueeze(1)
        outranks_positive = (scores[:, 1:] > scores[:, :1]) | (
            (scores[:, 1:] == scores[:, :1])
            & (negative_items < positive_items_tensor)
        )
        ranks = 1 + torch.sum(outranks_positive, dim=1)
        hits += int(torch.sum(ranks <= k).item())
        in_top_k = ranks <= k
        ndcg += torch.sum(
            torch.where(
                in_top_k,
                1.0 / torch.log2(ranks.to(torch.float32) + 1.0),
                0.0,
            )
        ).item()

    return {
        f"hr@{k}": hits / n,
        f"ndcg@{k}": ndcg / n,
        "users": n,
        "negatives_per_positive": negatives_per_positive,
    }


@torch.no_grad()
def evaluate_rmse_mae(model, loader, device, rating_min=1.0, rating_max=5.0):
    model.eval()
    sq_err, abs_err, n = 0.0, 0.0, 0
    for a, b, y in loader:
        a, b, y = a.to(device), b.to(device), y.to(device)
        pred = torch.clamp(model(a, b), rating_min, rating_max)
        diff = pred - y
        sq_err += torch.sum(diff**2).item()
        abs_err += torch.sum(torch.abs(diff)).item()
        n += y.size(0)
    return (sq_err / n) ** 0.5, abs_err / n
