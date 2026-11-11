"""Export a fixed set of unseen test items for each test user."""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.metrics import build_seen_items_index, sample_test_negatives


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "amazon"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "test_negatives.pkl"


def main():
    parser = argparse.ArgumentParser(
        description="Precompute negative candidates for sampled test ranking."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    interactions = [
        pd.read_parquet(data_dir / filename)
        for filename in (
            "train_ratings.parquet",
            "val_ratings.parquet",
            "test_ratings.parquet",
        )
    ]
    test_df = interactions[2]
    if test_df.empty:
        raise ValueError(f"Test split is empty: {data_dir / 'test_ratings.parquet'}")

    all_users = pd.concat(
        [df["user_idx"] for df in interactions], ignore_index=True
    ).to_numpy()
    all_items = pd.concat(
        [df["item_idx"] for df in interactions], ignore_index=True
    ).to_numpy()
    if (all_users < 0).any() or (all_items < 0).any():
        raise ValueError("Interaction IDs must be non-negative.")
    num_users = int(all_users.max()) + 1
    num_items = int(all_items.max()) + 1

    seen_index = build_seen_items_index(interactions, num_users, num_items)
    negatives = sample_test_negatives(
        test_df[["user_idx", "item_idx"]],
        seen_index,
        count=99,
        seed=args.seed,
    )
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(negatives, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Exported negatives for {len(negatives):,} test users to {output}")


if __name__ == "__main__":
    main()
