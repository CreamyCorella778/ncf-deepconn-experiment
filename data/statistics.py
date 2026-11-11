"""Print summary statistics for the processed Amazon and Yelp datasets."""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS = ("train", "val", "test")


def load_review_docs(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file review: {path}")

    with path.open("rb") as file:
        docs = pickle.load(file)
    if not isinstance(docs, dict):
        raise ValueError(f"Định dạng review không hợp lệ trong {path}; cần dict.")
    return docs


def summarize_dataset(dataset_dir: Path, name: str) -> dict:
    users = set()
    items = set()
    review_count = 0
    train_review_count = 0

    for split in SPLITS:
        path = dataset_dir / f"{split}_ratings.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file rating: {path}")

        ratings = pd.read_parquet(path, columns=["user_idx", "item_idx"])
        review_count += len(ratings)
        users.update(ratings["user_idx"].unique().tolist())
        items.update(ratings["item_idx"].unique().tolist())
        if split == "train":
            train_review_count = len(ratings)

    user_docs = load_review_docs(dataset_dir / "user_reviews.pkl")
    word_count = sum(len(str(text).split()) for text in user_docs.values())

    n_users = len(users)
    n_items = len(items)
    sparsity = (
        1 - review_count / (n_users * n_items)
        if n_users and n_items
        else 0.0
    )
    return {
        "dataset": name,
        "users": n_users,
        "items": n_items,
        "reviews": review_count,
        "words": word_count,
        "reviews_per_user": review_count / n_users if n_users else 0.0,
        "words_per_review": word_count / train_review_count
        if train_review_count
        else 0.0,
        "sparsity": sparsity,
    }


def print_table(stats: list[dict]) -> None:
    headers = (
        "Dataset",
        "#Users",
        "#Items",
        "#Reviews",
        "#Words*",
        "Reviews/user",
        "Words/review*",
        "Sparsity",
    )
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in stats:
        values = (
            row["dataset"],
            f'{row["users"]:,}',
            f'{row["items"]:,}',
            f'{row["reviews"]:,}',
            f'{row["words"]:,}',
            f'{row["reviews_per_user"]:.2f}',
            f'{row["words_per_review"]:.2f}',
            f'{row["sparsity"]:.4%}',
        )
        print("| " + " | ".join(values) + " |")
    print(
        "\n* #Words đếm từ user_reviews.pkl (review train đã làm sạch và giới hạn độ dài); "
        "Words/review = #Words / số review train."
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Thống kê các dataset Amazon và Yelp đã tiền xử lý."
    )
    parser.add_argument(
        "--dataset",
        choices=("amazon", "yelp", "both"),
        default="both",
        help="Dataset cần thống kê (mặc định: both).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROCESSED,
        help="Thư mục chứa các thư mục amazon/ và yelp/.",
    )
    args = parser.parse_args()

    names = ("amazon", "yelp") if args.dataset == "both" else (args.dataset,)
    stats = [summarize_dataset(args.data_root / name, name) for name in names]
    print_table(stats)


if __name__ == "__main__":
    main()
