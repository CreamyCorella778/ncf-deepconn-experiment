"""
Entry point: huấn luyện & đánh giá NCF hoặc DeepCoNN dựa trên file config.
Cách dùng:
    python run_compare.py --config configs/ncf.yaml
    python run_compare.py --config configs/deepconn.yaml
    python run_compare.py --config configs/ncf.yaml --config configs/deepconn.yaml   # chạy cả 2
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import set_seed, get_device, load_config, get_num_workers
from src.preprocessing import (
    load_ratings,
    infer_num_ids,
    build_vocab,
    load_review_pickle,
)
from src.dataset import NCFDataset, DeepCoNNDataset, encode_review_docs
from src.metrics import (
    build_seen_items_index,
    evaluate_rmse_mae,
    evaluate_sampled_ranking,
)
from src.train import fit
from src.evaluate import evaluate_test
from models import NCFModel, DeepCoNNModel


def build_ncf(cfg, device):
    d = cfg["data"]
    train_df = load_ratings(d["dir"], d["train_file"])
    val_df = load_ratings(d["dir"], d["val_file"])
    test_df = load_ratings(d["dir"], d["test_file"])
    n_users, n_items = infer_num_ids(train_df, val_df, test_df)
    print(f"[NCF] n_users={n_users:,}, n_items={n_items:,}")
    seen_index = build_seen_items_index(
        [train_df, val_df, test_df], n_users, n_items
    )

    nw = get_num_workers(cfg["training"]["num_workers"])
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        NCFDataset(train_df),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
    )
    val_loader = DataLoader(
        NCFDataset(val_df),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )
    test_loader = DataLoader(
        NCFDataset(test_df),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    mp = cfg["model_params"]
    model = NCFModel(
        n_users,
        n_items,
        gmf_dim=mp["gmf_dim"],
        mlp_dims=tuple(mp["mlp_dims"]),
        dropout=mp["dropout"],
    ).to(device)
    ranking_data = {
        "positives": test_df[["user_idx", "item_idx"]],
        "seen_index": seen_index,
    }
    return model, train_loader, val_loader, test_loader, ranking_data


def build_deepconn(cfg, device):
    d = cfg["data"]
    train_df = load_ratings(d["dir"], d["train_file"])
    val_df = load_ratings(d["dir"], d["val_file"])
    test_df = load_ratings(d["dir"], d["test_file"])

    user_docs = load_review_pickle(str(Path(d["dir"]) / d["user_reviews"]))
    item_docs = load_review_pickle(str(Path(d["dir"]) / d["item_reviews"]))

    mp = cfg["model_params"]
    vocab = build_vocab(
        user_docs, item_docs, min_freq=mp["min_freq"], max_size=mp["max_vocab"]
    )
    print(f"[DeepCoNN] Vocab size = {len(vocab):,}")
    n_users, n_items = infer_num_ids(train_df, val_df, test_df)
    user_encoded = encode_review_docs(user_docs, vocab, n_users, mp["max_len"])
    item_encoded = encode_review_docs(item_docs, vocab, n_items, mp["max_len"])
    del user_docs, item_docs
    print(
        f"[DeepCoNN] Encoded reviews: {n_users:,} users, {n_items:,} items "
        f"({(user_encoded.nbytes + item_encoded.nbytes) / 1024**3:.2f} GiB)"
    )
    seen_index = build_seen_items_index(
        [train_df, val_df, test_df], n_users, n_items
    )

    nw = get_num_workers(cfg["training"]["num_workers"])
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        DeepCoNNDataset(train_df, user_encoded, item_encoded, mp["max_len"]),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DeepCoNNDataset(val_df, user_encoded, item_encoded, mp["max_len"]),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )
    test_loader = DataLoader(
        DeepCoNNDataset(test_df, user_encoded, item_encoded, mp["max_len"]),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    model = DeepCoNNModel(
        vocab_size=len(vocab),
        embed_dim=mp["embed_dim"],
        out_channels=mp["out_channels"],
        kernel_size=mp["kernel_size"],
        latent_dim=mp["latent_dim"],
        dropout=mp["dropout"],
    ).to(device)
    ranking_data = {
        "positives": test_df[["user_idx", "item_idx"]],
        "seen_index": seen_index,
        "user_encoded": user_encoded,
        "item_encoded": item_encoded,
    }
    return model, train_loader, val_loader, test_loader, ranking_data


def run_one(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))
    print(
        f"\n{'='*60}\n>>> Cấu hình: {cfg_path} | Model: {cfg['model']} | Device: {device}\n{'='*60}"
    )

    if cfg["model"] == "ncf":
        model, train_loader, val_loader, test_loader, ranking_data = build_ncf(
            cfg, device
        )
    elif cfg["model"] == "deepconn":
        model, train_loader, val_loader, test_loader, ranking_data = build_deepconn(
            cfg, device
        )
    else:
        raise ValueError(f"Model không hỗ trợ: {cfg['model']}")

    criterion = nn.MSELoss()
    tr = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"]
    )

    model, _ = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        evaluate_rmse_mae,
        epochs=tr["epochs"],
        patience=tr["patience"],
        ckpt_path=tr["checkpoint"],
    )

    evaluation = cfg.get("evaluation", {})
    k = evaluation.get("k", 10)
    negatives_per_positive = evaluation.get("negatives_per_positive", 99)
    batch_users = evaluation.get(
        "batch_users", 8 if cfg["model"] == "deepconn" else 256
    )
    ranking_metrics = evaluate_sampled_ranking(
        model,
        ranking_data["positives"],
        ranking_data["seen_index"],
        device,
        k=k,
        negatives_per_positive=negatives_per_positive,
        seed=cfg.get("seed", 42),
        batch_users=batch_users,
        user_encoded=ranking_data.get("user_encoded"),
        item_encoded=ranking_data.get("item_encoded"),
    )
    return evaluate_test(
        model, test_loader, device, cfg["model"], ranking_metrics=ranking_metrics
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Đường dẫn tới file YAML. Dùng nhiều lần để chạy nhiều config.",
    )
    args = parser.parse_args()

    results = {}
    for cfg_path in args.config:
        results[Path(cfg_path).stem] = run_one(cfg_path)

    if len(results) > 1:
        first_result = next(iter(results.values()))
        hr_key = next(key for key in first_result if key.startswith("hr@"))
        ndcg_key = next(key for key in first_result if key.startswith("ndcg@"))
        if any(hr_key not in result or ndcg_key not in result for result in results.values()):
            raise ValueError("All comparison configs must use the same ranking K.")
        print("\n" + "=" * 50)
        print("BẢNG SO SÁNH")
        print(
            f"{'Model':<12} | {'RMSE':>8} | {'MAE':>8} | "
            f"{hr_key.upper():>8} | {ndcg_key.upper():>8}"
        )
        print("-" * 58)
        for name, r in results.items():
            print(
                f"{name:<12} | {r['rmse']:>8.4f} | {r['mae']:>8.4f} | "
                f"{r[hr_key]:>8.4f} | {r[ndcg_key]:>8.4f}"
            )


if __name__ == "__main__":
    main()
