"""
Entry point: huấn luyện & đánh giá NCF hoặc DeepCoNN dựa trên file config.
Cách dùng:
    python run_compare.py --config configs/ncf.yaml
    python run_compare.py --config configs/deepconn.yaml
    python run_compare.py --config configs/ncf.yaml --config configs/deepconn.yaml   # chạy cả 2
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import set_seed, get_device, load_config, get_num_workers
from src.preprocessing import (
    load_ratings,
    load_pickle_dict,
    infer_num_ids,
    build_vocab,
)
from src.dataset import NCFDataset, DeepCoNNDataset
from src.metrics import evaluate_rmse_mae
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
    return model, train_loader, val_loader, test_loader


def build_deepconn(cfg, device):
    d = cfg["data"]
    train_df = load_ratings(d["dir"], d["train_file"])
    val_df = load_ratings(d["dir"], d["val_file"])
    test_df = load_ratings(d["dir"], d["test_file"])

    user_docs = load_pickle_dict(d["dir"], d["user_reviews"])
    item_docs = load_pickle_dict(d["dir"], d["item_reviews"])

    mp = cfg["model_params"]
    vocab = build_vocab(
        user_docs, item_docs, min_freq=mp["min_freq"], max_size=mp["max_vocab"]
    )
    print(f"[DeepCoNN] Vocab size = {len(vocab):,}")

    nw = get_num_workers(cfg["training"]["num_workers"])
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        DeepCoNNDataset(train_df, user_docs, item_docs, vocab, mp["max_len"]),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DeepCoNNDataset(val_df, user_docs, item_docs, vocab, mp["max_len"]),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )
    test_loader = DataLoader(
        DeepCoNNDataset(test_df, user_docs, item_docs, vocab, mp["max_len"]),
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
    return model, train_loader, val_loader, test_loader


def run_one(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))
    print(
        f"\n{'='*60}\n>>> Cấu hình: {cfg_path} | Model: {cfg['model']} | Device: {device}\n{'='*60}"
    )

    if cfg["model"] == "ncf":
        model, train_loader, val_loader, test_loader = build_ncf(cfg, device)
    elif cfg["model"] == "deepconn":
        model, train_loader, val_loader, test_loader = build_deepconn(cfg, device)
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

    return evaluate_test(model, test_loader, device, cfg["model"])


def main():
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
        print("\n" + "=" * 50)
        print("BẢNG SO SÁNH")
        print(f"{'Model':<12} | {'RMSE':>8} | {'MAE':>8}")
        print("-" * 36)
        for name, r in results.items():
            print(f"{name:<12} | {r['rmse']:>8.4f} | {r['mae']:>8.4f}")


if __name__ == "__main__":
    main()
