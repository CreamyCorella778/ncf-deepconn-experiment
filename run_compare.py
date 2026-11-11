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

from src.utils import (
    set_seed,
    get_device,
    load_config,
    get_num_workers,
    resolve_data_dir,
)
from src.preprocessing import (
    load_ratings,
    infer_num_ids,
    build_vocab,
    load_review_pickle,
)
from src.dataset import NCFDataset, DeepCoNNDataset, encode_review_docs
from src.embeddings import build_embedding_matrix
from src.metrics import (
    build_seen_items_index,
    evaluate_rmse_mae,
    evaluate_sampled_ranking,
)
from src.train import fit, finish_wandb
from src.evaluate import evaluate_test, load_test_negatives
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
    pretrained_emb = None
    glove_file = mp.get("glove_file")
    if glove_file:
        glove_path = Path(glove_file)
        if not glove_path.is_absolute():
            glove_path = Path(__file__).resolve().parent / glove_path
        pretrained_emb = torch.from_numpy(
            build_embedding_matrix(
                vocab,
                glove_path,
                emb_dim=mp["embed_dim"],
                seed=cfg.get("seed", 42),
                cache_path=mp.get("glove_cache"),
            )
        )
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
        n_users=n_users,
        n_items=n_items,
        embed_dim=mp["embed_dim"],
        out_channels=mp["out_channels"],
        kernel_size=mp["kernel_size"],
        latent_dim=mp["latent_dim"],
        dropout=mp["dropout"],
        pretrained_emb=pretrained_emb,
    ).to(device)
    ranking_data = {
        "positives": test_df[["user_idx", "item_idx"]],
        "seen_index": seen_index,
        "user_encoded": user_encoded,
        "item_encoded": item_encoded,
    }
    return model, train_loader, val_loader, test_loader, ranking_data


SWEEP_PARAMETER_TARGETS = {
    "lr": ("training", "lr"),
    "batch_size": ("training", "batch_size"),
    "weight_decay": ("training", "weight_decay"),
    "dropout": ("model_params", "dropout"),
    "gmf_dim": ("model_params", "gmf_dim"),
    "latent_dim": ("model_params", "latent_dim"),
}


def _apply_sweep_parameters(cfg, sweep_parameters):
    allowed = {
        "ncf": {"lr", "batch_size", "weight_decay", "dropout", "gmf_dim"},
        "deepconn": {"lr", "batch_size", "weight_decay", "dropout", "latent_dim"},
    }
    if cfg["model"] not in allowed:
        raise ValueError(f"Model không hỗ trợ sweep: {cfg['model']}")

    applied = []
    for parameter in allowed[cfg["model"]]:
        if parameter in sweep_parameters:
            section, key = SWEEP_PARAMETER_TARGETS[parameter]
            cfg[section][key] = sweep_parameters[parameter]
            applied.append(parameter)
    if "dataset" in sweep_parameters:
        _apply_dataset(cfg, sweep_parameters["dataset"])
        applied.append("dataset")
    if not applied:
        raise ValueError(
            f"Sweep config không có tham số hợp lệ cho model '{cfg['model']}'."
        )
    print(f"[W&B Sweep] Applied parameters: {', '.join(sorted(applied))}")


def _apply_dataset(cfg, dataset):
    if dataset not in {"amazon", "yelp"}:
        raise ValueError(f"Dataset không hỗ trợ: {dataset}")
    cfg["data"]["dataset"] = dataset
    cfg["data"]["dir"] = f"data/processed/{dataset}"


def _run_config(cfg, cfg_path: str, wandb_settings, sweep: bool = False):
    project_root = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(cfg["data"], project_root)
    cfg["data"]["dir"] = str(data_dir)
    evaluation = cfg.get("evaluation", {})
    negatives_path = Path(
        evaluation.get("negatives_file", "data/processed/test_negatives.pkl")
    )
    if not negatives_path.is_absolute():
        negatives_path = project_root / negatives_path
    negative_samples = None if sweep else load_test_negatives(negatives_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))
    dataset_name = cfg["data"].get("dataset", "unspecified")
    print(
        f"\n{'='*60}\n>>> Cấu hình: {cfg_path} | Model: {cfg['model']} | "
        f"Dataset: {dataset_name} | Data dir: {data_dir} | Device: {device}\n"
        f"{'='*60}"
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

    model, history = fit(
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
        wandb_settings=wandb_settings,
    )

    if sweep:
        return {"best/val_rmse": min(epoch_metrics[2] for epoch_metrics in history)}

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
        negative_samples=negative_samples,
        k=k,
        negatives_per_positive=negatives_per_positive,
        batch_users=batch_users,
        user_encoded=ranking_data.get("user_encoded"),
        item_encoded=ranking_data.get("item_encoded"),
    )
    return evaluate_test(
        model, test_loader, device, cfg["model"], ranking_metrics=ranking_metrics
    )


def run_one(cfg_path: str, sweep: bool = False, dataset: str = None):
    cfg = load_config(cfg_path)
    if dataset is not None and not sweep:
        _apply_dataset(cfg, dataset)
    wandb_config = cfg.get("wandb", {})
    wandb_settings = None
    should_finish_wandb = sweep or wandb_config.get("enabled", False)
    try:
        if sweep:
            import wandb

            project = wandb_config.get("project", "ncf-deepconn-experiment")
            if wandb.run is None:
                wandb.init(project=project, config=cfg)
            _apply_sweep_parameters(cfg, wandb.config.as_dict())
            if dataset is not None:
                _apply_dataset(cfg, dataset)
            cfg["training"]["checkpoint"] = str(
                Path(__file__).resolve().parent
                / f"best_{cfg['model']}_{wandb.run.id}.pt"
            )
            wandb_settings = {"project": project, "config": cfg}
        elif wandb_config.get("enabled", False):
            dataset_name = cfg["data"].get("dataset", "unspecified")
            wandb_settings = {
                "project": wandb_config.get("project", "ncf-deepconn-experiment"),
                "name": wandb_config.get("name", f"{dataset_name}-{cfg['model']}"),
                "mode": wandb_config.get("mode"),
                "config": cfg,
            }
        return _run_config(cfg, cfg_path, wandb_settings, sweep=sweep)
    finally:
        if should_finish_wandb:
            finish_wandb()


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
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Apply hyperparameters supplied by a W&B Sweep agent.",
    )
    parser.add_argument(
        "--dataset",
        choices=("amazon", "yelp"),
        help="Override the dataset directory without editing the model config.",
    )
    args = parser.parse_args()

    if args.sweep and len(args.config) != 1:
        parser.error("--sweep accepts exactly one --config.")

    results = {}
    for cfg_path in args.config:
        results[Path(cfg_path).stem] = run_one(
            cfg_path, sweep=args.sweep, dataset=args.dataset
        )

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
