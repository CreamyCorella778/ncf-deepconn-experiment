"""Đánh giá mô hình trên tập test độc lập."""

import pickle
from pathlib import Path

from .metrics import evaluate_rmse_mae


def load_test_negatives(path: Path) -> dict:
    """Load the precomputed test negatives from a pickle file."""
    with path.open("rb") as file:
        negative_samples = pickle.load(file)
    if not isinstance(negative_samples, dict):
        raise ValueError(f"Expected a user-to-negative-items dictionary in {path}.")
    return negative_samples


def evaluate_test(
    model,
    test_loader,
    device,
    model_name: str = "model",
    ranking_metrics: dict = None,
):
    rmse, mae = evaluate_rmse_mae(model, test_loader, device)
    print("\n" + "=" * 50)
    print(f"KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST ({model_name.upper()})")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    if ranking_metrics is not None:
        k = next(
            int(name.split("@", 1)[1])
            for name in ranking_metrics
            if name.startswith("hr@")
        )
        print(
            f"  HR@{k} = {ranking_metrics[f'hr@{k}']:.4f} "
            f"(1 positive + {ranking_metrics['negatives_per_positive']} sampled negatives)"
        )
        print(f"  NDCG@{k} = {ranking_metrics[f'ndcg@{k}']:.4f}")
    print("=" * 50)
    results = {"rmse": rmse, "mae": mae, **(ranking_metrics or {})}

    import wandb

    if wandb.run is not None:
        summary_metrics = {
            "test/rmse": rmse,
            "test/mae": mae,
        }
        if ranking_metrics is not None:
            summary_metrics.update(
                {
                    f"test/{name}": ranking_metrics[name]
                    for name in ranking_metrics
                    if name.startswith(("hr@", "ndcg@"))
                }
            )
        wandb.run.summary.update(summary_metrics)
    return results
