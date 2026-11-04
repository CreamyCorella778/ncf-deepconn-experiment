"""Đánh giá mô hình trên tập test độc lập."""

from .metrics import evaluate_rmse_mae


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
    return {"rmse": rmse, "mae": mae, **(ranking_metrics or {})}
