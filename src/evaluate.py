"""Đánh giá mô hình trên tập test độc lập."""

from .metrics import evaluate_rmse_mae


def evaluate_test(model, test_loader, device, model_name: str = "model"):
    rmse, mae = evaluate_rmse_mae(model, test_loader, device)
    print("\n" + "=" * 50)
    print(f"KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST ({model_name.upper()})")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print("=" * 50)
    return {"rmse": rmse, "mae": mae}
