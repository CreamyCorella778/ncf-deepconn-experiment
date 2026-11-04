"""Các độ đo đánh giá."""

import torch


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
