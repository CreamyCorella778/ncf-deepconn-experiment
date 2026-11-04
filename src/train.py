"""Vòng huấn luyện cho một epoch và toàn bộ quá trình (có early stopping)."""

import time
import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for a, b, y in loader:
        a, b, y = a.to(device), b.to(device), y.to(device)
        pred = torch.clamp(model(a, b), 1.0, 5.0)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    evaluate_fn,
    epochs: int,
    patience: int = 5,
    ckpt_path: str = "best_model.pt",
):
    """Huấn luyện có early stopping. Trả về model + history."""
    best_rmse = float("inf")
    bad_epochs = 0
    history = []

    print(
        f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val RMSE':>9} | {'Val MAE':>8} | {'Time(s)':>8}"
    )
    print("-" * 60)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_rmse, val_mae = evaluate_fn(model, val_loader, device)
        dt = time.time() - t0
        history.append((epoch, train_loss, val_rmse, val_mae))
        print(
            f"{epoch:>5} | {train_loss:>10.4f} | {val_rmse:>9.4f} | {val_mae:>8.4f} | {dt:>8.1f}"
        )

        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[EarlyStop] Dừng tại epoch {epoch}.")
                break

    print(f"[Best] Val RMSE = {best_rmse:.4f} -> {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, history
