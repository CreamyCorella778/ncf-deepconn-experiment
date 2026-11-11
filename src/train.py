"""Vòng huấn luyện cho một epoch và toàn bộ quá trình (có early stopping)."""

import math
import time
import torch


class EarlyStopping:
    def __init__(self, patience: int = 3, mode: str = "min"):
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.patience = patience
        self.mode = mode
        self.best_score = math.inf if mode == "min" else -math.inf
        self.counter = 0

    def __call__(self, score: float, model, ckpt_path: str) -> bool:
        if not math.isfinite(score):
            raise ValueError(f"Early stopping score must be finite, got {score}.")

        improved = (
            score < self.best_score if self.mode == "min" else score > self.best_score
        )
        if improved:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        *inputs, y = batch
        inputs = [value.to(device) for value in inputs]
        y = y.to(device)
        pred = model(*inputs)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def finish_wandb():
    """Finish the active W&B run if one was started."""
    import wandb

    if wandb.run is not None:
        wandb.finish()


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    evaluate_fn,
    epochs: int,
    patience: int = 3,
    ckpt_path: str = "best_model.pt",
    wandb_settings: dict = None,
):
    """Huấn luyện có early stopping. Trả về model + history."""
    if wandb_settings is not None:
        import wandb

        if wandb.run is None:
            wandb.init(
                project=wandb_settings["project"],
                name=wandb_settings.get("name"),
                config=wandb_settings.get("config"),
                mode=wandb_settings.get("mode"),
            )

    early_stopping = EarlyStopping(patience=patience, mode="min")
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
        if wandb_settings is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/rmse": val_rmse,
                    "val/mae": val_mae,
                    "epoch/seconds": dt,
                },
                step=epoch,
            )

        if early_stopping(val_rmse, model, ckpt_path):
            print(f"[EarlyStop] Dừng tại epoch {epoch}.")
            break

    print(f"[Best] Val RMSE = {early_stopping.best_score:.4f} -> {ckpt_path}")
    if wandb_settings is not None:
        wandb.run.summary["best/val_rmse"] = early_stopping.best_score
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, history
