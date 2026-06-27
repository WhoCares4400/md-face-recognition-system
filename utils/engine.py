"""
utils/engine.py
===============
Wspólny silnik treningu i ewaluacji dla modeli głębokich (PyTorch).

Trzy podejścia oparte na sieciach neuronowych (CNN-od-zera, transfer learning,
ViT) dzielą tę samą pętlę treningową. To celowy zabieg metodologiczny: różnice
w wynikach wynikają wówczas z architektury, a nie z różnic w procedurze uczenia.

Funkcjonalności:
  - early stopping na podstawie F1-macro walidacji,
  - scheduler kosinusowy z rozgrzewką,
  - ważenie klas (niezbalansowany FER2013),
  - mixed precision (AMP) na GPU,
  - zapis najlepszego checkpointu.
"""
from __future__ import annotations

import copy
import os
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import config
from utils import metrics as M


def _make_optimizer_scheduler(model, cfg: config.TrainConfig, steps_per_epoch: int):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.epochs * max(steps_per_epoch, 1)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ewaluacja modelu na całym loaderze.

    Returns:
        (y_true, y_pred, y_prob) – tablice numpy.
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        all_true.append(y.numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def train_model(
    model,
    loaders: dict,
    cfg: config.TrainConfig,
    device,
    model_name: str = "model",
    checkpoint_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Trenuje model PyTorch i zwraca historię oraz najlepsze wagi.

    Args:
        model: sieć neuronowa.
        loaders: słownik z 'train', 'val', 'test', 'class_counts'.
        cfg: hiperparametry treningu.
        device: urządzenie obliczeniowe.
        model_name: nazwa używana w logach i nazwie pliku checkpointu.
        checkpoint_path: gdzie zapisać najlepszy model (opcjonalnie).

    Returns:
        dict z kluczami: 'history', 'best_f1', 'best_state', 'epochs_run'.
    """
    model.to(device)

    # Funkcja straty z ważeniem klas i wygładzaniem etykiet
    if cfg.use_class_weights and "class_counts" in loaders:
        weights = M_data_weights(loaders["class_counts"]).to(device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    optimizer, scheduler = _make_optimizer_scheduler(model, cfg, len(loaders["train"]))
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(cfg.epochs):
        epochs_run = epoch + 1
        model.train()
        running_loss = 0.0
        t0 = time.perf_counter()

        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(loaders["train"].dataset)

        # Walidacja
        y_true, y_pred, _ = evaluate(model, loaders["val"], device)
        val_acc = float((y_true == y_pred).mean())
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        # strata walidacyjna (do śledzenia przeuczenia)
        val_loss = _val_loss(model, loaders["val"], criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if verbose:
            dt = time.perf_counter() - t0
            print(
                f"[{model_name}] epoka {epoch + 1:3d}/{cfg.epochs} | "
                f"strata_tren={train_loss:.4f} strata_wal={val_loss:.4f} "
                f"acc_wal={val_acc:.4f} f1_wal={val_f1:.4f} | {dt:.1f}s"
            )

        # Early stopping wg F1-macro (lepsze niż accuracy przy niezbalansowaniu)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                if verbose:
                    print(f"[{model_name}] early stopping po {epoch + 1} epokach.")
                break

    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_f1": best_f1,
        "best_state": best_state,
        "epochs_run": epochs_run,
    }


def _val_loss(model, loader, criterion, device) -> float:
    """Średnia strata na zbiorze walidacyjnym."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            total += criterion(model(x), y).item() * x.size(0)
    return total / len(loader.dataset)


def M_data_weights(class_counts):
    """Pomocniczo: import leniwy, by uniknąć cyklicznego importu z utils.data."""
    from utils.data import compute_class_weights
    return compute_class_weights(class_counts)


def full_evaluation(model, loaders, device, model_name: str, sample_shape) -> M.EvalResult:
    """
    Pełna ewaluacja modelu na zbiorze testowym + pomiar wydajności.

    Args:
        sample_shape: kształt pojedynczej próbki (C, H, W) do benchmarku inferencji.
    """
    y_true, y_pred, _ = evaluate(model, loaders["test"], device)
    sample = torch.randn(1, *sample_shape)
    inf_ms = M.benchmark_inference(model, sample)
    return M.compute_metrics(
        y_true, y_pred,
        name=model_name,
        inference_ms=inf_ms,
        num_params=M.count_parameters(model),
        model_size_mb=M.model_size_mb(model),
    )
