"""
utils/metrics.py
================
Ujednolicone metryki i format wyników dla badań porównawczych.

Każde z czterech podejść po ewaluacji produkuje obiekt `EvalResult`,
co umożliwia zestawienie ich w jednej tabeli i na wspólnych wykresach.

Mierzone wielkości:
  - dokładność (accuracy),
  - F1 (makro i ważone) – istotne przy niezbalansowanym FER2013,
  - precyzja / czułość per klasa,
  - macierz pomyłek,
  - czas inferencji na próbkę (ms),
  - liczba parametrów i rozmiar modelu.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import config


@dataclass
class EvalResult:
    """Kontener na komplet metryk pojedynczego modelu."""
    name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    per_class_f1: dict[str, float]
    confusion: list[list[int]]                # macierz pomyłek (lista list dla serializacji)
    inference_ms: float = 0.0                 # średni czas inferencji na próbkę [ms]
    num_params: int = 0                       # liczba parametrów (0 dla modeli klasycznych)
    model_size_mb: float = 0.0
    extra: dict = field(default_factory=dict)

    def summary_row(self) -> dict:
        """Zwraca jeden wiersz do tabeli porównawczej."""
        return {
            "Model": self.name,
            "Dokładność": round(self.accuracy, 4),
            "F1 (makro)": round(self.f1_macro, 4),
            "F1 (ważone)": round(self.f1_weighted, 4),
            "Inferencja [ms]": round(self.inference_ms, 3),
            "Parametry": self.num_params,
            "Rozmiar [MB]": round(self.model_size_mb, 2),
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, ensure_ascii=False, indent=2)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    inference_ms: float = 0.0,
    num_params: int = 0,
    model_size_mb: float = 0.0,
    extra: dict | None = None,
) -> EvalResult:
    """Liczy komplet metryk z wektorów etykiet prawdziwych i przewidzianych."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(config.NUM_CLASSES))

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    _, _, f1_each, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {config.EMOTIONS_PL[config.IDX_TO_LABEL[i]]: float(f1_each[i]) for i in labels}
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return EvalResult(
        name=name,
        accuracy=float(acc),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        per_class_f1=per_class,
        confusion=cm.tolist(),
        inference_ms=float(inference_ms),
        num_params=int(num_params),
        model_size_mb=float(model_size_mb),
        extra=extra or {},
    )


def text_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Czytelny raport klasyfikacji per klasa (do logów / pracy)."""
    return classification_report(
        y_true, y_pred,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.EMOTIONS_PL_LIST,
        zero_division=0,
        digits=4,
    )


# --------------------------------------------------------------------------- #
# Pomiary właściwości modeli PyTorch
# --------------------------------------------------------------------------- #
def count_parameters(model) -> int:
    """Liczba trenowalnych parametrów modelu PyTorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model) -> float:
    """Szacunkowy rozmiar modelu w MB (parametry + bufory)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def benchmark_inference(model, sample_input, n_iters: int = 100, warmup: int = 10) -> float:
    """
    Mierzy średni czas inferencji na pojedynczą próbkę [ms].

    Args:
        model: model PyTorch w trybie eval().
        sample_input: tensor wejściowy o kształcie (1, C, H, W).
        n_iters: liczba iteracji pomiarowych.
        warmup: liczba iteracji rozgrzewających (pomijanych w pomiarze).
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1000.0
