"""
utils/plots.py
==============
Wizualizacja wyników badań porównawczych (gotowe do wklejenia do pracy).

Generuje:
  - macierze pomyłek (znormalizowane) dla każdego modelu,
  - zbiorczy wykres słupkowy dokładności i F1,
  - wykres rozproszenia dokładność vs. czas inferencji (kompromis jakość/szybkość),
  - krzywe uczenia (strata i F1 w funkcji epoki) dla modeli głębokich.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # backend bez GUI – zapis do plików
import matplotlib.pyplot as plt
import numpy as np

import config


def plot_confusion(result, save_path: str, normalize: bool = True) -> None:
    """Rysuje macierz pomyłek z pliku wyniku (EvalResult)."""
    cm = np.array(result.confusion, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.EMOTIONS_PL_LIST, rotation=45, ha="right")
    ax.set_yticklabels(config.EMOTIONS_PL_LIST)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawda")
    ax.set_title(f"Macierz pomyłek — {result.name}")
    thresh = cm.max() / 2.0
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_comparison_bars(results: list, save_path: str) -> None:
    """Słupkowe porównanie dokładności i F1-macro wszystkich modeli."""
    names = [r.name for r in results]
    acc = [r.accuracy for r in results]
    f1 = [r.f1_macro for r in results]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.8), 5))
    ax.bar(x - width / 2, acc, width, label="Dokładność")
    ax.bar(x + width / 2, f1, width, label="F1 (makro)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Wartość metryki")
    ax.set_ylim(0, 1)
    ax.set_title("Porównanie dokładności i F1-makro")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_speed(results: list, save_path: str) -> None:
    """Wykres kompromisu: dokładność vs. czas inferencji (z rozmiarem = liczba param)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        size = max(r.num_params, 1)
        ax.scatter(r.inference_ms, r.accuracy, s=80, alpha=0.7)
        ax.annotate(r.name, (r.inference_ms, r.accuracy),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Czas inferencji na próbkę [ms]")
    ax.set_ylabel("Dokładność")
    ax.set_title("Kompromis dokładność / szybkość inferencji")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: dict, model_name: str, save_path: str) -> None:
    """Krzywe uczenia: strata treningowa/walidacyjna oraz F1-macro walidacji."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="strata treningowa")
    ax1.plot(epochs, history["val_loss"], label="strata walidacyjna")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Strata")
    ax1.set_title(f"Krzywe straty — {model_name}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_acc"], label="dokładność (wal.)")
    ax2.plot(epochs, history["val_f1"], label="F1-makro (wal.)")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Wartość metryki")
    ax2.set_title(f"Metryki walidacyjne — {model_name}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance: dict, save_path: str, top_k: int = 20) -> None:
    """Wykres ważności cech geometrycznych (dla modelu landmarks z RF/XGB)."""
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    ax.barh(names, vals)
    ax.set_xlabel("Ważność cechy")
    ax.set_title("Ważność cech geometrycznych (model landmarks)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
