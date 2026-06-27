"""
config.py
=========
Wspólna konfiguracja dla całego systemu rozpoznawania emocji.

Centralizuje parametry, które są współdzielone przez wszystkie cztery podejścia
algorytmiczne. Dzięki temu badania porównawcze są wykonywane w spójnych warunkach
(te same klasy, ten sam podział danych, to samo ziarno losowości).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # podejście geometryczne (landmarks) nie wymaga PyTorch
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Klasy emocji wg standardu FER2013 (kolejność etykiet jest istotna!)
# --------------------------------------------------------------------------- #
EMOTIONS: list[str] = [
    "angry",     # 0 - złość
    "disgust",   # 1 - obrzydzenie
    "fear",      # 2 - strach
    "happy",     # 3 - radość
    "sad",       # 4 - smutek
    "surprise",  # 5 - zaskoczenie
    "neutral",   # 6 - neutralność
]
EMOTIONS_PL: dict[str, str] = {
    "angry": "złość",
    "disgust": "wstręt",
    "fear": "strach",
    "happy": "radość",
    "sad": "smutek",
    "surprise": "zaskoczenie",
    "neutral": "neutralny",
}
EMOTIONS_PL_LIST: list[str] = [EMOTIONS_PL[e] for e in EMOTIONS]
NUM_CLASSES: int = len(EMOTIONS)
LABEL_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_LABEL: dict[int, str] = {i: e for i, e in enumerate(EMOTIONS)}

# Parametry obrazu FER2013
IMG_SIZE_NATIVE: int = 48          # natywny rozmiar FER2013 (48x48, skala szarości)
IMG_SIZE_TRANSFER: int = 96        # rozmiar dla sieci transfer-learning (upscaling)
IMG_SIZE_VIT_PRETRAINED: int = 224  # rozmiar dla wytrenowanych ViT z ImageNet

SEED: int = 42                     # ziarno dla powtarzalności eksperymentów

# Ścieżki (można nadpisać przez argumenty CLI)
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")


@dataclass
class TrainConfig:
    """Hiperparametry treningu wspólne dla modeli głębokich."""
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10               # cierpliwość early-stopping
    val_split: float = 0.1           # część zbioru treningowego na walidację (gdy brak osobnego)
    num_workers: int = 2
    use_class_weights: bool = True   # ważenie klas (FER2013 jest niezbalansowany!)
    label_smoothing: float = 0.05
    mixed_precision: bool = True     # AMP na GPU
    extra: dict = field(default_factory=dict)


def get_device():
    """Zwraca dostępne urządzenie obliczeniowe (CUDA jeśli dostępne)."""
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    """Ustawia ziarno losowości we wszystkich bibliotekach dla powtarzalności."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Dla pełnej determinacji (kosztem wydajności) odkomentuj:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    """Tworzy katalogi na checkpointy i wyniki, jeśli nie istnieją."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
