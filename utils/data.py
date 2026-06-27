"""
utils/data.py
=============
Wczytywanie i przygotowanie zbioru FER2013.

Obsługiwane są dwa najczęstsze formaty dystrybucji FER2013:

1. Plik CSV (`fer2013.csv`) z kolumnami: `emotion`, `pixels`, `Usage`,
   gdzie `pixels` to 2304 (=48*48) wartości 0-255 oddzielone spacjami.
2. Układ katalogów (ImageFolder):
       root/
         train/<emotion>/*.png
         test/<emotion>/*.png

Moduł zwraca obiekty DataLoader z transformacjami dobranymi do konkretnego
podejścia algorytmicznego (CNN-od-zera, transfer learning, ViT).
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image

import config

ModelType = Literal["cnn", "transfer", "vit", "vit_pretrained"]


# --------------------------------------------------------------------------- #
# Transformacje – dobierane do wymagań wejściowych każdej architektury
# --------------------------------------------------------------------------- #
# Statystyki ImageNet (dla modeli wstępnie trenowanych)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
# Przybliżona statystyka FER2013 w skali szarości
_FER_MEAN = [0.507]
_FER_STD = [0.255]


def get_transforms(model_type: ModelType, train: bool):
    """
    Buduje potok transformacji dopasowany do architektury.

    - cnn:            48x48, 1 kanał (skala szarości), normalizacja FER
    - transfer:       96x96, 3 kanały (RGB), normalizacja ImageNet
    - vit:            48x48, 1 kanał (lekki ViT trenowany od zera)
    - vit_pretrained: 224x224, 3 kanały, normalizacja ImageNet
    """
    if model_type == "cnn":
        size, channels, mean, std = config.IMG_SIZE_NATIVE, 1, _FER_MEAN, _FER_STD
    elif model_type == "transfer":
        size, channels, mean, std = config.IMG_SIZE_TRANSFER, 3, _IMAGENET_MEAN, _IMAGENET_STD
    elif model_type == "vit":
        size, channels, mean, std = config.IMG_SIZE_NATIVE, 1, _FER_MEAN, _FER_STD
    elif model_type == "vit_pretrained":
        size, channels, mean, std = config.IMG_SIZE_VIT_PRETRAINED, 3, _IMAGENET_MEAN, _IMAGENET_STD
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

    ops: list = []
    # Konwersja kanałów
    if channels == 3:
        ops.append(transforms.Grayscale(num_output_channels=3))
    else:
        ops.append(transforms.Grayscale(num_output_channels=1))

    ops.append(transforms.Resize((size, size)))

    # Augmentacja tylko dla zbioru treningowego
    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    if train:
        # Random erasing po normalizacji – symuluje okluzje twarzy
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)))
    return transforms.Compose(ops)


# --------------------------------------------------------------------------- #
# Dataset oparty na CSV
# --------------------------------------------------------------------------- #
class FER2013CSVDataset(Dataset):
    """
    Dataset czytający FER2013 z pliku CSV.

    Args:
        df: ramka danych z kolumnami 'emotion' (int 0-6) i 'pixels' (str).
        transform: transformacja torchvision aplikowana do obrazu PIL.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.emotions = df["emotion"].astype(int).to_numpy()
        # Parsowanie kolumny pixels do tablicy (N, 48, 48) uint8
        pixels = df["pixels"].to_numpy()
        self.images = np.empty((len(df), config.IMG_SIZE_NATIVE, config.IMG_SIZE_NATIVE), dtype=np.uint8)
        for i, row in enumerate(pixels):
            arr = np.fromstring(row, dtype=np.uint8, sep=" ") if isinstance(row, str) else row
            self.images[i] = arr.reshape(config.IMG_SIZE_NATIVE, config.IMG_SIZE_NATIVE)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.emotions)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx], mode="L")
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.emotions[idx])
        return img, label


def _load_csv(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Wczytuje CSV i dzieli wg kolumny 'Usage' na train/val/test.

    FER2013 standardowo używa:
      - Training            -> zbiór treningowy
      - PublicTest          -> walidacja
      - PrivateTest         -> test
    Jeśli kolumna 'Usage' nie istnieje, dokonujemy podziału losowego.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Usage" in df.columns:
        train_df = df[df["Usage"] == "Training"].reset_index(drop=True)
        val_df = df[df["Usage"] == "PublicTest"].reset_index(drop=True)
        test_df = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)
    else:
        # Podział losowy 80/10/10 ze stratyfikacją
        from sklearn.model_selection import train_test_split
        train_df, tmp = train_test_split(
            df, test_size=0.2, random_state=config.SEED, stratify=df["emotion"]
        )
        val_df, test_df = train_test_split(
            tmp, test_size=0.5, random_state=config.SEED, stratify=tmp["emotion"]
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Obsługa układu katalogowego (ImageFolder) z poprawnym mapowaniem klas
# --------------------------------------------------------------------------- #
# Warianty nazw podkatalogów podziału
_TRAIN_DIR_NAMES = ("train", "training", "Training", "Train")
_TEST_DIR_NAMES = ("test", "testing", "Test", "val", "validation",
                   "PrivateTest", "PublicTest")


def _match_class_to_config(folder_names) -> dict[str, int]:
    """
    Buduje mapowanie {nazwa_folderu: indeks_klasy_wg_config}.

    Obsługuje:
      - angielskie nazwy emocji (bez względu na wielkość liter),
      - cyfry 0-6 interpretowane wg standardu FER2013 (0=angry ... 6=neutral).

    To gwarantuje, że etykiety z układu katalogowego są SPÓJNE z formatem CSV
    i z kolejnością config.EMOTIONS — inaczej ImageFolder ponumerowałby klasy
    alfabetycznie (neutral=4, sad=5, surprise=6), psując etykiety.
    """
    mapping = {}
    for name in folder_names:
        key = str(name).strip().lower()
        if key in config.LABEL_TO_IDX:
            mapping[name] = config.LABEL_TO_IDX[key]
        elif key.isdigit() and 0 <= int(key) < config.NUM_CLASSES:
            mapping[name] = int(key)
        else:
            raise ValueError(
                f"Nie rozpoznano nazwy folderu klasy: '{name}'. Oczekiwano jednej z "
                f"{config.EMOTIONS} lub cyfry 0-{config.NUM_CLASSES - 1}."
            )
    return mapping


def _make_imagefolder(directory: str, transform):
    """
    Tworzy ImageFolder i REMAPUJE etykiety na kolejność z config.EMOTIONS.

    Domyślnie ImageFolder numeruje klasy alfabetycznie wg nazw folderów, co dla
    FER2013 daje neutral=4, sad=5, surprise=6 — niezgodnie ze standardem
    (sad=4, surprise=5, neutral=6). Tutaj wymuszamy poprawne mapowanie.
    """
    ds = datasets.ImageFolder(directory, transform=transform)
    remap = _match_class_to_config(ds.classes)
    old_to_new = {old_idx: remap[name] for old_idx, name in enumerate(ds.classes)}
    ds.samples = [(p, old_to_new[i]) for (p, i) in ds.samples]
    ds.imgs = ds.samples
    ds.targets = [old_to_new[i] for i in ds.targets]
    ds.class_to_idx = {name: remap[name] for name in ds.classes}
    return ds


def _find_split_dir(root: str, candidates) -> str | None:
    """Zwraca pierwszy istniejący podkatalog z listy wariantów nazw lub None."""
    for c in candidates:
        d = os.path.join(root, c)
        if os.path.isdir(d):
            return d
    return None


# --------------------------------------------------------------------------- #
# Publiczny interfejs: budowa DataLoaderów
# --------------------------------------------------------------------------- #
def get_dataloaders(
    data_path: str,
    model_type: ModelType,
    cfg: config.TrainConfig | None = None,
):
    """
    Buduje DataLoadery train/val/test dla wskazanego typu modelu.

    Args:
        data_path: ścieżka do pliku CSV LUB katalogu z podkatalogami train/test.
        model_type: jeden z {'cnn', 'transfer', 'vit', 'vit_pretrained'}.
        cfg: konfiguracja treningu (batch_size, num_workers, ...).

    Returns:
        dict z kluczami 'train', 'val', 'test' (DataLoader) oraz
        'class_counts' (np.ndarray liczności klas w zbiorze treningowym).
    """
    cfg = cfg or config.TrainConfig()
    tf_train = get_transforms(model_type, train=True)
    tf_eval = get_transforms(model_type, train=False)

    if os.path.isfile(data_path) and data_path.endswith(".csv"):
        train_df, val_df, test_df = _load_csv(data_path)
        train_ds = FER2013CSVDataset(train_df, transform=tf_train)
        val_ds = FER2013CSVDataset(val_df, transform=tf_eval)
        test_ds = FER2013CSVDataset(test_df, transform=tf_eval)
        class_counts = np.bincount(train_df["emotion"].astype(int), minlength=config.NUM_CLASSES)

    elif os.path.isdir(data_path):
        train_dir = _find_split_dir(data_path, _TRAIN_DIR_NAMES)
        test_dir = _find_split_dir(data_path, _TEST_DIR_NAMES)

        # Przypadek: katalog bez train/ — może bezpośrednio zawiera foldery klas
        if train_dir is None:
            subdirs = [d for d in os.listdir(data_path)
                       if os.path.isdir(os.path.join(data_path, d))]
            try:
                _match_class_to_config(subdirs)
                train_dir = data_path  # cały katalog to jeden zbiór -> podzielimy
            except ValueError:
                raise FileNotFoundError(
                    f"W '{data_path}' nie znaleziono podkatalogu 'train' ani podkatalogów "
                    f"klas. Oczekiwano układu train/<emocja>/ oraz test/<emocja>/."
                )

        full_train = _make_imagefolder(train_dir, tf_train)
        full_train_eval = _make_imagefolder(train_dir, tf_eval)

        gen = torch.Generator().manual_seed(config.SEED)
        perm = torch.randperm(len(full_train), generator=gen).tolist()
        targets = np.array(full_train.targets)

        if test_dir is not None:
            # Standard FER2013: osobny zbiór testowy; walidację wydzielamy z train
            n_val = int(len(full_train) * cfg.val_split)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            test_ds = _make_imagefolder(test_dir, tf_eval)
        else:
            # Brak osobnego testu -> podział 80/10/10 (stały, powtarzalny)
            n_test = int(len(full_train) * 0.1)
            n_val = int(len(full_train) * 0.1)
            test_idx = perm[:n_test]
            val_idx = perm[n_test:n_test + n_val]
            train_idx = perm[n_test + n_val:]
            test_ds = torch.utils.data.Subset(full_train_eval, test_idx)
            print("[INFO] Brak podkatalogu testowego — dzielę dane 80/10/10 "
                  "(train/val/test).")

        train_ds = torch.utils.data.Subset(full_train, train_idx)
        val_ds = torch.utils.data.Subset(full_train_eval, val_idx)
        class_counts = np.bincount(targets[train_idx], minlength=config.NUM_CLASSES)
    else:
        raise FileNotFoundError(f"Nie znaleziono danych pod ścieżką: {data_path}")

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return {
        "train": DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
        "class_counts": class_counts,
    }


def compute_class_weights(class_counts: np.ndarray) -> torch.Tensor:
    """
    Oblicza wagi klas odwrotnie proporcjonalne do ich liczności.
    Niezbędne dla FER2013, gdzie klasa 'disgust' jest skrajnie rzadka.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)
