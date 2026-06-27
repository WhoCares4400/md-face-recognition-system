"""
models/model3_landmarks.py
==========================
PODEJŚCIE 3: Cechy geometryczne + klasyczny uczenie maszynowe.

Zupełnie inny paradygmat niż uczenie głębokie: zamiast uczyć sieć surowych
pikseli, najpierw ekstrahujemy z twarzy interpretowalne cechy geometryczne
(patrz utils/landmarks.py), a następnie klasyfikujemy je klasycznym modelem ML:
SVM (RBF), Random Forest lub XGBoost.

Charakterystyka dla pracy magisterskiej:
  + pełna interpretowalność (można wskazać, które cechy decydują o emocji),
  + bardzo lekki, szybki, działa bez GPU,
  + odporny na zmiany oświetlenia (operuje na geometrii, nie na pikselach),
  - zależny od jakości detekcji landmarków (problem przy okluzji/profilu),
  - traci informację o teksturze (np. zmarszczki, rumieniec).

Ten model NIE jest siecią PyTorch — ma własną procedurę uczenia i ewaluacji,
ale produkuje ten sam obiekt EvalResult, co zapewnia spójność porównania.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

import config
from utils import metrics as M
from utils.landmarks import LandmarkExtractor, image_to_features, FEATURE_NAMES


def build_classifier(kind: str = "svm", class_weight=None):
    """
    Buduje potok (skalowanie -> klasyfikator) dla wybranego algorytmu.

    Args:
        kind: 'svm' | 'rf' | 'xgb'.
        class_weight: 'balanced' lub None (XGBoost obsługuje przez sample_weight).
    """
    if kind == "svm":
        clf = SVC(
            kernel="rbf", C=10.0, gamma="scale",
            class_weight=class_weight, probability=True,
            random_state=config.SEED,
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight=class_weight, n_jobs=-1, random_state=config.SEED,
        )
    elif kind == "xgb":
        if not _HAS_XGB:
            raise ImportError("XGBoost nie jest zainstalowany (pip install xgboost).")
        clf = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            objective="multi:softprob", num_class=config.NUM_CLASSES,
            tree_method="hist", random_state=config.SEED, n_jobs=-1,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"Nieznany klasyfikator: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# --------------------------------------------------------------------------- #
# Ekstrakcja cech ze zbioru obrazów
# --------------------------------------------------------------------------- #
def extract_feature_matrix(images: np.ndarray, labels: np.ndarray, verbose: bool = True):
    """
    Z tablicy obrazów buduje macierz cech X i wektor etykiet y.

    Obrazy, dla których nie wykryto twarzy, są pomijane (raportowana jest
    liczba odrzuconych próbek — istotny wskaźnik niezawodności tego podejścia).

    Args:
        images: tablica (N, 48, 48) uint8 (skala szarości).
        labels: tablica (N,) etykiet całkowitych.
    Returns:
        (X, y, detection_rate)
    """
    extractor = LandmarkExtractor(static_image_mode=True)
    feats_list, labels_list = [], []
    n_failed = 0
    try:
        for i, (img, lab) in enumerate(zip(images, labels)):
            f = image_to_features(img, extractor)
            if f is None:
                n_failed += 1
                continue
            feats_list.append(f)
            labels_list.append(int(lab))
            if verbose and (i + 1) % 2000 == 0:
                print(f"  przetworzono {i + 1}/{len(images)} (odrzucone: {n_failed})")
    finally:
        extractor.close()

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    detection_rate = len(X) / max(len(images), 1)
    if verbose:
        print(f"  wykryto twarz w {detection_rate*100:.1f}% obrazów "
              f"({len(X)}/{len(images)})")
    return X, y, detection_rate


class LandmarkEmotionModel:
    """
    Kompletny model: ekstrakcja cech + klasyfikator klasyczny.
    Udostępnia jednolity interfejs fit/evaluate/predict_image.
    """

    def __init__(self, kind: str = "svm", use_class_weights: bool = True):
        self.kind = kind
        cw = "balanced" if use_class_weights else None
        # XGBoost nie przyjmuje class_weight w konstruktorze -> obsługa przez wagi próbek
        self.pipeline = build_classifier(kind, class_weight=None if kind == "xgb" else cw)
        self.use_class_weights = use_class_weights
        self._extractor = None  # leniwa inicjalizacja do predykcji online

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        sample_weight = None
        if self.kind == "xgb" and self.use_class_weights:
            # waga próbki = odwrotność częstości klasy
            counts = np.bincount(y, minlength=config.NUM_CLASSES).astype(float)
            counts[counts == 0] = 1.0
            inv = counts.sum() / (len(counts) * counts)
            sample_weight = inv[y]
            self.pipeline.fit(X, y, clf__sample_weight=sample_weight)
        else:
            self.pipeline.fit(X, y)

    def evaluate(self, X: np.ndarray, y: np.ndarray, name: str | None = None) -> M.EvalResult:
        """Ewaluacja na gotowej macierzy cech testowych."""
        name = name or f"Landmarks+{self.kind.upper()}"
        # pomiar czasu inferencji na próbkę (sam klasyfikator, bez ekstrakcji)
        t0 = time.perf_counter()
        y_pred = self.pipeline.predict(X)
        elapsed_ms = (time.perf_counter() - t0) / max(len(X), 1) * 1000.0

        # rozmiar modelu = rozmiar zserializowanego pliku
        size_mb = len(pickle.dumps(self.pipeline)) / (1024 ** 2)
        return M.compute_metrics(
            y, y_pred, name=name,
            inference_ms=elapsed_ms, num_params=0, model_size_mb=size_mb,
            extra={"classifier": self.kind},
        )

    def predict_image(self, image: np.ndarray):
        """
        Predykcja dla pojedynczego obrazu (używane w trybie real-time).
        Returns:
            (idx_klasy, wektor_prawdopodobieństw) lub (None, None) gdy brak twarzy.
        """
        if self._extractor is None:
            self._extractor = LandmarkExtractor(static_image_mode=False)
        f = image_to_features(image, self._extractor)
        if f is None:
            return None, None
        proba = self.pipeline.predict_proba(f.reshape(1, -1))[0]
        return int(np.argmax(proba)), proba

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.pipeline, fh)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            self.pipeline = pickle.load(fh)

    def feature_importance(self) -> dict[str, float] | None:
        """Ważność cech (tylko dla modeli drzewiastych: rf, xgb)."""
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            return dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))
        return None


MODEL_META = {
    "name": "Cechy geometryczne + ML (SVM/RF/XGB)",
    "model_type": "landmarks",
    "is_classical": True,   # sygnał dla pipeline: nie używaj pętli PyTorch
}
