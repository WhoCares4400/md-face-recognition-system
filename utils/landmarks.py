"""
utils/landmarks.py
==================
Ekstrakcja punktów charakterystycznych twarzy (face landmarks) i wyliczanie
interpretowalnych cech geometrycznych — fundament PODEJŚCIA 3.

Wykorzystuje MediaPipe Face Mesh (468 punktów 3D). Z surowych punktów liczone są
znormalizowane cechy opisujące konfigurację mięśni mimicznych (zgodnie z intuicją
systemu kodowania ruchów twarzy FACS):

  - otwarcie / szerokość ust,
  - uniesienie kącików ust (uśmiech vs. grymas),
  - otwarcie oczu, uniesienie brwi,
  - odległości brew-oko, marszczenie nosa, itd.

Wszystkie odległości są normalizowane przez rozstaw oczu (odległość
międzyźreniczna), co czyni cechy niezależnymi od skali i odległości twarzy
od kamery. To podejście jest lekkie i w pełni interpretowalne — kluczowa
zaleta w dyskusji wyników pracy.
"""
from __future__ import annotations

import numpy as np

try:
    import mediapipe as mp
    _HAS_MP = True
    # Wykrycie dostępnego API MediaPipe:
    #   - "solutions": klasyczne mp.solutions.face_mesh (większość instalacji)
    #   - "tasks":     nowe API FaceLandmarker (niektóre buildy mediapipe >= 0.10)
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        _MP_API = "solutions"
    else:
        _MP_API = "tasks"
except ImportError:
    _HAS_MP = False
    _MP_API = None

# URL modelu dla API Tasks (pobierany automatycznie przy pierwszym użyciu)
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


# --------------------------------------------------------------------------- #
# Indeksy kluczowych punktów w siatce MediaPipe Face Mesh (468 punktów)
# --------------------------------------------------------------------------- #
# (numeracja zgodna z oficjalną topologią canonical_face_model)
IDX = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "left_brow_inner": 55,
    "left_brow_outer": 105,
    "right_brow_inner": 285,
    "right_brow_outer": 334,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "upper_lip_top": 0,
    "lower_lip_bottom": 17,
    "nose_tip": 1,
    "nose_bridge": 168,
    "chin": 152,
    "left_cheek": 50,
    "right_cheek": 280,
}

# Nazwy cech wyjściowych (przydatne do analizy ważności cech w pracy)
FEATURE_NAMES = [
    "mouth_open_ratio", "mouth_width_ratio", "mouth_corner_lift",
    "left_eye_open", "right_eye_open", "eye_open_mean",
    "left_brow_eye_dist", "right_brow_eye_dist", "brow_eye_mean",
    "brow_inner_dist", "mouth_to_nose", "nose_wrinkle",
    "lip_press", "jaw_drop", "mouth_aspect_ratio",
    "left_brow_raise", "right_brow_raise", "smile_asymmetry",
    "cheek_raise", "mouth_curvature",
]

# Polskie etykiety cech (do wykresów ważności cech w pracy)
FEATURE_NAMES_PL = {
    "mouth_open_ratio": "stosunek otwarcia ust",
    "mouth_width_ratio": "stosunek szerokości ust",
    "mouth_corner_lift": "uniesienie kącików ust",
    "left_eye_open": "otwarcie lewego oka",
    "right_eye_open": "otwarcie prawego oka",
    "eye_open_mean": "średnie otwarcie oczu",
    "left_brow_eye_dist": "odległość lewa brew–oko",
    "right_brow_eye_dist": "odległość prawa brew–oko",
    "brow_eye_mean": "średnia odległość brew–oko",
    "brow_inner_dist": "odległość między brwiami",
    "mouth_to_nose": "odległość usta–nos",
    "nose_wrinkle": "zmarszczenie nosa",
    "lip_press": "zaciśnięcie warg",
    "jaw_drop": "opuszczenie żuchwy",
    "mouth_aspect_ratio": "proporcja ust (MAR)",
    "left_brow_raise": "uniesienie lewej brwi",
    "right_brow_raise": "uniesienie prawej brwi",
    "smile_asymmetry": "asymetria uśmiechu",
    "cheek_raise": "uniesienie policzków",
    "mouth_curvature": "krzywizna ust",
}


def _ensure_task_model(model_path: str | None = None) -> str:
    """
    Zapewnia obecność pliku modelu .task dla API Tasks; pobiera go, jeśli brak.
    Zwraca ścieżkę do pliku modelu.
    """
    import os
    import urllib.request

    if model_path is None:
        cache = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        os.makedirs(cache, exist_ok=True)
        model_path = os.path.join(cache, "face_landmarker.task")
    if not os.path.exists(model_path):
        print(f"Pobieranie modelu MediaPipe FaceLandmarker -> {model_path}")
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, model_path)
    return model_path


class LandmarkExtractor:
    """
    Opakowanie na detektor punktów charakterystycznych twarzy MediaPipe.

    Automatycznie wybiera dostępne API:
      - klasyczne mp.solutions.face_mesh (preferowane, najczęstsze),
      - nowe API Tasks (FaceLandmarker) z modelem .task jako fallback.

    Z obrazu (numpy RGB lub grayscale) zwraca tablicę 468x3 znormalizowanych
    współrzędnych punktów lub None, gdy nie wykryto twarzy.
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        min_detection_confidence: float = 0.3,
        task_model_path: str | None = None,
    ):
        if not _HAS_MP:
            raise ImportError("MediaPipe nie jest zainstalowane (pip install mediapipe).")
        self.api = _MP_API
        if self.api == "solutions":
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
            )
        else:
            # API Tasks: FaceLandmarker z pliku modelu
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            model_path = _ensure_task_model(task_model_path)
            running_mode = (vision.RunningMode.IMAGE if static_image_mode
                            else vision.RunningMode.VIDEO)
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=running_mode,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self._video_ts = 0

    def extract(self, image: np.ndarray) -> np.ndarray | None:
        """
        Args:
            image: obraz HxW (grayscale) lub HxWx3 (RGB), dtype uint8.
        Returns:
            tablica (468, 3) współrzędnych znormalizowanych [0,1] lub None,
            gdy nie wykryto twarzy.
        """
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if self.api == "solutions":
            res = self.mesh.process(image)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
            return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            if hasattr(self, "_video_ts"):  # tryb VIDEO wymaga znacznika czasu
                try:
                    res = self.landmarker.detect_for_video(mp_image, self._video_ts)
                    self._video_ts += 33
                except Exception:
                    res = self.landmarker.detect(mp_image)
            else:
                res = self.landmarker.detect(mp_image)
            if not res.face_landmarks:
                return None
            lm = res.face_landmarks[0]
            return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    def close(self):
        if self.api == "solutions":
            self.mesh.close()
        else:
            self.landmarker.close()


def _dist(pts: np.ndarray, a: int, b: int) -> float:
    """Odległość euklidesowa (2D) między dwoma punktami siatki."""
    return float(np.linalg.norm(pts[a, :2] - pts[b, :2]))


def compute_geometric_features(pts: np.ndarray) -> np.ndarray:
    """
    Przekształca surowe punkty (468x3) w wektor interpretowalnych cech.

    Wszystkie odległości normalizowane są przez rozstaw oczu, co zapewnia
    niezmienniczość względem skali twarzy.

    Returns:
        wektor cech o długości len(FEATURE_NAMES).
    """
    # Skala normalizująca: odległość międzyźreniczna
    eye_dist = _dist(pts, IDX["left_eye_outer"], IDX["right_eye_outer"])
    eye_dist = max(eye_dist, 1e-6)

    def nd(a, b):  # odległość znormalizowana
        return _dist(pts, a, b) / eye_dist

    # --- Usta ---
    mouth_open = nd(IDX["mouth_top"], IDX["mouth_bottom"])
    mouth_width = nd(IDX["mouth_left"], IDX["mouth_right"])
    # uniesienie kącików ust względem środka ust (oś Y; mniejsze y = wyżej)
    mouth_center_y = (pts[IDX["mouth_top"], 1] + pts[IDX["mouth_bottom"], 1]) / 2
    corner_y = (pts[IDX["mouth_left"], 1] + pts[IDX["mouth_right"], 1]) / 2
    mouth_corner_lift = float((mouth_center_y - corner_y) / eye_dist)

    # --- Oczy ---
    left_eye_open = nd(IDX["left_eye_top"], IDX["left_eye_bottom"])
    right_eye_open = nd(IDX["right_eye_top"], IDX["right_eye_bottom"])
    eye_open_mean = (left_eye_open + right_eye_open) / 2

    # --- Brwi ---
    left_brow_eye = nd(IDX["left_brow_inner"], IDX["left_eye_top"])
    right_brow_eye = nd(IDX["right_brow_inner"], IDX["right_eye_top"])
    brow_eye_mean = (left_brow_eye + right_brow_eye) / 2
    brow_inner_dist = nd(IDX["left_brow_inner"], IDX["right_brow_inner"])

    # --- Pozostałe ---
    mouth_to_nose = nd(IDX["mouth_top"], IDX["nose_tip"])
    nose_wrinkle = nd(IDX["nose_tip"], IDX["nose_bridge"])
    lip_press = nd(IDX["upper_lip_top"], IDX["lower_lip_bottom"])
    jaw_drop = nd(IDX["nose_tip"], IDX["chin"])
    mouth_aspect_ratio = mouth_open / max(mouth_width, 1e-6)

    # uniesienie brwi względem linii oczu (osobno L/P -> wykrywa asymetrie)
    left_brow_raise = float((pts[IDX["left_eye_top"], 1] - pts[IDX["left_brow_inner"], 1]) / eye_dist)
    right_brow_raise = float((pts[IDX["right_eye_top"], 1] - pts[IDX["right_brow_inner"], 1]) / eye_dist)
    smile_asymmetry = abs(
        (pts[IDX["mouth_left"], 1] - pts[IDX["mouth_right"], 1]) / eye_dist
    )
    cheek_raise = nd(IDX["left_cheek"], IDX["left_eye_bottom"])
    # krzywizna ust: kąciki względem środka (dodatnia = uśmiech)
    mouth_curvature = float(
        ((pts[IDX["mouth_left"], 1] + pts[IDX["mouth_right"], 1]) / 2
         - pts[IDX["mouth_top"], 1]) / eye_dist
    )

    feats = np.array([
        mouth_open, mouth_width, mouth_corner_lift,
        left_eye_open, right_eye_open, eye_open_mean,
        left_brow_eye, right_brow_eye, brow_eye_mean,
        brow_inner_dist, mouth_to_nose, nose_wrinkle,
        lip_press, jaw_drop, mouth_aspect_ratio,
        left_brow_raise, right_brow_raise, smile_asymmetry,
        cheek_raise, mouth_curvature,
    ], dtype=np.float32)
    return feats


def image_to_features(image: np.ndarray, extractor: LandmarkExtractor) -> np.ndarray | None:
    """Pełny potok: obraz -> punkty -> wektor cech (lub None gdy brak twarzy)."""
    pts = extractor.extract(image)
    if pts is None:
        return None
    return compute_geometric_features(pts)
