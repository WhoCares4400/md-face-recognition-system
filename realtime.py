"""
realtime.py
===========
Rozpoznawanie emocji w czasie rzeczywistym z kamery internetowej (OpenCV).

Obsługuje wszystkie cztery podejścia poprzez jednolity interfejs predyktora.
Pipeline dla każdej klatki:
  1. (opcjonalne) lustrzane odbicie obrazu kamery,
  2. detekcja twarzy (Haar cascade OpenCV — szybki, bez zależności od GPU),
  3. kwadratowe wycięcie regionu twarzy z marginesem kontekstowym,
  4. (opcjonalne) wyrównanie histogramu (CLAHE) dla modeli szaroskalowych,
  5. preprocessing zależny od modelu + predykcja emocji,
  6. wygładzanie czasowe prawdopodobieństw (EMA) i nałożenie nakładki.

Ulepszenia względem wersji podstawowej:
  * kwadratowe wycięcie z marginesem (mniej zniekształceń, zgodność z treningiem),
  * wygładzanie czasowe predykcji (EMA) — eliminuje migotanie etykiet,
  * opcjonalne CLAHE (odporność na zmienne oświetlenie),
  * lustrzane odbicie (naturalny podgląd typu „selfie"),
  * detekcja twarzy co N klatek (wyższy FPS),
  * automatyczny dobór pliku wag (checkpoints/<model>_main.*),
  * czytelne komunikaty błędów.

Przykłady:
    python realtime.py --model cnn
    python realtime.py --model transfer --backbone resnet18 --checkpoint checkpoints/transfer_resnet18.pt
    python realtime.py --model vit --variant pretrained --checkpoint checkpoints/vit_vitb16.pt
    python realtime.py --model landmarks --classifier svm
    python realtime.py --model cnn --equalize --detect-every 3 --no-mirror

Klawisze: 'q' = wyjście, 's' = zrzut ekranu.
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

import config


# --------------------------------------------------------------------------- #
# Jednolity predyktor opakowujący każdy typ modelu
# --------------------------------------------------------------------------- #
class EmotionPredictor:
    """Ujednolicony interfejs predykcji niezależny od typu modelu."""

    def __init__(self, model_type: str, checkpoint: str, **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        if model_type == "landmarks":
            self._init_landmarks(checkpoint, kwargs.get("classifier", "svm"))
        else:
            self._init_deep(model_type, checkpoint, kwargs)

    # ---- inicjalizacja modeli głębokich ----
    def _init_deep(self, model_type, checkpoint, kwargs):
        import torch
        self.torch = torch
        self.device = config.get_device()

        if model_type == "cnn":
            from models import model1_cnn as mod
            self.model = mod.build_model()
            self.input_size = config.IMG_SIZE_NATIVE
            self.channels = 1
        elif model_type == "transfer":
            from models import model2_transfer as mod
            self.model = mod.build_model(backbone=kwargs.get("backbone", "mobilenet_v2"),
                                         pretrained=False)
            self.input_size = config.IMG_SIZE_TRANSFER
            self.channels = 3
        elif model_type == "vit":
            from models import model4_vit as mod
            variant = kwargs.get("variant", "lite")
            self.model = mod.build_model(variant=variant, pretrained=False)
            if variant == "pretrained":
                self.input_size, self.channels = config.IMG_SIZE_VIT_PRETRAINED, 3
            else:
                self.input_size, self.channels = config.IMG_SIZE_NATIVE, 1
        else:
            raise ValueError(model_type)

        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # statystyki normalizacji (spójne z utils/data.py)
        if self.channels == 3:
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:
            self.mean = np.array([0.507], dtype=np.float32)
            self.std = np.array([0.255], dtype=np.float32)

    def _init_landmarks(self, checkpoint, classifier):
        from models import model3_landmarks as mod
        self.model = mod.LandmarkEmotionModel(kind=classifier)
        self.model.load(checkpoint)

    # ---- predykcja ----
    def predict(self, face_gray: np.ndarray):
        """
        Args:
            face_gray: wycięty obszar twarzy w skali szarości (uint8, dowolny rozmiar).
        Returns:
            (idx, proba) — indeks klasy i wektor prawdopodobieństw, lub (None, None).
        """
        if face_gray is None or face_gray.size == 0:
            return None, None
        if self.model_type == "landmarks":
            return self.model.predict_image(face_gray)
        return self._predict_deep(face_gray)

    def _predict_deep(self, face_gray):
        torch = self.torch
        img = cv2.resize(face_gray, (self.input_size, self.input_size))
        if self.channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            arr = img.astype(np.float32) / 255.0
            arr = (arr - self.mean) / self.std
            arr = np.transpose(arr, (2, 0, 1))
        else:
            arr = img.astype(np.float32) / 255.0
            arr = (arr - self.mean[0]) / self.std[0]
            arr = arr[None, :, :]
        tensor = torch.from_numpy(arr[None]).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return int(np.argmax(proba)), proba


# --------------------------------------------------------------------------- #
# Pomocnicze: dobór checkpointu, wycięcie twarzy, lekki tracker do EMA
# --------------------------------------------------------------------------- #
def resolve_checkpoint(args) -> str:
    """Zwraca ścieżkę wag — jawną (--checkpoint) lub domyślną checkpoints/<model>_main.*"""
    if args.checkpoint:
        path = args.checkpoint
    elif args.model == "landmarks":
        path = os.path.join(config.CHECKPOINT_DIR, f"landmarks_{args.classifier}_main.pkl")
    else:
        path = os.path.join(config.CHECKPOINT_DIR, f"{args.model}_main.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku wag: {path}\n"
            f"Podaj poprawną ścieżkę przez --checkpoint lub najpierw wytrenuj model "
            f"(python train.py --model {args.model} --data ./fer2013)."
        )
    return path


def crop_face_square(gray: np.ndarray, box, margin: float = 0.25) -> np.ndarray:
    """Wycina kwadratowy obszar twarzy z marginesem kontekstowym (z przycięciem do kadru)."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + margin)
    half = side / 2.0
    H, W = gray.shape[:2]
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
    return gray[y0:y1, x0:x1]


class SmoothTracker:
    """Lekki tracker łączący twarze między klatkami (po najbliższym środku) do EMA."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.tracks: list[dict] = []  # {center:(cx,cy), proba:np.ndarray}

    def update(self, detections: list[tuple]):
        """detections: lista (box, proba). Zwraca listę (box, idx, proba_wygładzone)."""
        new_tracks, out = [], []
        for box, proba in detections:
            if proba is None:
                out.append((box, None, None))
                continue
            x, y, w, h = box
            center = (x + w / 2.0, y + h / 2.0)
            thr = max(w, h) * 0.6
            match = None
            for t in self.tracks:
                d = np.hypot(center[0] - t["center"][0], center[1] - t["center"][1])
                if d < thr:
                    match = t
                    break
            if match is not None:
                proba = self.alpha * proba + (1.0 - self.alpha) * match["proba"]
                proba = proba / proba.sum()
            new_tracks.append({"center": center, "proba": proba})
            out.append((box, int(np.argmax(proba)), proba))
        self.tracks = new_tracks
        return out


# --------------------------------------------------------------------------- #
# Wizualizacja klatki
# --------------------------------------------------------------------------- #
# Kolory BGR dla każdej emocji (czytelna nakładka)
EMOTION_COLORS = {
    "angry": (60, 60, 220), "disgust": (60, 160, 60), "fear": (160, 60, 160),
    "happy": (40, 200, 240), "sad": (200, 120, 40), "surprise": (40, 220, 220),
    "neutral": (180, 180, 180),
}

# --------------------------------------------------------------------------- #
# Renderowanie polskich znaków (cv2.putText nie obsługuje diakrytyków)
# --------------------------------------------------------------------------- #
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_FONT_CACHE: dict[int, "ImageFont.FreeTypeFont"] = {}
# transliteracja zapasowa, gdy brak Pillow/fontu
_ASCII_MAP = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _ascii(text: str) -> str:
    return text.translate(_ASCII_MAP)


def _get_font(size: int):
    """Zwraca font TrueType obsługujący polskie znaki (z cache)."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = []
    try:  # DejaVuSans dołączany z matplotlibem — pewny, wieloplatformowy
        import matplotlib
        candidates.append(os.path.join(os.path.dirname(matplotlib.__file__),
                                       "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"))
    except Exception:
        pass
    candidates += [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    font = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def render_labels_pl(frame, labels, size: int = 20):
    """Nanosi etykiety z polskimi znakami. labels: lista (tekst, (x, y), kolor_BGR)."""
    if not labels:
        return frame
    if _PIL_OK:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = _get_font(size)
        for text, (x, y), color_bgr in labels:
            rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
            draw.text((x, y), text, font=font, fill=rgb)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # fallback bez Pillow — transliteracja na ASCII
    for text, (x, y), color_bgr in labels:
        cv2.putText(frame, _ascii(text), (x, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
    return frame


def draw_overlay(frame, box, idx, proba, labels):
    """Rysuje ramkę twarzy i słupki; etykietę (z polskimi znakami) dopisuje do `labels`."""
    x, y, w, h = box
    emotion = config.EMOTIONS[idx]
    color = EMOTION_COLORS.get(emotion, (0, 255, 0))
    label = f"{config.EMOTIONS_PL[emotion]} {proba[idx]*100:.0f}%"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, y - 28), (x + w, y), color, -1)
    # tekst nakładany później przez render_labels_pl (obsługa polskich znaków)
    labels.append((label, (x + 4, y - 26), (0, 0, 0)))

    # panel słupkowy z rozkładem prawdopodobieństw (prawy górny róg)
    bw, mr, lw = 150, 16, 96       # szer. paska, prawy margines, miejsce na etykietę z lewej
    bx = max(lw + 10, frame.shape[1] - bw - mr)
    by = 44                         # poniżej górnego paska HUD
    # półprzezroczyste tło panelu dla czytelności
    _translucent_bar(frame, bx - lw - 8, by - 8, bx + bw + 8, by + len(config.EMOTIONS) * 22, 0.45)
    for i, emo in enumerate(config.EMOTIONS):
        yy = by + i * 22
        bar_len = int(bw * float(proba[i]))
        c = EMOTION_COLORS.get(emo, (200, 200, 200))
        # etykieta (skrót + wartość %) PO LEWEJ stronie paska — zawsze w kadrze
        cv2.putText(frame, f"{emo[:4]} {proba[i]*100:3.0f}%", (bx - lw, yy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bx, yy), (bx + bar_len, yy + 16), c, -1)
        cv2.rectangle(frame, (bx, yy), (bx + bw, yy + 16), (90, 90, 90), 1)


# --------------------------------------------------------------------------- #
# Interfejs: presety modeli przełączane klawiszami 1-4 + menedżer z cache
# --------------------------------------------------------------------------- #
def _ckpt(name: str) -> str:
    return os.path.join(config.CHECKPOINT_DIR, name)


# Każdy preset ma listę kandydatów (plik_wag, kwargs); używany jest pierwszy istniejący.
DEFAULT_PRESETS = {
    "1": {"label": "CNN mini-Xception", "model": "cnn",
          "candidates": [(_ckpt("cnn_main.pt"), {})]},
    "2": {"label": "Transfer ResNet18", "model": "transfer",
          "candidates": [(_ckpt("transfer_resnet18.pt"), {"backbone": "resnet18"}),
                         (_ckpt("transfer_main.pt"), {"backbone": "mobilenet_v2"})]},
    "3": {"label": "ViT-B/16", "model": "vit",
          "candidates": [(_ckpt("vit_vitb16.pt"), {"variant": "pretrained"}),
                         (_ckpt("vit_main.pt"), {"variant": "lite"})]},
    "4": {"label": "Geometria + SVM", "model": "landmarks",
          "candidates": [(_ckpt("landmarks_svm_main.pkl"), {"classifier": "svm"})]},
}
MODEL_TO_KEY = {"cnn": "1", "transfer": "2", "vit": "3", "landmarks": "4"}


def build_presets(args):
    """Buduje presety, nadpisując wybrany w CLI model jawnymi opcjami użytkownika."""
    presets = {k: {**v, "candidates": list(v["candidates"])} for k, v in DEFAULT_PRESETS.items()}
    key = MODEL_TO_KEY[args.model]
    kw = {}
    if args.model == "transfer":
        kw = {"backbone": args.backbone}
    elif args.model == "vit":
        kw = {"variant": args.variant}
    elif args.model == "landmarks":
        kw = {"classifier": args.classifier}
    if args.checkpoint:  # jawnie podane wagi mają priorytet
        presets[key]["candidates"].insert(0, (args.checkpoint, kw))
    return presets, key


class PredictorManager:
    """Tworzy i buforuje predyktory; pozwala przełączać model w trakcie działania."""

    def __init__(self, presets):
        self.presets = presets
        self.cache: dict[str, EmotionPredictor] = {}
        self.active: str | None = None

    def _resolve(self, key):
        for ckpt, kw in self.presets[key]["candidates"]:
            if os.path.isfile(ckpt):
                return ckpt, kw
        return None

    def get(self, key):
        """Ładuje (i buforuje) predyktor presetu BEZ zmiany aktywnego. None gdy brak wag."""
        if key in self.cache:
            return self.cache[key]
        if key not in self.presets:
            return None
        sel = self._resolve(key)
        if sel is None:
            return None
        ckpt, kw = sel
        try:
            print(f"Wczytywanie '{self.presets[key]['model']}' z: {ckpt} ...")
            self.cache[key] = EmotionPredictor(self.presets[key]["model"], ckpt, **kw)
        except Exception as exc:
            print(f"Blad ladowania {self.presets[key]['label']}: {exc}")
            return None
        return self.cache[key]

    def switch(self, key):
        """Przełącza aktywny model. Zwraca (sukces, komunikat)."""
        if key not in self.presets:
            return False, "nieznany preset"
        label = self.presets[key]["label"]
        if self.get(key) is None:
            return False, f"Brak wag dla: {label}"
        self.active = key
        return True, f"Model: {label}"

    @property
    def predictor(self):
        return self.cache[self.active]

    @property
    def active_label(self):
        return self.presets[self.active]["label"]


def _translucent_bar(frame, x0, y0, x1, y1, alpha=0.5):
    """Półprzezroczyste ciemne tło pod tekst (czytelność HUD)."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def make_comparison(frame_bgr, gray, faces, manager, margin=0.25):
    """Uruchamia wszystkie 4 modele na TEJ SAMEJ klatce i składa montaż 2x2."""
    box = max(faces, key=lambda b: b[2] * b[3]) if len(faces) else None
    tiles = []
    for key in ("1", "2", "3", "4"):
        tile = frame_bgr.copy()
        lines = []
        pred = manager.get(key)
        title = f"[{key}] {manager.presets[key]['label']}"
        if pred is None:
            lines.append(("brak wag modelu", (20, 44), (0, 0, 255)))
        elif box is None:
            lines.append(("brak wykrytej twarzy", (20, 44), (0, 0, 255)))
        else:
            face = crop_face_square(gray, box, margin=margin)
            idx, proba = pred.predict(face)
            if idx is not None:
                draw_overlay(tile, box, idx, proba, lines)
            else:
                x, y = box[0], box[1]
                lines.append(("brak detekcji cech", (x, y - 26), (0, 0, 255)))
        _translucent_bar(tile, 0, 0, tile.shape[1], 32)
        lines.append((title, (10, 5), (0, 255, 0)))
        tiles.append(render_labels_pl(tile, lines, size=18))

    # kafelki w pełnej rozdzielczości (montaż 2x większy — ostry obraz do pracy)
    montage = np.vstack([np.hstack([tiles[0], tiles[1]]),
                         np.hstack([tiles[2], tiles[3]])])
    return montage


# Przyciski bocznego panelu: (akcja, etykieta, klawisz)
PANEL_W = 220
PANEL_BUTTONS = [
    ("model1", "CNN", "1"), ("model2", "ResNet18", "2"),
    ("model3", "ViT-B/16", "3"), ("model4", "SVM", "4"),
    ("compare", "Porównaj 4", "C"),
    ("clahe", "CLAHE", "E"), ("mirror", "Lustro", "M"),
    ("smooth-", "Wygładz. −", "−"), ("smooth+", "Wygładz. +", "+"),
    ("panel", "Panel wł/wył", "H"), ("snapshot", "Zrzut", "S"),
    ("quit", "Wyjście", "Q"),
]


def build_panel(height, x_off, st, manager):
    """Buduje boczny panel przycisków. Zwraca (obraz_panelu, etykiety_canvas, prostokąty_canvas)."""
    panel = np.full((height, PANEL_W, 3), 38, np.uint8)
    labels = [("STEROWANIE", (x_off + 12, 12), (0, 255, 0))]
    rects = []
    y, bh, gap = 40, 30, 6
    for action, text, k in PANEL_BUTTONS:
        x0, x1 = 10, PANEL_W - 10
        y0, y1 = y, y + bh
        active = action.startswith("model") and action[-1] == manager.active
        toggle = (action == "clahe" and st["equalize"]) or (action == "mirror" and st["mirror"])
        color = (40, 160, 40) if active else (150, 110, 30) if toggle else (70, 70, 70)
        cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (120, 120, 120), 1)
        lbl = text
        if action == "clahe":
            lbl = f"CLAHE: {'wł' if st['equalize'] else 'wył'}"
        elif action == "mirror":
            lbl = f"Lustro: {'wł' if st['mirror'] else 'wył'}"
        labels.append((f"{lbl}  ({k})", (x_off + x0 + 10, y0 + 8), (240, 240, 240)))
        rects.append((action, x_off + x0, y0, x_off + x1, y1))
        y += bh + gap
    labels.append((f"Wygładzanie: {st['smooth']:.1f}", (x_off + 12, y + 2), (200, 200, 200)))
    return panel, labels, rects


def _on_mouse(event, x, y, flags, st):
    if event == cv2.EVENT_LBUTTONDOWN:
        for action, x0, y0, x1, y1 in st.get("rects", []):
            if x0 <= x <= x1 and y0 <= y <= y1:
                st["click"] = action
                break


# Mapowanie klawiszy na akcje (wspólne dla klawiatury i myszy)
KEYMAP = {
    ord("1"): "model1", ord("2"): "model2", ord("3"): "model3", ord("4"): "model4",
    ord("c"): "compare", ord("e"): "clahe", ord("m"): "mirror", ord("h"): "panel",
    ord("s"): "snapshot", ord("q"): "quit",
    ord("+"): "smooth+", ord("="): "smooth+", ord("-"): "smooth-", ord("_"): "smooth-",
}


def dispatch(action, st, manager):
    """Wykonuje akcję (z klawiatury lub myszy), aktualizując stan `st`."""
    now = time.time()
    if action == "quit":
        st["running"] = False
    elif action in ("model1", "model2", "model3", "model4"):
        ok, msg = manager.switch(action[-1])
        st["status"] = (msg, now)
        if ok:
            st["tracker"] = SmoothTracker(alpha=st["smooth"])
    elif action == "clahe":
        st["equalize"] = not st["equalize"]
        st["status"] = (f"CLAHE: {'wł' if st['equalize'] else 'wył'}", now)
    elif action == "mirror":
        st["mirror"] = not st["mirror"]
        st["status"] = (f"Lustro: {'wł' if st['mirror'] else 'wył'}", now)
    elif action == "panel":
        st["show_panel"] = not st["show_panel"]
    elif action in ("smooth+", "smooth-"):
        step = 0.1 if action == "smooth+" else -0.1
        st["smooth"] = min(1.0, max(0.1, round(st["smooth"] + step, 1)))
        st["tracker"].alpha = st["smooth"]
        st["status"] = (f"Wygładzanie: {st['smooth']:.1f}", now)
    elif action == "snapshot":
        st["snapshot"] = True
    elif action == "compare":
        st["compare"] = True


def run(args):
    presets, init_key = build_presets(args)
    manager = PredictorManager(presets)
    ok, msg = manager.switch(init_key)
    if not ok:  # awaryjnie wczytaj pierwszy dostępny preset
        for k in ("1", "2", "3", "4"):
            ok, msg = manager.switch(k)
            if ok:
                break
    if not ok:
        raise FileNotFoundError(
            "Brak jakichkolwiek wag w katalogu checkpoints/. "
            "Wytrenuj najpierw model: python train.py --model cnn --data ./fer2013"
        )
    print(msg)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    os.makedirs(args.out_dir, exist_ok=True)  # katalog na zrzuty/porównania

    # współdzielony stan (klawiatura + mysz aktualizują to samo)
    st = {
        "equalize": args.equalize, "mirror": args.mirror, "smooth": args.smooth,
        "show_panel": True, "running": True, "status": (msg, time.time()),
        "tracker": SmoothTracker(alpha=args.smooth),
        "snapshot": False, "compare": False, "rects": [], "click": None,
    }

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Nie można otworzyć kamery o indeksie {args.camera}. "
            f"Sprawdź podłączenie kamery lub podaj inny indeks (--camera)."
        )
    # poproś o wyższą rozdzielczość źródłową (ostrzejszy obraz i zrzuty/porównania)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    print("Sterowanie: [1-4] model | [C] porównaj 4 | [E] CLAHE | [M] lustro | "
          "[+/-] wygładzanie | [H] panel | [S] zrzut | [Q] wyjście (lub klikaj panel)")
    win_name = "Rozpoznawanie emocji (q=wyjscie)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win_name, _on_mouse, st)

    fps_t0, fps_count, fps = time.time(), 0, 0.0
    frame_idx = 0
    last_faces: list = []
    last_canvas_shape = None   # do dopasowania rozmiaru okna do proporcji canvasu

    while st["running"]:
        ok, frame = cap.read()
        if not ok:
            break
        if st["mirror"]:
            frame = cv2.flip(frame, 1)
        if args.width and frame.shape[1] < args.width:  # powiększenie klatki
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, (int(args.width), int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clean_frame = frame.copy()  # czysta klatka (bez HUD) — do montażu porównawczego

        # detekcja twarzy co N klatek (reszta klatek reużywa ostatnich ramek)
        if frame_idx % max(1, args.detect_every) == 0:
            last_faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(args.min_size, args.min_size)
            )
        frame_idx += 1

        # predykcja aktywnego modelu dla każdej wykrytej twarzy
        detections = []
        for (x, y, w, h) in last_faces:
            face = crop_face_square(gray, (x, y, w, h), margin=args.margin)
            if st["equalize"] and face.size > 0:
                face = clahe.apply(face)
            idx, proba = manager.predictor.predict(face)
            detections.append(((x, y, w, h), proba))

        labels = []
        for box, idx, proba in st["tracker"].update(detections):
            if idx is not None:
                draw_overlay(frame, box, idx, proba, labels)

        # licznik FPS
        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps = fps_count / (time.time() - fps_t0)
            fps_t0, fps_count = time.time(), 0

        # górny pasek + status
        _translucent_bar(frame, 0, 0, frame.shape[1], 34)
        labels.append((f"{manager.active_label}  |  {fps:.0f} FPS  |  {len(last_faces)} twarz(y)",
                       (10, 7), (0, 255, 0)))
        status_msg, status_t = st["status"]
        if status_msg and time.time() - status_t < 2.0:
            labels.append((status_msg, (10, 38), (40, 220, 255)))
        frame = render_labels_pl(frame, labels, size=18)

        # boczny panel przycisków
        if st["show_panel"]:
            panel, plabels, rects = build_panel(frame.shape[0], frame.shape[1], st, manager)
            st["rects"] = rects
            canvas = np.hstack([frame, panel])
            canvas = render_labels_pl(canvas, plabels, size=15)
        else:
            st["rects"] = []
            canvas = frame

        # dopasuj rozmiar okna do faktycznych proporcji canvasu (raz i po zmianie szer.)
        cur_shape = (canvas.shape[1], canvas.shape[0])
        if cur_shape != last_canvas_shape:
            cw, ch = cur_shape
            if cw > 1500:                       # ogranicz wstępną szerokość do ekranu
                s = 1500 / cw
                cw, ch = int(cw * s), int(ch * s)
            cv2.resizeWindow(win_name, cw, ch)
            last_canvas_shape = cur_shape

        cv2.imshow(win_name, canvas)

        # obsługa zdarzeń odroczonych (potrzebują bieżącej klatki)
        if st["snapshot"]:
            fn = os.path.join(args.out_dir, f"snapshot_{int(time.time())}.png")
            cv2.imwrite(fn, canvas)
            st["status"] = (f"Zapisano: {fn}", time.time())
            print(f"Zapisano: {fn}")
            st["snapshot"] = False
        if st["compare"]:
            st["status"] = ("Porównanie 4 modeli...", time.time())
            montage = make_comparison(clean_frame, gray, last_faces, manager, margin=args.margin)
            fn = os.path.join(args.out_dir, f"compare_models_{int(time.time())}.png")
            cv2.imwrite(fn, montage)  # zapis w pełnej rozdzielczości (2x klatka)
            cv2.namedWindow("Porownanie 4 modeli", cv2.WINDOW_NORMAL)
            disp_w = min(montage.shape[1], 1280)  # podgląd dopasowany do ekranu
            cv2.resizeWindow("Porownanie 4 modeli", disp_w,
                             int(montage.shape[0] * disp_w / montage.shape[1]))
            cv2.imshow("Porownanie 4 modeli", montage)
            st["status"] = (f"Zapisano: {fn}", time.time())
            print(f"Zapisano porównanie: {fn}")
            st["compare"] = False

        # akcje: klawiatura + mysz przez wspólny dispatcher
        key = cv2.waitKey(1) & 0xFF
        action = KEYMAP.get(key)
        if st["click"] is not None:
            action = st["click"]
            st["click"] = None
        if action:
            dispatch(action, st, manager)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Rozpoznawanie emocji na żywo z kamery.")
    parser.add_argument("--model", default="cnn",
                        choices=["cnn", "transfer", "vit", "landmarks"],
                        help="Model początkowy (można zmieniać w trakcie klawiszami 1-4).")
    parser.add_argument("--checkpoint", default=None,
                        help="Ścieżka do wag/modelu. Domyślnie checkpoints/<model>_main.*")
    parser.add_argument("--camera", type=int, default=0, help="Indeks kamery.")
    parser.add_argument("--out-dir", default="screenshots", dest="out_dir",
                        help="Katalog zapisu zrzutów i porównań (tworzony automatycznie).")
    parser.add_argument("--width", type=int, default=960,
                        help="Szerokość okna podglądu w px (klatka skalowana w górę; 0 = bez skalowania).")
    parser.add_argument("--cam-width", type=int, default=1280, dest="cam_width",
                        help="Żądana szerokość przechwytywania z kamery (ostrość zrzutów/porównań).")
    parser.add_argument("--cam-height", type=int, default=720, dest="cam_height",
                        help="Żądana wysokość przechwytywania z kamery.")
    parser.add_argument("--backbone", default="mobilenet_v2",
                        help="Sieć bazowa dla --model transfer.")
    parser.add_argument("--variant", default="lite", choices=["lite", "pretrained"],
                        help="Wariant dla --model vit.")
    parser.add_argument("--classifier", default="svm", choices=["svm", "rf", "xgb"],
                        help="Klasyfikator dla --model landmarks.")
    # parametry jakości / wydajności
    parser.add_argument("--margin", type=float, default=0.25,
                        help="Margines kontekstowy wokół wykrytej twarzy (udział, np. 0.25).")
    parser.add_argument("--smooth", type=float, default=0.2,
                        help="Współczynnik EMA wygładzania predykcji (1.0 = brak wygładzania).")
    parser.add_argument("--equalize", action="store_true",
                        help="Włącz wyrównanie histogramu (CLAHE) — pomaga przy złym oświetleniu.")
    parser.add_argument("--detect-every", type=int, default=2, dest="detect_every",
                        help="Wykrywaj twarze co N klatek (większe = wyższy FPS).")
    parser.add_argument("--min-size", type=int, default=60, dest="min_size",
                        help="Minimalny rozmiar wykrywanej twarzy w pikselach.")
    parser.add_argument("--mirror", action="store_true", default=True,
                        help="Lustrzane odbicie obrazu kamery (domyślnie włączone).")
    parser.add_argument("--no-mirror", action="store_false", dest="mirror",
                        help="Wyłącz lustrzane odbicie obrazu.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
