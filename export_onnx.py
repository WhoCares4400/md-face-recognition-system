"""
export_onnx.py
==============
Eksport pretrenowanych wag (checkpoints/) do formatu ONNX na potrzeby
demonstracji webowej działającej w przeglądarce (docs/, GitHub Pages).

Dlaczego ONNX: GitHub Pages serwuje wyłącznie pliki statyczne — inferencja musi
odbywać się po stronie klienta (ONNX Runtime Web / WASM). Skrypt konwertuje:

  * 6 sieci głębokich (PyTorch state_dict) -> ONNX (torch.onnx.export),
  * 2 modele "landmarkowe" (potok sklearn/XGBoost) -> ONNX (skl2onnx),

oraz generuje manifest docs/models/models.json opisujący każdy model
(preprocessing, rozmiar, dokładność) — plik ten steruje listą modeli na stronie.

Świadomie POMIJAMY dwa najcięższe modele (zbyt długie pobieranie w przeglądarce):
  * ViT-B/16          (~343 MB),
  * Landmarks + RandomForest (~430 MB).

Uruchomienie:
    pip install -r requirements-export.txt
    python export_onnx.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

# Polska konsola Windows (cp1250) nie koduje niektórych znaków w logach — wymuszamy UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import config
from models import model1_cnn, model2_transfer, model4_vit

OUT_DIR = os.path.join(config.PROJECT_ROOT, "docs", "models")
OPSET = 17

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRAY_MEAN = [0.507]
GRAY_STD = [0.255]

# Kolejność cech geometrycznych musi być zgodna z utils/landmarks.FEATURE_NAMES
FEATURE_DIM = 20


# --------------------------------------------------------------------------- #
# Definicje modeli do eksportu
# --------------------------------------------------------------------------- #
DEEP_MODELS = [
    dict(id="cnn", label="CNN mini-Xception", ckpt="cnn_main.pt", results="cnn_main",
         size=config.IMG_SIZE_NATIVE, channels=1, mean=GRAY_MEAN, std=GRAY_STD,
         build=lambda: model1_cnn.build_model()),
    dict(id="vit_lite", label="ViT-Lite (od zera)", ckpt="vit_main.pt", results="vit_main",
         size=config.IMG_SIZE_NATIVE, channels=1, mean=GRAY_MEAN, std=GRAY_STD,
         build=lambda: model4_vit.build_model(variant="lite", pretrained=False)),
    dict(id="mobilenet_v2", label="MobileNetV2 (transfer)", ckpt="transfer_main.pt",
         results="transfer_main", size=config.IMG_SIZE_TRANSFER, channels=3,
         mean=IMAGENET_MEAN, std=IMAGENET_STD,
         build=lambda: model2_transfer.build_model(backbone="mobilenet_v2", pretrained=False)),
    dict(id="efficientnet_b0", label="EfficientNet-B0 (transfer)", ckpt="transfer_effb0.pt",
         results="transfer_effb0", size=config.IMG_SIZE_TRANSFER, channels=3,
         mean=IMAGENET_MEAN, std=IMAGENET_STD,
         build=lambda: model2_transfer.build_model(backbone="efficientnet_b0", pretrained=False)),
    dict(id="resnet18", label="ResNet18 (transfer)", ckpt="transfer_resnet18.pt",
         results="transfer_resnet18", size=config.IMG_SIZE_TRANSFER, channels=3,
         mean=IMAGENET_MEAN, std=IMAGENET_STD,
         build=lambda: model2_transfer.build_model(backbone="resnet18", pretrained=False)),
    dict(id="resnet50", label="ResNet50 (transfer)", ckpt="transfer_resnet50.pt",
         results="transfer_resnet50", size=config.IMG_SIZE_TRANSFER, channels=3,
         mean=IMAGENET_MEAN, std=IMAGENET_STD,
         build=lambda: model2_transfer.build_model(backbone="resnet50", pretrained=False)),
]

LANDMARK_MODELS = [
    dict(id="landmarks_svm", label="Landmarki + SVM", pkl="landmarks_svm_main.pkl",
         results="landmarks_svm_main", kind="svm"),
    dict(id="landmarks_xgb", label="Landmarki + XGBoost", pkl="landmarks_xgb_xgb.pkl",
         results="landmarks_xgb_xgb", kind="xgb"),
]


# --------------------------------------------------------------------------- #
# Pomocnicze
# --------------------------------------------------------------------------- #
def _metrics(results_name: str) -> dict:
    """Wczytuje dokładność i macro-F1 z results/<name>.json (jeśli istnieje)."""
    path = os.path.join(config.RESULTS_DIR, f"{results_name}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return {"accuracy": round(float(d.get("accuracy", 0.0)), 4),
                "f1_macro": round(float(d.get("f1_macro", 0.0)), 4)}
    except (OSError, ValueError):
        return {"accuracy": None, "f1_macro": None}


# --------------------------------------------------------------------------- #
# Eksport sieci głębokich
# --------------------------------------------------------------------------- #
def export_deep(spec: dict) -> dict:
    import torch

    model = spec["build"]()
    state = torch.load(os.path.join(config.CHECKPOINT_DIR, spec["ckpt"]), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, spec["channels"], spec["size"], spec["size"])
    out_path = os.path.join(OUT_DIR, f"{spec['id']}.onnx")

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True,
        dynamo=False,  # stary (TorchScript) eksporter — nie wymaga onnxscript
    )

    # Walidacja: ONNX Runtime vs PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    size = os.path.getsize(out_path)
    print(f"  [OK] {spec['id']:16s} {size/1e6:7.2f} MB  max_diff={max_diff:.2e}")

    entry = {
        "id": spec["id"], "label": spec["label"], "file": f"{spec['id']}.onnx",
        "type": "image", "inputSize": spec["size"], "channels": spec["channels"],
        "mean": spec["mean"], "std": spec["std"], "output": "logits",
        "sizeBytes": size,
    }
    entry.update(_metrics(spec["results"]))
    if max_diff > 1e-2:
        print(f"  [UWAGA] duża rozbieżność ONNX vs PyTorch dla {spec['id']}: {max_diff:.3e}")
    return entry


# --------------------------------------------------------------------------- #
# Eksport modeli landmarkowych (sklearn / XGBoost -> ONNX)
#
# Potok to StandardScaler + klasyfikator. Ze względu na niezgodność typów między
# skl2onnx a onnxmltools (XGBoost) NIE eksportujemy scalera do grafu ONNX — jego
# parametry (mean_, scale_) zapisujemy w manifeście i standaryzację wykonuje JS.
# Do ONNX trafia sam klasyfikator, konwertowany przez onnxmltools (spójnie dla
# SVM i XGB). Dzięki temu preprocessing w przeglądarce jest jednolity.
# --------------------------------------------------------------------------- #
def _find_proba_output(outputs: list) -> np.ndarray:
    """Zwraca wyjście prawdopodobieństw (2D, N x liczba_klas) spośród wyjść ONNX."""
    for o in outputs:
        arr = np.asarray(o)
        if arr.ndim == 2 and arr.shape[1] == config.NUM_CLASSES:
            return arr
    # ostatnia deska ratunku: pierwsze 2D
    for o in outputs:
        arr = np.asarray(o)
        if arr.ndim == 2:
            return arr
    raise ValueError("Nie znaleziono wyjścia prawdopodobieństw w modelu ONNX")


def export_landmarks(spec: dict) -> dict:
    from onnxmltools.convert.common.data_types import FloatTensorType

    with open(os.path.join(config.CHECKPOINT_DIR, spec["pkl"]), "rb") as fh:
        pipeline = pickle.load(fh)

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    scaler_mean = scaler.mean_.astype(np.float64)
    scaler_scale = scaler.scale_.astype(np.float64)

    # onnxmltools/skl2onnx: konwerter XGBoost obsługuje maks. opset 15.
    landmark_opset = 15
    if spec["kind"] == "xgb":
        # XGBoost -> ONNX przez onnxmltools (własne typy danych).
        from onnxmltools.convert import convert_xgboost
        onnx_model = convert_xgboost(
            clf, initial_types=[("input", FloatTensorType([None, FEATURE_DIM]))],
            target_opset=landmark_opset,
        )
    else:  # svm (SVC) -> ONNX natywnie przez skl2onnx
        from skl2onnx import convert_sklearn as skl_convert
        from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
        onnx_model = skl_convert(
            clf, initial_types=[("input", SklFloatTensorType([None, FEATURE_DIM]))],
            target_opset=landmark_opset, options={id(clf): {"zipmap": False}},
        )

    out_path = os.path.join(OUT_DIR, f"{spec['id']}.onnx")
    with open(out_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())

    # Walidacja: ONNX Runtime (na ustandaryzowanych cechach) vs sklearn pipeline.
    import onnxruntime as ort
    rng = np.random.RandomState(0)
    sample = rng.randn(4, FEATURE_DIM).astype(np.float32)
    ref = pipeline.predict_proba(sample)                       # scaler + clf (sklearn)
    scaled = ((sample - scaler_mean) / scaler_scale).astype(np.float32)  # scaler w JS

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {"input": scaled})
    proba = _find_proba_output(outputs)
    proba_idx = next(i for i, o in enumerate(outputs)
                     if np.asarray(o).ndim == 2 and np.asarray(o).shape[1] == config.NUM_CLASSES)
    max_diff = float(np.abs(ref - proba).max())
    size = os.path.getsize(out_path)
    print(f"  [OK] {spec['id']:16s} {size/1e6:7.2f} MB  max_diff={max_diff:.2e}  "
          f"proba_out_idx={proba_idx}")
    if max_diff > 1e-3:
        print(f"  [UWAGA] rozbieznosc prawdopodobienstw dla {spec['id']}: {max_diff:.3e}")

    entry = {
        "id": spec["id"], "label": spec["label"], "file": f"{spec['id']}.onnx",
        "type": "landmarks", "featureDim": FEATURE_DIM,
        "output": "proba", "probaOutputIndex": proba_idx,
        "scalerMean": scaler_mean.tolist(), "scalerScale": scaler_scale.tolist(),
        "sizeBytes": size,
    }
    entry.update(_metrics(spec["results"]))
    return entry


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {
        "classes": config.EMOTIONS,
        "classesPl": config.EMOTIONS_PL_LIST,
        "excluded": [
            {"label": "ViT-B/16 (ImageNet)", "sizeMB": 343,
             "reason": "zbyt długie pobieranie w przeglądarce"},
            {"label": "Landmarki + RandomForest", "sizeMB": 430,
             "reason": "zbyt długie pobieranie w przeglądarce"},
        ],
        "models": [],
    }

    print("== Eksport sieci głębokich (torch.onnx) ==")
    for spec in DEEP_MODELS:
        try:
            manifest["models"].append(export_deep(spec))
        except Exception as exc:  # noqa: BLE001 - chcemy kontynuować pozostałe modele
            print(f"  [BŁĄD] {spec['id']}: {exc}")

    print("== Eksport modeli landmarkowych (skl2onnx) ==")
    for spec in LANDMARK_MODELS:
        try:
            manifest["models"].append(export_landmarks(spec))
        except Exception as exc:  # noqa: BLE001
            print(f"  [BŁĄD] {spec['id']}: {exc}  (pomijam ten model)")

    manifest_path = os.path.join(OUT_DIR, "models.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\nZapisano {len(manifest['models'])} modeli + manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
