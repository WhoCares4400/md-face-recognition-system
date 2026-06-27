"""
train.py
========
Jednolity punkt wejścia do treningu pojedynczego modelu.

Przykłady użycia:
    python train.py --model cnn       --data fer2013.csv --epochs 60
    python train.py --model transfer  --data ./fer2013/  --backbone mobilenet_v2
    python train.py --model vit        --data fer2013.csv --variant lite
    python train.py --model landmarks --data fer2013.csv --classifier svm

Dla modeli głębokich (cnn/transfer/vit) używana jest wspólna pętla z engine.py.
Dla modelu landmarks używana jest ścieżka klasyczna (ekstrakcja cech + sklearn).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import config
from config import TrainConfig, set_seed, get_device, ensure_dirs
from utils import data as data_mod
from utils import engine
from utils import plots
from utils import metrics as M
from utils.run_logger import RunLogger


def _build_deep_model(args):
    """Buduje model głęboki + zwraca metadane (typ danych, kształt próbki)."""
    if args.model == "cnn":
        from models import model1_cnn as mod
        model = mod.build_model()
        meta = dict(mod.MODEL_META)
    elif args.model == "transfer":
        from models import model2_transfer as mod
        model = mod.build_model(backbone=args.backbone, pretrained=not args.no_pretrained)
        meta = dict(mod.MODEL_META)
        meta["name"] = f"Uczenie transferowe ({args.backbone})"
    elif args.model == "vit":
        from models import model4_vit as mod
        model = mod.build_model(variant=args.variant, pretrained=not args.no_pretrained)
        meta = dict(mod.MODEL_META)
        if args.variant == "pretrained":
            meta["name"] = "Transformer wizyjny (ViT-B/16, ImageNet)"
            meta["model_type"] = "vit_pretrained"
            meta["sample_shape"] = (3, config.IMG_SIZE_VIT_PRETRAINED, config.IMG_SIZE_VIT_PRETRAINED)
    else:
        raise ValueError(args.model)
    return model, meta


def train_deep(args, cfg: TrainConfig):
    """Ścieżka treningu dla modeli PyTorch (cnn/transfer/vit)."""
    run_name = f"{args.model}_{args.tag}"
    with RunLogger(run_name=run_name, args=args, cfg=cfg) as logger:
        device = get_device()
        print(f"Urządzenie: {device}")

        model, meta = _build_deep_model(args)
        model_type = meta["model_type"]
        loaders = data_mod.get_dataloaders(args.data, model_type, cfg)
        print(f"Liczności klas (train): {loaders['class_counts'].tolist()}")

        ckpt = os.path.join(config.CHECKPOINT_DIR, f"{args.model}_{args.tag}.pt")

        # Trening dwuetapowy dla transfer learning (i opcjonalnie ViT pretrained)
        two_stage = meta.get("two_stage_finetune", False) or \
            (args.model == "vit" and args.variant == "pretrained")

        if two_stage and not args.no_pretrained:
            print("\n=== ETAP 1: ekstrakcja cech (zamrożony backbone) ===")
            model.freeze_backbone()
            cfg_stage1 = TrainConfig(**{**cfg.__dict__, "epochs": max(5, cfg.epochs // 4),
                                        "lr": cfg.lr})
            engine.train_model(model, loaders, cfg_stage1, device,
                               model_name=meta["name"] + " [etap1]", checkpoint_path=ckpt)

            print("\n=== ETAP 2: fine-tuning (odmrożone górne warstwy) ===")
            if hasattr(model, "unfreeze_last"):
                model.unfreeze_last(n_blocks=3)
            else:
                model.unfreeze_all()
            cfg_stage2 = TrainConfig(**{**cfg.__dict__, "lr": cfg.lr * 0.1})
            result_train = engine.train_model(model, loaders, cfg_stage2, device,
                                              model_name=meta["name"] + " [etap2]",
                                              checkpoint_path=ckpt)
        else:
            result_train = engine.train_model(model, loaders, cfg, device,
                                              model_name=meta["name"], checkpoint_path=ckpt)

        # Pełna ewaluacja na zbiorze testowym
        print("\n=== EWALUACJA (zbiór testowy) ===")
        eval_result = engine.full_evaluation(
            model, loaders, device, meta["name"], meta["sample_shape"]
        )
        print(M.text_report(*engine.evaluate(model, loaders["test"], device)[:2]))
        print(f"Accuracy={eval_result.accuracy:.4f} | F1-macro={eval_result.f1_macro:.4f} | "
              f"inferencja={eval_result.inference_ms:.2f} ms | "
              f"parametry={eval_result.num_params:,}")

        # Zapis wyników i wykresów
        ensure_dirs()
        eval_result.to_json(os.path.join(config.RESULTS_DIR, f"{args.model}_{args.tag}.json"))
        plots.plot_confusion(eval_result,
                             os.path.join(config.RESULTS_DIR, f"cm_{args.model}_{args.tag}.png"))
        plots.plot_training_curves(result_train["history"], meta["name"],
                                   os.path.join(config.RESULTS_DIR, f"curves_{args.model}_{args.tag}.png"))
        # Automatyczna dokumentacja: raport per-klasa, metadane, rejestr zbiorczy
        logger.log_result(eval_result, result_train)
    return eval_result


def train_landmarks(args, cfg: TrainConfig):
    """Ścieżka treningu dla podejścia geometrycznego (sklearn/xgboost)."""
    from models import model3_landmarks as mod

    run_name = f"landmarks_{args.classifier}_{args.tag}"
    with RunLogger(run_name=run_name, args=args, cfg=cfg) as logger:
        print("Wczytywanie surowych obrazów do ekstrakcji landmarków...")
        train_imgs, train_lab, test_imgs, test_lab = _load_raw_images(args.data)

        print(f"\nEkstrakcja cech (zbiór treningowy, {len(train_imgs)} obrazów)...")
        X_tr, y_tr, det_tr = mod.extract_feature_matrix(train_imgs, train_lab)
        print(f"Ekstrakcja cech (zbiór testowy, {len(test_imgs)} obrazów)...")
        X_te, y_te, det_te = mod.extract_feature_matrix(test_imgs, test_lab)

        print(f"\nTrening klasyfikatora: {args.classifier.upper()}")
        model = mod.LandmarkEmotionModel(kind=args.classifier, use_class_weights=cfg.use_class_weights)
        model.fit(X_tr, y_tr)

        name = f"Cechy geometryczne+{args.classifier.upper()}"
        eval_result = model.evaluate(X_te, y_te, name=name)
        eval_result.extra["detection_rate_test"] = round(det_te, 4)
        eval_result.extra["detection_rate_train"] = round(det_tr, 4)

        print(f"\nAccuracy={eval_result.accuracy:.4f} | F1-macro={eval_result.f1_macro:.4f} | "
              f"wykrywalność twarzy (test)={det_te*100:.1f}%")

        ensure_dirs()
        model.save(os.path.join(config.CHECKPOINT_DIR, f"landmarks_{args.classifier}_{args.tag}.pkl"))
        eval_result.to_json(os.path.join(config.RESULTS_DIR, f"landmarks_{args.classifier}_{args.tag}.json"))
        plots.plot_confusion(eval_result,
                             os.path.join(config.RESULTS_DIR, f"cm_landmarks_{args.classifier}_{args.tag}.png"))
        imp = model.feature_importance()
        if imp:
            plots.plot_feature_importance(
                imp, os.path.join(config.RESULTS_DIR, f"feat_imp_{args.classifier}_{args.tag}.png"))
        # Automatyczna dokumentacja (raport per-klasa zawiera też wykrywalność twarzy)
        logger.log_result(eval_result, None)
    return eval_result


def _load_raw_images(data_path: str):
    """Wczytuje surowe obrazy 48x48 (uint8) i etykiety dla ścieżki landmarks."""
    if os.path.isfile(data_path) and data_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(data_path)
        df.columns = [c.strip() for c in df.columns]

        def to_imgs(sub):
            imgs = np.stack([
                np.fromstring(p, dtype=np.uint8, sep=" ").reshape(48, 48)
                for p in sub["pixels"].to_numpy()
            ])
            return imgs, sub["emotion"].astype(int).to_numpy()

        if "Usage" in df.columns:
            tr = df[df["Usage"] == "Training"]
            te = df[df["Usage"] == "PrivateTest"]
        else:
            from sklearn.model_selection import train_test_split
            tr, te = train_test_split(df, test_size=0.2, random_state=config.SEED,
                                      stratify=df["emotion"])
        train_imgs, train_lab = to_imgs(tr)
        test_imgs, test_lab = to_imgs(te)
        return train_imgs, train_lab, test_imgs, test_lab

    elif os.path.isdir(data_path):
        import cv2
        from utils.data import (_find_split_dir, _match_class_to_config,
                                _TRAIN_DIR_NAMES, _TEST_DIR_NAMES)

        def load_dir(directory):
            subdirs = sorted([d for d in os.listdir(directory)
                              if os.path.isdir(os.path.join(directory, d))])
            mapping = _match_class_to_config(subdirs)  # spójne z config.EMOTIONS
            imgs, labs = [], []
            for name in subdirs:
                d = os.path.join(directory, name)
                for fn in os.listdir(d):
                    img = cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    imgs.append(cv2.resize(img, (48, 48)))
                    labs.append(mapping[name])
            return np.stack(imgs), np.array(labs)

        train_dir = _find_split_dir(data_path, _TRAIN_DIR_NAMES) or data_path
        test_dir = _find_split_dir(data_path, _TEST_DIR_NAMES)
        train_imgs, train_lab = load_dir(train_dir)
        if test_dir is not None:
            test_imgs, test_lab = load_dir(test_dir)
        else:
            from sklearn.model_selection import train_test_split
            train_imgs, test_imgs, train_lab, test_lab = train_test_split(
                train_imgs, train_lab, test_size=0.2,
                random_state=config.SEED, stratify=train_lab)
        return train_imgs, train_lab, test_imgs, test_lab
    raise FileNotFoundError(data_path)


def main():
    parser = argparse.ArgumentParser(description="Trening modelu rozpoznawania emocji.")
    parser.add_argument("--model", required=True,
                        choices=["cnn", "transfer", "vit", "landmarks"])
    parser.add_argument("--data", required=True, help="Ścieżka do CSV lub katalogu z danymi.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tag", default="run1", help="Etykieta eksperymentu (nazwa plików).")
    # opcje specyficzne
    parser.add_argument("--backbone", default="mobilenet_v2",
                        help="Backbone dla transfer learning.")
    parser.add_argument("--variant", default="lite", choices=["lite", "pretrained"],
                        help="Wariant ViT.")
    parser.add_argument("--classifier", default="svm", choices=["svm", "rf", "xgb"],
                        help="Klasyfikator dla podejścia landmarks.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Wyłącz wagi ImageNet (trening od zera).")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Ziarno losowości (do uruchomień powtarzalnych / wielokrotnych).")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    if args.model == "landmarks":
        train_landmarks(args, cfg)
    else:
        train_deep(args, cfg)


if __name__ == "__main__":
    main()
