"""
compare.py
==========
Badania porównawcze czterech podejść algorytmicznych.

Skrypt trenuje (lub wczytuje gotowe) wszystkie modele, ewaluuje je w identycznych
warunkach na zbiorze testowym FER2013 i generuje:

  - zbiorczą tabelę metryk (CSV + wydruk konsolowy),
  - słupkowy wykres porównawczy accuracy / F1,
  - wykres kompromisu dokładność vs. szybkość,
  - macierze pomyłek dla każdego modelu.

To centralny element warstwy eksperymentalnej pracy magisterskiej: wszystkie
modele są oceniane na tym samym podziale danych i tymi samymi metrykami, więc
różnice w wynikach są przypisywalne wyłącznie podejściu algorytmicznemu.

Przykład:
    python compare.py --data fer2013.csv --epochs 60 \
        --models cnn transfer vit landmarks
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import config
from config import TrainConfig, set_seed, ensure_dirs
from utils import plots
import train as train_mod


def run_all(args):
    set_seed(args.seed)
    ensure_dirs()
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    results = []

    for model_name in args.models:
        print("\n" + "=" * 70)
        print(f"  MODEL: {model_name.upper()}")
        print("=" * 70)
        set_seed(args.seed)  # ten sam punkt startowy dla każdego modelu

        # Budujemy lekki obiekt args dla funkcji treningowych z train.py
        sub = argparse.Namespace(
            model=model_name, data=args.data, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, tag=args.tag,
            backbone=args.backbone, variant=args.variant,
            classifier=args.classifier, no_pretrained=args.no_pretrained,
            seed=args.seed,
        )
        try:
            if model_name == "landmarks":
                res = train_mod.train_landmarks(sub, cfg)
            else:
                res = train_mod.train_deep(sub, cfg)
            results.append(res)
        except Exception as exc:  # nie przerywaj całego porównania przez jeden model
            print(f"[OSTRZEŻENIE] Model {model_name} zakończył się błędem: {exc}")

    if not results:
        print("Brak wyników — żaden model nie zakończył się sukcesem.")
        return

    # ---- Tabela zbiorcza ----
    table = pd.DataFrame([r.summary_row() for r in results])
    table = table.sort_values("F1 (makro)", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(config.RESULTS_DIR, f"comparison_{args.tag}.csv")
    table.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("  ZBIORCZE WYNIKI PORÓWNANIA")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nTabela zapisana: {csv_path}")

    # ---- Wykresy zbiorcze ----
    plots.plot_comparison_bars(
        results, os.path.join(config.RESULTS_DIR, f"compare_bars_{args.tag}.png"))
    plots.plot_accuracy_vs_speed(
        results, os.path.join(config.RESULTS_DIR, f"compare_speed_{args.tag}.png"))
    print(f"Wykresy zapisane w katalogu: {config.RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Badania porównawcze modeli emocji.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+",
                        default=["cnn", "transfer", "vit", "landmarks"],
                        choices=["cnn", "transfer", "vit", "landmarks"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tag", default="compare")
    parser.add_argument("--backbone", default="mobilenet_v2")
    parser.add_argument("--variant", default="lite", choices=["lite", "pretrained"])
    parser.add_argument("--classifier", default="svm", choices=["svm", "rf", "xgb"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Ziarno losowości (do uruchomień wielokrotnych).")
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
