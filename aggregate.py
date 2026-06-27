"""
aggregate.py
============
Agregacja wyników z wielu uruchomień (np. różne ziarna losowości).

Czyta zbiorczy rejestr `results/experiments_log.csv` (uzupełniany automatycznie
przez każdy trening) i wytwarza tabelę zbiorczą w formie **średnia ± odchylenie
standardowe** per model — gotową do wstawienia do rozdziału wynikowego pracy
(wzmacnia istotność statystyczną porównania).

Generuje:
  - `results/summary_mean_std.csv` — tabela średnia±odchylenie,
  - wydruk konsolowy w formacie czytelnym dla człowieka.

Przykłady:
    python aggregate.py
    python aggregate.py --metric accuracy f1_macro inferencja_ms
    python aggregate.py --filter-tag seed     # tylko uruchomienia z tagiem zawierającym 'seed'
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config


def aggregate(metrics: list[str], filter_tag: str | None = None):
    path = os.path.join(config.RESULTS_DIR, "experiments_log.csv")
    if not os.path.exists(path):
        print(f"Brak rejestru: {path}. Uruchom najpierw treningi (train.py / compare.py).")
        return

    df = pd.read_csv(path)
    if filter_tag:
        df = df[df["run_name"].str.contains(filter_tag, na=False)]
        if df.empty:
            print(f"Brak uruchomień pasujących do '{filter_tag}'.")
            return

    # Grupujemy po nazwie modelu (uśredniamy po ziarnach / powtórzeniach)
    available = [m for m in metrics if m in df.columns]
    rows = []
    for model_name, grp in df.groupby("model"):
        row = {"Model": model_name, "Liczba uruchomień": len(grp)}
        for m in available:
            vals = pd.to_numeric(grp[m], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                continue
            mean, std = float(np.mean(vals)), float(np.std(vals, ddof=0))
            row[f"{m} (śr.)"] = round(mean, 4)
            row[f"{m} (odch.)"] = round(std, 4)
            row[f"{m}"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)

    summary = pd.DataFrame(rows)
    # Sortujemy po średniej F1-macro, jeśli dostępne
    sort_key = "f1_macro (śr.)"
    if sort_key in summary.columns:
        summary = summary.sort_values(sort_key, ascending=False).reset_index(drop=True)

    out_path = os.path.join(config.RESULTS_DIR, "summary_mean_std.csv")
    summary.to_csv(out_path, index=False)

    # Czytelny wydruk: tylko kolumny "metryka = śr ± odch"
    print("=" * 70)
    print("  ZBIORCZE WYNIKI: średnia ± odchylenie standardowe (po uruchomieniach)")
    print("=" * 70)
    display_cols = ["Model", "Liczba uruchomień"] + [m for m in available if m in summary.columns]
    print(summary[display_cols].to_string(index=False))
    print(f"\nPełna tabela (ze średnimi i odchyleniami osobno): {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Agregacja wyników wielu uruchomień.")
    parser.add_argument("--metric", nargs="+",
                        default=["accuracy", "f1_macro", "f1_weighted", "inferencja_ms"],
                        help="Metryki do zagregowania (nazwy kolumn z experiments_log.csv).")
    parser.add_argument("--filter-tag", default=None,
                        help="Uwzględnij tylko uruchomienia, których nazwa zawiera ten ciąg.")
    args = parser.parse_args()
    aggregate(args.metric, args.filter_tag)


if __name__ == "__main__":
    main()
