"""
utils/run_logger.py
===================
Automatyczna dokumentacja eksperymentów.

Każde uruchomienie treningu samo zapisuje komplet informacji potrzebnych do
opisania badań w pracy magisterskiej — bez ręcznego notowania:

  - pełny log konsoli (wszystko, co pojawia się na ekranie),
  - metadane środowiska (GPU/CPU, wersje bibliotek, system),
  - użyte hiperparametry i argumenty wywołania,
  - czas treningu i ewaluacji,
  - raport per-klasa (precision / recall / F1),
  - dla landmarków: wykrywalność twarzy,
  - zbiorczy rejestr WSZYSTKICH uruchomień (experiments_log.csv),
    do którego każdy eksperyment dopisuje swój wiersz.

Pliki trafiają do katalogu results/ (oraz results/logs/).
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

import config


class Tee:
    """Dubluje strumień wyjścia: pisze jednocześnie na ekran i do pliku logu."""

    def __init__(self, stream, file_handle):
        self.stream = stream
        self.file = file_handle

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def _collect_environment() -> dict:
    """Zbiera metadane środowiska obliczeniowego (do rozdziału metodologicznego)."""
    info = {
        "data_uruchomienia": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": platform.platform(),
        "python": platform.python_version(),
        "procesor": platform.processor() or "n/d",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_dostepne"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_pamiec_GB"] = round(props.total_memory / (1024 ** 3), 1)
        else:
            info["gpu"] = "brak (trening na CPU)"
    except Exception:
        info["torch"] = "n/d"
    for lib in ("torchvision", "sklearn", "xgboost", "mediapipe", "cv2", "numpy"):
        try:
            mod = __import__(lib)
            info[f"wersja_{lib}"] = getattr(mod, "__version__", "n/d")
        except Exception:
            pass
    return info


class RunLogger:
    """
    Kontekstowy rejestrator pojedynczego eksperymentu.

    Użycie:
        with RunLogger(run_name="cnn_main", args=args, cfg=cfg) as logger:
            ... trening i ewaluacja ...
            logger.log_result(eval_result, train_summary)

    Po wyjściu z bloku automatycznie zapisuje log konsoli, metadane i
    dopisuje wiersz do zbiorczego rejestru experiments_log.csv.
    """

    def __init__(self, run_name: str, args=None, cfg=None, extra_meta: dict | None = None):
        self.run_name = run_name
        self.args = args
        self.cfg = cfg
        self.extra_meta = extra_meta or {}
        self.logs_dir = os.path.join(config.RESULTS_DIR, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.log_path = os.path.join(self.logs_dir, f"{run_name}.log")
        self.meta_path = os.path.join(self.logs_dir, f"{run_name}_meta.json")
        self.registry_path = os.path.join(config.RESULTS_DIR, "experiments_log.csv")
        self._result = None
        self._train_summary = None
        self._t0 = None
        self._fh = None
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._fh = open(self.log_path, "w", encoding="utf-8")
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(self._old_stdout, self._fh)
        sys.stderr = Tee(self._old_stderr, self._fh)

        print("=" * 70)
        print(f"  EKSPERYMENT: {self.run_name}")
        print("=" * 70)
        self.environment = _collect_environment()
        print("Środowisko:")
        for k, v in self.environment.items():
            print(f"  {k}: {v}")
        if self.args is not None:
            print("\nArgumenty wywołania:")
            for k, v in vars(self.args).items():
                print(f"  --{k}: {v}")
        if self.cfg is not None:
            print("\nHiperparametry (TrainConfig):")
            for k, v in vars(self.cfg).items():
                print(f"  {k}: {v}")
        print("-" * 70)
        return self

    def log_result(self, eval_result, train_summary: dict | None = None):
        """Rejestruje wynik ewaluacji i (opcjonalnie) podsumowanie treningu."""
        self._result = eval_result
        self._train_summary = train_summary

        # Raport per-klasa zapisywany do osobnego, czytelnego pliku .txt
        report_path = os.path.join(self.logs_dir, f"{self.run_name}_raport_per_klasa.txt")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(f"Raport klasyfikacji per klasa — {eval_result.name}\n")
            fh.write("=" * 60 + "\n\n")
            fh.write(f"{'Emocja':<12}{'F1':>10}\n")
            for emo, f1 in eval_result.per_class_f1.items():
                fh.write(f"{emo:<12}{f1:>10.4f}\n")
            fh.write("\n")
            fh.write(f"Accuracy:      {eval_result.accuracy:.4f}\n")
            fh.write(f"F1 (macro):    {eval_result.f1_macro:.4f}\n")
            fh.write(f"F1 (weighted): {eval_result.f1_weighted:.4f}\n")
            fh.write(f"Inferencja:    {eval_result.inference_ms:.3f} ms/próbkę\n")
            fh.write(f"Parametry:     {eval_result.num_params:,}\n")
            fh.write(f"Rozmiar:       {eval_result.model_size_mb:.2f} MB\n")
            if eval_result.extra:
                fh.write("\nDodatkowe:\n")
                for k, v in eval_result.extra.items():
                    fh.write(f"  {k}: {v}\n")
        print(f"\n[dokumentacja] Raport per-klasa zapisany: {report_path}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._t0
        if exc_type is not None:
            print(f"\n[BŁĄD] Eksperyment przerwany: {exc_type.__name__}: {exc_val}")
        print(f"\nCałkowity czas eksperymentu: {elapsed:.1f} s ({elapsed/60:.1f} min)")

        # Zapis metadanych do JSON
        meta = {
            "run_name": self.run_name,
            "environment": self.environment,
            "args": vars(self.args) if self.args is not None else {},
            "config": vars(self.cfg) if self.cfg is not None else {},
            "czas_calkowity_s": round(elapsed, 1),
            "extra": self.extra_meta,
        }
        if self._train_summary:
            meta["train"] = {
                "best_f1_walidacja": round(self._train_summary.get("best_f1", 0), 4),
                "epoki_wykonane": self._train_summary.get("epochs_run"),
            }
        if self._result is not None:
            meta["wynik"] = self._result.summary_row()
            meta["per_class_f1"] = self._result.per_class_f1
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        # Dopisanie wiersza do zbiorczego rejestru wszystkich eksperymentów
        if self._result is not None:
            self._append_registry(elapsed)

        # Przywrócenie standardowych strumieni
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        if self._fh:
            self._fh.close()
        print(f"[dokumentacja] Log: {self.log_path}")
        print(f"[dokumentacja] Metadane: {self.meta_path}")
        if self._result is not None:
            print(f"[dokumentacja] Rejestr zbiorczy: {self.registry_path}")
        return False  # nie tłumimy ewentualnych wyjątków

    def _append_registry(self, elapsed: float):
        """Dopisuje jeden wiersz do experiments_log.csv (tworzy nagłówek, jeśli brak)."""
        row = {
            "run_name": self.run_name,
            "data": self.environment.get("data_uruchomienia", ""),
            "model": self._result.name,
            "accuracy": round(self._result.accuracy, 4),
            "f1_macro": round(self._result.f1_macro, 4),
            "f1_weighted": round(self._result.f1_weighted, 4),
            "inferencja_ms": round(self._result.inference_ms, 3),
            "parametry": self._result.num_params,
            "rozmiar_MB": round(self._result.model_size_mb, 2),
            "epoki": (self._train_summary or {}).get("epochs_run", ""),
            "seed": getattr(self.args, "seed", ""),
            "czas_treningu_s": round(elapsed, 1),
            "gpu": self.environment.get("gpu", ""),
            "detection_rate_test": self._result.extra.get("detection_rate_test", ""),
        }
        write_header = not os.path.exists(self.registry_path)
        with open(self.registry_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
