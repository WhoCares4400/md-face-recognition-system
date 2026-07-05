# System widzenia maszynowego dedykowany do rozpoznawania emocji człowieka

Porównawcze studium **czterech podejść sztucznej inteligencji** do rozpoznawania
emocji z mimiki twarzy, ocenionych w identycznych, kontrolowanych warunkach na
zbiorze **FER2013**. Repozytorium zawiera kod, potok eksperymentalny, wyniki
(wykresy, metryki), a także **dołączony zbiór danych** (`fer2013/`) oraz
**wytrenowane wagi modeli** (`checkpoints/`, przez Git LFS) — dzięki czemu program
można uruchomić od razu po sklonowaniu, bez ponownego treningu.

Wszystkie modele trenowane i testowane są na tym samym podziale danych i tymi samymi
metrykami, więc różnice w wynikach są przypisywalne wyłącznie podejściu algorytmicznemu.

## Porównywane podejścia

| # | Podejście | Opis |
|---|-----------|------|
| 1 | **CNN od zera (mini-Xception)** | Lekka sieć splotowa (~57 tys. parametrów) ze splotami rozdzielnymi i połączeniami rezydualnymi, trenowana od zera. |
| 2 | **Uczenie transferowe** | Sieci pretrenowane na ImageNet (MobileNetV2, EfficientNet-B0, ResNet18, ResNet50), dostrajane dwuetapowo. |
| 3 | **Cechy geometryczne + SVM** | 468 punktów twarzy (MediaPipe FaceMesh) → cechy geometryczne → klasyfikator SVM. W pełni interpretowalne. |
| 4 | **Vision Transformer** | ViT-Lite (od zera, 48×48) oraz ViT-B/16 (pretrenowany na ImageNet, 224×224). |

## Wyniki (zbiór testowy FER2013, ziarno = 42)

| Model | Dokładność | F1-makro | Parametry | Inferencja [ms] |
|-------|:----------:|:--------:|:---------:|:---------------:|
| ViT-B/16 (ImageNet) | 0,6962 | **0,6929** | 85 804 039 | 21,20 |
| Uczenie transferowe (ResNet18) | 0,6790 | 0,6702 | 11 152 647 | 2,83 |
| Uczenie transferowe (ResNet50) | 0,6779 | 0,6632 | 23 809 543 | 6,76 |
| Uczenie transferowe (EfficientNet-B0) | 0,6064 | 0,5773 | 3 485 987 | 8,17 |
| CNN od zera (mini-Xception) | 0,5727 | 0,5236 | **56 951** | **1,46** |
| Cechy geometryczne + SVM | 0,5653 | 0,5082 | — | 1,57 |
| Uczenie transferowe (MobileNetV2) | 0,5039 | 0,4536 | 1 536 327 | 5,04 |
| Transformer wizyjny (ViT-Lite) | 0,3794 | 0,3101 | 1 006 599 | 2,57 |

Metryką nadrzędną jest **F1-makro** (odporna na silne niezbalansowanie klas). Najlepszy
ogólnie jest pretrenowany ViT-B/16; najlepszy kompromis jakość–koszt oferuje
mini-Xception. Pełne metryki, macierze pomyłek i krzywe uczenia znajdują się w katalogu
`results/`.

## Zbiór danych FER2013

48×48 px, skala szarości, 7 klas emocji (złość, wstręt, strach, radość, smutek,
zaskoczenie, neutralny), ~28,7 tys. obrazów treningowych i 7178 testowych. Zbiór jest
silnie niezbalansowany (radość 8989 vs wstręt 547 obrazów łącznie).

Zbiór jest **dołączony do repozytorium** w katalogu `fer2013/` o następującej
strukturze (źródło: Kaggle, <https://www.kaggle.com/datasets/msambare/fer2013>):

```
fer2013/
├── train/
│   ├── angry/  disgust/  fear/  happy/  sad/  surprise/  neutral/
└── test/
    ├── angry/  disgust/  fear/  happy/  sad/  surprise/  neutral/
```

Alternatywnie obsługiwany jest pojedynczy plik `fer2013.csv` (kolumny `emotion`,
`pixels`, `Usage`).

## Instalacja

Repozytorium używa **Git LFS** do przechowywania wag modeli (`checkpoints/`).
Zainstaluj Git LFS przed klonowaniem, aby pobrały się pełne pliki wag:

```bash
git lfs install
git clone https://github.com/WhoCares4400/md-face-recognition-system.git
cd md-face-recognition-system
```

Następnie utwórz środowisko i zainstaluj zależności:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

> **GPU (zalecane do treningu):** zainstaluj `torch` zgodnie z instrukcją z
> <https://pytorch.org> (dobór wersji CUDA). Pozostałe pakiety bez zmian. Trening
> przeprowadzono na NVIDIA RTX 3070 Laptop (8 GB), PyTorch 2.6.0 / CUDA 12.4.

## Użycie

### 1. Trening pojedynczego modelu — `train.py`

```bash
# CNN od zera (mini-Xception)
python train.py --model cnn --data ./fer2013 --epochs 60 --tag main

# Uczenie transferowe (wybór sieci bazowej)
python train.py --model transfer --data ./fer2013 --backbone resnet18 --tag resnet18
python train.py --model transfer --data ./fer2013 --backbone efficientnet_b0 --tag effb0

# Vision Transformer
python train.py --model vit --data ./fer2013 --variant lite           --tag main
python train.py --model vit --data ./fer2013 --variant pretrained      --tag vitb16

# Cechy geometryczne + SVM (lub --classifier rf / xgb — badanie ablacyjne)
python train.py --model landmarks --data ./fer2013 --classifier svm --tag main
```

> Modele drzewiaste (`rf`, `xgb`) dodatkowo zapisują wykres ważności cech do `results/`.

Każdy trening zapisuje wagi do `checkpoints/`, metryki i wykresy do `results/` oraz
log z przebiegiem per-epoka do `results/logs/`.

### 2. Badania porównawcze — `compare.py`

Trenuje (lub wczytuje) wszystkie modele i ewaluuje je w identycznych warunkach,
generując tabelę zbiorczą oraz wykresy porównawcze:

```bash
python compare.py --data ./fer2013 --models cnn transfer vit landmarks --tag main
```

### 3. Powtarzalność i agregacja — `aggregate.py`

Uruchom trening na kilku ziarnach (np. `--seed 1`, `--seed 2`, `--tag seed1` …),
a następnie zagreguj wyniki do tabeli średnia ± odchylenie:

```bash
python train.py --model cnn --data ./fer2013 --seed 1 --tag seed1
python train.py --model cnn --data ./fer2013 --seed 2 --tag seed2
python aggregate.py                 # zbiorcza tabela z results/experiments_log.csv
python aggregate.py --filter-tag seed   # tylko uruchomienia z 'seed' w nazwie
```

### 4. Regeneracja wykresów — `regen_plots.py`

Odtwarza macierze pomyłek, wykresy porównawcze i tabelę z zapisanych plików JSON
(bez ponownego treningu):

```bash
python regen_plots.py --tag main
```

### 5. Rozpoznawanie na żywo z kamery — `realtime.py`

```bash
# Najprościej (auto-dobór checkpoints/<model>_main.*)
python realtime.py --model cnn

# Wskazanie konkretnych wag i wariantu
python realtime.py --model transfer --backbone resnet18 --checkpoint checkpoints/transfer_resnet18.pt
python realtime.py --model vit --variant pretrained --checkpoint checkpoints/vit_vitb16.pt
python realtime.py --model landmarks --classifier svm

# Opcje jakości / wydajności
python realtime.py --model cnn --equalize --detect-every 3 --no-mirror
```

Aplikacja ma **boczny panel z klikalnymi przyciskami** (mysz) oraz skróty klawiszowe.
Model AI można zmieniać w trakcie działania, a klawiszem **`c`** uruchomić porównanie
wszystkich czterech modeli na tej samej klatce (montaż 2×2 zapisywany do pliku).

Sterowanie (klawisz lub kliknięcie przycisku w panelu):

| Klawisz | Działanie |
|:-------:|-----------|
| `1` / `2` / `3` / `4` | zmiana modelu: CNN ▸ ResNet18 ▸ ViT-B/16 ▸ Geometria+SVM |
| `c` | porównanie 4 modeli na jednej klatce (montaż 2×2, zapis do `compare_models_*.png`) |
| `e` | CLAHE (wyrównanie histogramu) wł/wył |
| `m` | lustrzane odbicie wł/wył |
| `+` / `-` | siła wygładzania predykcji (EMA) |
| `h` | pokaż/ukryj panel sterowania |
| `s` | zrzut ekranu (`snapshot_*.png`) |
| `q` | wyjście |

Najważniejsze opcje wiersza poleceń:

| Opcja | Domyślnie | Opis |
|-------|:---------:|------|
| `--model` | `cnn` | Model początkowy (zmienny w trakcie klawiszami 1–4). |
| `--checkpoint` | `checkpoints/<model>_main.*` | Ścieżka do wag/modelu. |
| `--width` | `960` | Szerokość okna podglądu w px (`0` = bez skalowania). |
| `--margin` | `0.25` | Margines kontekstowy wokół twarzy (kwadratowe wycięcie). |
| `--smooth` | `0.5` | Wygładzanie czasowe predykcji (EMA); `1.0` = wyłączone. |
| `--equalize` | wył. | Wyrównanie histogramu (CLAHE) — odporność na oświetlenie. |
| `--detect-every` | `2` | Detekcja twarzy co N klatek (wyższy FPS). |
| `--mirror` / `--no-mirror` | wł. | Lustrzane odbicie obrazu kamery. |
| `--camera` | `0` | Indeks kamery. |

## Demo webowe na żywo (GitHub Pages)

Interaktywne demo działające **w przeglądarce** (również na telefonie) — bez instalacji,
bez serwera. Modele uruchamiane są po stronie klienta przez **ONNX Runtime Web** (WASM),
a twarz wykrywa **MediaPipe**. Obraz nie opuszcza urządzenia. Kod strony znajduje się
w katalogu [`docs/`](docs/).

**Adres (po włączeniu Pages):** `https://whocares4400.github.io/md-face-recognition-system/`

### Jak to działa

GitHub Pages serwuje wyłącznie pliki statyczne, więc checkpointy PyTorch są najpierw
eksportowane do formatu **ONNX**:

```bash
pip install -r requirements-export.txt
python export_onnx.py          # zapisuje docs/models/*.onnx + docs/models/models.json
```

Skrypt eksportuje 6 sieci (CNN, ViT-Lite, MobileNetV2, EfficientNet-B0, ResNet18, ResNet50)
oraz 2 modele landmarkowe (SVM, XGBoost), waliduje zgodność ONNX vs PyTorch i generuje
manifest `models.json` (steruje listą modeli na stronie).

> **Świadomie pominięte w wersji webowej:** **ViT-B/16** (~343 MB) i **Landmarki + RandomForest**
> (~430 MB) — ich pobieranie w przeglądarce trwałoby zbyt długo (informacja jest też widoczna
> na samej stronie). Oba modele pozostają dostępne w `checkpoints/` do użytku lokalnego
> (`realtime.py`).

### Test lokalny

```bash
cd docs
python -m http.server 8000
# otwórz http://localhost:8000
```

### Publikacja na GitHub Pages

1. Zatwierdź i wypchnij katalog `docs/` (pliki `*.onnx` są **poza Git LFS**, bo Pages nie
   serwuje treści LFS — commitowane są bezpośrednio).
2. W repozytorium: **Settings → Pages → Build and deployment → Deploy from a branch**,
   wybierz gałąź `main` i katalog `/docs`, zapisz.
3. Po chwili strona będzie dostępna pod adresem podanym wyżej.

## Struktura repozytorium

```
.
├── config.py            # wspólna konfiguracja (klasy, ścieżki, hiperparametry, ziarno)
├── export_onnx.py       # eksport wag -> ONNX dla dema webowego (docs/)
├── docs/                # demo webowe (GitHub Pages): index.html, app.js, style.css, models/
├── train.py             # trening pojedynczego modelu (cnn / transfer / vit / landmarks)
├── compare.py           # badania porównawcze wszystkich podejść
├── aggregate.py         # agregacja wielu uruchomień (średnia ± odchylenie)
├── regen_plots.py       # regeneracja wykresów z plików JSON
├── regen_feat_imp.py    # regeneracja wykresów ważności cech (modele drzewiaste)
├── realtime.py          # rozpoznawanie emocji na żywo z kamery
├── models/              # 4 architektury: model1_cnn, model2_transfer, model3_landmarks, model4_vit
├── utils/               # dane, pętla treningowa, metryki, wykresy, logger
├── results/             # metryki (JSON/CSV), macierze pomyłek, krzywe, wykresy porównawcze
├── checkpoints/         # wytrenowane wagi modeli (Git LFS)
├── fer2013/             # zbiór danych FER2013 (train/ + test/)
├── requirements.txt
└── README.md
```

> **Uwaga:** wagi modeli (`checkpoints/`) są przechowywane przez **Git LFS** —
> do ich pobrania wymagany jest zainstalowany Git LFS (`git lfs install`) przed
> klonowaniem. Zbiór `fer2013/` jest dołączony bezpośrednio.

## Środowisko

PyTorch 2.6.0 (CUDA 12.4), Python 3.12, scikit-learn, XGBoost, MediaPipe, OpenCV.
Pełna lista zależności w `requirements.txt`.

## Uwagi

Kod powstał przy wsparciu narzędzia **Claude Code** (Anthropic).
