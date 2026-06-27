"""
models/model1_cnn.py
====================
PODEJŚCIE 1: Konwolucyjna sieć neuronowa trenowana od zera.

Architektura: mini-Xception — kompaktowa sieć z rozdzielnymi splotami
(depthwise separable convolutions) i połączeniami rezydualnymi. Inspirowana
pracą Arriaga i in. ("Real-time Convolutional Neural Networks for Emotion
and Gender Classification"), powszechnie używaną jako lekki baseline dla FER2013.

Charakterystyka dla pracy magisterskiej:
  + pełna kontrola nad architekturą, mało parametrów (~60 tys.),
  + szybka inferencja, działa w czasie rzeczywistym nawet na CPU,
  - ograniczona pojemność -> niższy sufit dokładności niż modele wstępnie trenowane.

Wejście: obraz 1x48x48 (skala szarości).
"""
from __future__ import annotations

import torch
import torch.nn as nn

import config


class SeparableConv2d(nn.Module):
    """Splot rozdzielny: depthwise + pointwise. Redukuje liczbę parametrów."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    """
    Blok rezydualny: dwa rozdzielne sploty + skrót (1x1 conv ze stride=2).
    Po bloku następuje redukcja przestrzenna (max-pool stride 2).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.residual = nn.Conv2d(in_ch, out_ch, 1, stride=2, padding=0, bias=False)
        self.residual_bn = nn.BatchNorm2d(out_ch)

        self.sep1 = SeparableConv2d(in_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.sep2 = SeparableConv2d(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        res = self.residual_bn(self.residual(x))
        out = self.act(self.bn1(self.sep1(x)))
        out = self.bn2(self.sep2(out))
        out = self.pool(out)
        return out + res


class MiniXception(nn.Module):
    """
    Kompaktowa sieć Xception dla FER2013.

    Args:
        num_classes: liczba klas emocji.
        in_channels: liczba kanałów wejściowych (1 dla skali szarości).
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, in_channels: int = 1):
        super().__init__()
        # Wstępne sploty
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        # Cztery bloki rezydualne ze wzrastającą liczbą kanałów
        self.block1 = XceptionBlock(8, 16)
        self.block2 = XceptionBlock(16, 32)
        self.block3 = XceptionBlock(32, 64)
        self.block4 = XceptionBlock(64, 128)

        # Klasyfikator: splot 1x1 -> globalne uśrednianie -> softmax (przez CE)
        self.head = nn.Sequential(
            nn.Conv2d(128, num_classes, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)


def build_model(**kwargs) -> nn.Module:
    """Fabryka modelu — jednolity interfejs dla wszystkich podejść."""
    return MiniXception(**kwargs)


# Metadane wykorzystywane przez pipeline treningowy i porównawczy
MODEL_META = {
    "name": "CNN-od-zera (mini-Xception)",
    "model_type": "cnn",
    "sample_shape": (1, config.IMG_SIZE_NATIVE, config.IMG_SIZE_NATIVE),
}
