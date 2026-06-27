"""
models/model2_transfer.py
=========================
PODEJŚCIE 2: Transfer learning (uczenie transferowe).

Wykorzystuje sieć wstępnie wytrenowaną na ImageNet (MobileNetV2, ResNet18/50
lub EfficientNet-B0) jako ekstraktor cech, z wymienioną głowicą klasyfikującą
dostosowaną do 7 klas emocji. Obsługuje dwustopniowy fine-tuning:

  Etap 1 (feature extraction): zamrożony backbone, trenowana tylko głowica.
  Etap 2 (fine-tuning):        odmrożone górne warstwy, niższy learning rate.

Charakterystyka dla pracy magisterskiej:
  + zwykle najwyższa dokładność spośród porównywanych podejść,
  + szybka zbieżność dzięki przeniesionym cechom wizualnym,
  - więcej parametrów, wymaga upscalingu 48->96 i konwersji do RGB,
  - ryzyko niedopasowania domeny (ImageNet =/= twarze w skali szarości).

Wejście: obraz 3x96x96 (RGB, normalizacja ImageNet).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

import config

# Dostępne warianty backbone'u (łatwa zamiana w badaniach ablacyjnych)
SUPPORTED_BACKBONES = ("mobilenet_v2", "resnet18", "resnet50", "efficientnet_b0")


class TransferModel(nn.Module):
    """
    Model transfer learning z wymienną siecią bazową.

    Args:
        backbone: nazwa sieci bazowej (patrz SUPPORTED_BACKBONES).
        num_classes: liczba klas.
        pretrained: czy użyć wag ImageNet.
        dropout: dropout w głowicy klasyfikującej.
    """

    def __init__(
        self,
        backbone: str = "mobilenet_v2",
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Nieobsługiwany backbone: {backbone}")
        self.backbone_name = backbone
        self.features, num_feats = self._build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_feats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _build_backbone(name: str, pretrained: bool):
        """Tworzy ekstraktor cech i zwraca liczbę cech wyjściowych."""
        weights = "DEFAULT" if pretrained else None
        if name == "mobilenet_v2":
            net = models.mobilenet_v2(weights=weights)
            return net.features, net.last_channel  # 1280
        if name == "resnet18":
            net = models.resnet18(weights=weights)
            feats = nn.Sequential(*list(net.children())[:-2])
            return feats, 512
        if name == "resnet50":
            net = models.resnet50(weights=weights)
            feats = nn.Sequential(*list(net.children())[:-2])
            return feats, 2048
        if name == "efficientnet_b0":
            net = models.efficientnet_b0(weights=weights)
            return net.features, 1280
        raise ValueError(name)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    # ----- sterowanie zamrażaniem warstw (dwustopniowy fine-tuning) ----- #
    def freeze_backbone(self) -> None:
        """Etap 1: zamraża wszystkie warstwy ekstraktora cech."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last(self, n_blocks: int = 2) -> None:
        """
        Etap 2: odmraża ostatnie n bloków backbone'u do fine-tuningu.
        Warstwy wcześniejsze (cechy ogólne: krawędzie, tekstury) zostają zamrożone.
        """
        children = list(self.features.children())
        for block in children[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True


def build_model(backbone: str = "mobilenet_v2", pretrained: bool = True, **kwargs) -> nn.Module:
    """Fabryka modelu — jednolity interfejs."""
    return TransferModel(backbone=backbone, pretrained=pretrained, **kwargs)


MODEL_META = {
    "name": "Uczenie transferowe (MobileNetV2)",
    "model_type": "transfer",
    "sample_shape": (3, config.IMG_SIZE_TRANSFER, config.IMG_SIZE_TRANSFER),
    "two_stage_finetune": True,   # sygnał dla pipeline: użyj treningu dwuetapowego
}
