"""
models/model4_vit.py
====================
PODEJŚCIE 4: Vision Transformer (ViT).

Architektura oparta na mechanizmie samo-uwagi (self-attention) zamiast splotów.
Obraz dzielony jest na łatki (patche), które — po liniowym rzutowaniu i dodaniu
kodowania pozycyjnego — przetwarzane są przez stos bloków transformera.

Moduł udostępnia dwa warianty (do badań porównawczych):

  A) ViT-Lite — kompaktowy transformer trenowany OD ZERA na obrazach 48x48.
     Mały rozmiar patcha (6x6) i płytki stos. Pozwala porównać "czysty"
     transformer z CNN przy zbliżonej liczbie parametrów.

  B) ViT-B/16 — duży transformer WSTĘPNIE TRENOWANY na ImageNet (z torchvision),
     fine-tunowany na FER2013 przy obrazach 224x224. Reprezentuje podejście
     "transformer + transfer learning".

Charakterystyka dla pracy magisterskiej:
  + globalny kontekst od pierwszej warstwy (uwaga między wszystkimi łatkami),
  + ViT-B/16 osiąga wysoką dokładność,
  - transformer od zera jest "głodny danych" — FER2013 bywa za mały,
  - wyższy koszt obliczeniowy, wolniejsza inferencja niż lekka CNN.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import config


# --------------------------------------------------------------------------- #
# Wariant A: lekki ViT trenowany od zera (dla obrazów 48x48, skala szarości)
# --------------------------------------------------------------------------- #
class PatchEmbedding(nn.Module):
    """Dzieli obraz na łatki i rzutuje każdą do wektora o wymiarze embed_dim."""

    def __init__(self, img_size: int, patch_size: int, in_ch: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size musi być podzielne przez patch_size"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """Standardowy blok transformera: MHSA + MLP z normalizacją pre-LN."""

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTLite(nn.Module):
    """
    Kompaktowy Vision Transformer trenowany od zera.

    Domyślne hiperparametry dobrane tak, by liczba parametrów była umiarkowana
    i porównywalna z innymi podejściami (~1-2 mln).
    """

    def __init__(
        self,
        img_size: int = config.IMG_SIZE_NATIVE,
        patch_size: int = 6,
        in_ch: int = 1,
        num_classes: int = config.NUM_CLASSES,
        embed_dim: int = 128,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 3.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, embed_dim)
        n_patches = self.patch_embed.n_patches

        # token klasyfikacyjny [CLS] + kodowanie pozycyjne (uczone)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                   # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                        # klasyfikacja z tokenu [CLS]


# --------------------------------------------------------------------------- #
# Wariant B: ViT-B/16 wstępnie trenowany (transfer learning na transformerze)
# --------------------------------------------------------------------------- #
class ViTPretrained(nn.Module):
    """Opakowanie na torchvision vit_b_16 z wymienioną głowicą (7 klas)."""

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        from torchvision import models
        weights = "DEFAULT" if pretrained else None
        self.vit = models.vit_b_16(weights=weights)
        in_feats = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.vit(x)

    def freeze_backbone(self):
        for name, p in self.vit.named_parameters():
            if "heads" not in name:
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.vit.parameters():
            p.requires_grad = True


def build_model(variant: str = "lite", pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Fabryka modelu — jednolity interfejs.

    Args:
        variant: 'lite' (ViT od zera, 48x48) lub 'pretrained' (ViT-B/16, 224x224).
    """
    if variant == "lite":
        return ViTLite(**kwargs)
    if variant == "pretrained":
        return ViTPretrained(pretrained=pretrained, **kwargs)
    raise ValueError(f"Nieznany wariant ViT: {variant}")


# Domyślnie raportujemy wariant lekki (porównywalny rozmiarowo z CNN).
# Aby porównać wariant wstępnie trenowany, ustaw variant='pretrained'
# i model_type='vit_pretrained' przy budowie dataloaderów.
MODEL_META = {
    "name": "Transformer wizyjny (ViT-Lite)",
    "model_type": "vit",
    "sample_shape": (1, config.IMG_SIZE_NATIVE, config.IMG_SIZE_NATIVE),
    "variants": {
        "lite": {"model_type": "vit", "sample_shape": (1, 48, 48)},
        "pretrained": {"model_type": "vit_pretrained", "sample_shape": (3, 224, 224)},
    },
}
