"""Architectures — baseline CNN maison + modèle dual-branch T/B.

Voir la justification des choix dans `reports/choix_justifies.md` §4-5.
"""

import os

# macOS : évite l'échec SSL au téléchargement des poids pré-entraînés torchvision.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import torch
import torch.nn as nn
from torchvision import models as tvm

NUM_CLASSES = 4


class SimpleCNN(nn.Module):
    """Baseline exigée par le sujet : CNN compact codé à la main (image unique).

    4 blocs conv(3x3)-BN-ReLU-MaxPool -> global average pooling -> classifieur.
    ~0.5 M paramètres : volontairement simple, sert de référence basse.
    """

    def __init__(self, num_classes=NUM_CLASSES, in_ch=3):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, 32), block(32, 64), block(64, 128), block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def _backbone(name="mobilenetv3_small", pretrained=True):
    """MobileNetV3 (small ou large) sans sa tête -> extracteur de features."""
    if name == "mobilenetv3_large":
        w = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.mobilenet_v3_large(weights=w)
    else:
        w = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.mobilenet_v3_small(weights=w)
    feat_dim = m.classifier[0].in_features  # 576 (small) / 960 (large)
    m.classifier = nn.Identity()
    return m, feat_dim


class DualBranchNet(nn.Module):
    """Deux vues (T, B) -> 1 label. Backbones à poids partagés (siamois),
    fusion par concaténation, puis tête MLP.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, shared=True,
                 fusion="concat", backbone="mobilenetv3_small"):
        super().__init__()
        self.enc_t, feat = _backbone(backbone, pretrained)
        self.enc_b = self.enc_t if shared else _backbone(backbone, pretrained)[0]
        self.fusion = fusion
        fused = feat * 2 if fusion == "concat" else feat
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(fused, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, xt, xb):
        ft, fb = self.enc_t(xt), self.enc_b(xb)
        z = torch.cat([ft, fb], 1) if self.fusion == "concat" else (ft + fb)
        return self.head(z)


def build_model(name, pretrained=True, backbone="mobilenetv3_small"):
    if name == "simplecnn":
        return SimpleCNN()
    if name == "dualbranch":
        return DualBranchNet(pretrained=pretrained, backbone=backbone)
    raise ValueError(f"modèle inconnu : {name}")
