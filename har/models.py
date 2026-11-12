"""The backbone is torchvision; pooling modifications are implemented here."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from har.pooling import AttentionPool, AvgMaxPool, GeM


def build_model(variant, num_classes, pretrained=False):
    if variant not in {"baseline", "gem", "avgmax", "attention"}:
        raise ValueError(f"Unknown variant: {variant}")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    # Same classifier initialization in all equal-width variants, independent of
    # the random initialization consumed by an optional attention module.
    head = nn.Linear(1024 if variant == "avgmax" else 512, num_classes)
    if variant == "gem":
        model.avgpool = GeM()
    elif variant == "avgmax":
        model.avgpool = AvgMaxPool()
    elif variant == "attention":
        model.avgpool = AttentionPool()
    model.fc = head
    return model
