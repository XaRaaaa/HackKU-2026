from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torchvision import models

MODEL_NAMES: tuple[str, ...] = (
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_v2_s",
    "convnext_tiny",
)


def available_model_names() -> tuple[str, ...]:
    return MODEL_NAMES


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized not in MODEL_NAMES:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return normalized


def default_image_size(model_name: str) -> int:
    normalized = _normalize_model_name(model_name)
    if normalized in {"efficientnet_v2_s", "convnext_tiny"}:
        return 320
    return 224


def build_model(model_name: str, num_classes: int) -> nn.Module:
    normalized = _normalize_model_name(model_name)

    if normalized == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if normalized == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if normalized == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if normalized == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if normalized == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def classifier_feature_module(model_name: str, model: nn.Module) -> nn.Module:
    normalized = _normalize_model_name(model_name)

    if normalized in {"resnet18", "resnet50"}:
        return model.fc

    if normalized in {"efficientnet_b0", "efficientnet_v2_s"}:
        return model.classifier[1]

    if normalized == "convnext_tiny":
        return model.classifier[2]

    raise ValueError(f"Unsupported model_name: {model_name}")


def resolve_model_config(
    model_path: Path,
    model_name: str = "",
    image_size: int = 0,
    summary_filename: str = "train_summary.json",
) -> tuple[str, int]:
    summary_path = model_path.with_name(summary_filename)
    summary: dict[str, object] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    resolved_model_name = model_name.strip().lower() if model_name.strip() else ""
    if not resolved_model_name:
        resolved_model_name = str(summary.get("model_name", "resnet18"))

    resolved_image_size = int(image_size) if image_size else 0
    if resolved_image_size <= 0:
        summary_image_size = summary.get("image_size")
        if isinstance(summary_image_size, int) and summary_image_size > 0:
            resolved_image_size = summary_image_size
        else:
            resolved_image_size = default_image_size(resolved_model_name)

    return resolved_model_name, resolved_image_size
