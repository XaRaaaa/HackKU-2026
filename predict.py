from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model_factory import build_model, resolve_model_config

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def load_model(
    model_path: Path,
    labels_path: Path,
    model_name: str = "",
    image_size: int = 0,
):
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    resolved_model_name, resolved_image_size = resolve_model_config(
        model_path,
        model_name=model_name,
        image_size=image_size,
    )

    model = build_model(resolved_model_name, len(class_names))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    tfms = transforms.Compose(
        [
            transforms.Resize((resolved_image_size + 32, resolved_image_size + 32)),
            transforms.CenterCrop(resolved_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )

    return model, class_names, tfms, resolved_model_name, resolved_image_size


def severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"


def recommended_action(label: str, severity: str) -> str:
    label_l = label.lower()
    if "pothole" in label_l and severity in {"high", "medium"}:
        return "urgent patching and lane safety review"
    if "crack" in label_l and severity == "high":
        return "priority surface repair within 1 week"
    if severity == "medium":
        return "schedule maintenance cycle and monitor weekly"
    return "monitor condition and re-inspect in 30 days"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict infrastructure damage class from image")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="artifacts_new_design/baseline_model.pt")
    parser.add_argument("--labels", type=str, default="artifacts_new_design/labels.json")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--image_size", type=int, default=0)
    args = parser.parse_args()

    model, class_names, tfms, resolved_model_name, resolved_image_size = load_model(
        Path(args.model),
        Path(args.labels),
        model_name=args.model_name,
        image_size=args.image_size,
    )

    image = Image.open(args.image).convert("RGB")
    x = tfms(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        confidence = float(probs[idx].item())

    label = class_names[idx]
    severity = severity_from_confidence(confidence)
    action = recommended_action(label, severity)

    print("model_name:", resolved_model_name)
    print("image_size:", resolved_image_size)
    print("prediction:", label)
    print("confidence:", round(confidence, 4))
    print("severity:", severity)
    print("recommended_action:", action)


if __name__ == "__main__":
    main()
