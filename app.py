from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model_factory import build_model, resolve_model_config

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


@st.cache_resource
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
        return "Urgent patching and lane safety review"
    if "crack" in label_l and severity == "high":
        return "Priority surface repair within 1 week"
    if severity == "medium":
        return "Schedule maintenance cycle and monitor weekly"
    return "Monitor condition and re-inspect in 30 days"


def main() -> None:
    st.set_page_config(page_title="Infrastructure Damage Assessment", page_icon="RD")
    st.title("Infrastructure Damage Assessment")
    st.write("Upload a road image to get a damage class, severity, and maintenance recommendation.")

    model_path = Path("artifacts_new_design/baseline_model.pt")
    labels_path = Path("artifacts_new_design/labels.json")

    if not model_path.exists() or not labels_path.exists():
        st.warning("Model artifacts not found. Train first with train_baseline.py")
        st.stop()

    model, class_names, tfms, resolved_model_name, resolved_image_size = load_model(model_path, labels_path)
    st.caption(f"Loaded {resolved_model_name} at {resolved_image_size}px")

    uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload an image to run prediction.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    x = tfms(image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        confidence = float(probs[idx].item())

    label = class_names[idx]
    severity = severity_from_confidence(confidence)
    action = recommended_action(label, severity)

    st.subheader("Prediction")
    st.write(f"Damage class: **{label}**")
    st.write(f"Confidence: **{confidence:.2%}**")
    st.write(f"Severity: **{severity}**")
    st.write(f"Recommended action: **{action}**")


if __name__ == "__main__":
    main()
