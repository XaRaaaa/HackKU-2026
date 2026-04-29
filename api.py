import io
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import your existing functions from predict.py
from predict import load_model, severity_from_confidence, recommended_action

app = FastAPI(title="RoadScout ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup to prevent slow cold starts and OOM crashes
model_path = Path("artifacts_new_design/baseline_model.pt")
labels_path = Path("artifacts_new_design/labels.json")

model, class_names, tfms, resolved_model_name, resolved_image_size = load_model(
    model_path, labels_path
)

@app.post("/predict")
async def predict_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        x = tfms(img).unsqueeze(0)

        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            confidence = float(probs[idx].item())

        label = class_names[idx]
        severity = severity_from_confidence(confidence)
        action = recommended_action(label, severity)

        return {
            "model_name": resolved_model_name,
            "image_size": resolved_image_size,
            "prediction": label,
            "confidence": confidence,
            "severity": severity,
            "recommended_action": action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))