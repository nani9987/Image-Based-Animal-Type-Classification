from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import cv2
import base64
import os

from gradcam import generate_gradcam  # corrected import (file is gradcam.py at repo root)

app = Flask(__name__)
CORS(app)

# Load trained model robustly
MODEL_PATH = "goat_sheep_model.pth"

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Please place your model checkpoint there.")
    checkpoint = torch.load(path, map_location="cpu")
    # If checkpoint is a dict with 'state_dict', user must provide model architecture
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raise RuntimeError("Checkpoint contains 'state_dict' only. Please load using the model architecture and call load_state_dict().")
    return checkpoint

model = load_model(MODEL_PATH)
model.eval()

# transform - keep consistent with image resizing below
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # add normalization here if your model was trained with it
])

classes = ["Goat", "Sheep"]

# Resolve frontend directory reliably relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    image = Image.open(file.stream).convert("RGB")

    # Use the resized image for both display and model input to avoid inconsistency
    img_resized = image.resize((224, 224))

    image_tensor = transform(img_resized).unsqueeze(0)  # shape (1, C, H, W)

    # prediction (no gradients)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, 1).item()
        confidence = float(torch.max(probs))

    # GradCAM (this needs gradients, so run outside no_grad)
    try:
        cam = generate_gradcam(model, image_tensor)  # returns HxW float array normalized to [0,1]
    except Exception as e:
        # Return prediction but include an informative message about Grad-CAM failure
        return jsonify({
            "prediction": classes[predicted],
            "confidence": round(confidence * 100, 2),
            "heatmap": None,
            "accuracy": 95.8,
            "gradcam_error": str(e)
        })

    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # convert heatmap from BGR (OpenCV) to RGB for overlay to match PIL->numpy ordering
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original = np.array(img_resized).astype(np.uint8)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", overlay)
    heatmap_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return jsonify({
        "prediction": classes[predicted],
        "confidence": round(confidence * 100, 2),
        "heatmap": heatmap_base64,
        "accuracy": 95.8
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
