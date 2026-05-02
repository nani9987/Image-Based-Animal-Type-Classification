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

from utils.gradcam import generate_gradcam

app = Flask(__name__)
CORS(app)

# Load trained model
model = torch.load("goat_sheep_model.pth", map_location="cpu", weights_only=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes = ["Goat", "Sheep"]

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("../frontend", path)

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({"error":"No image uploaded"})

    file = request.files['image']

    image = Image.open(file.stream).convert("RGB")

    img_resized = image.resize((224,224))

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image_tensor)

        probs = F.softmax(outputs, dim=1)

        predicted = torch.argmax(probs,1).item()

        confidence = float(torch.max(probs))

    # GradCAM
    cam = generate_gradcam(model, image_tensor)

    cam = cv2.resize(cam,(224,224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    original = np.array(img_resized)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", overlay)

    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({

        "prediction": classes[predicted],

        "confidence": round(confidence * 100,2),

        "heatmap": heatmap_base64,

        "accuracy": 95.8

    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)