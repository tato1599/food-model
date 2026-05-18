import io
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_final_comida.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.6288, 0.5124, 0.3893]
STD = [0.2264, 0.2395, 0.2512]


class FoodResNet50(torch.nn.Module):
    def __init__(self, num_classes=34):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.6),
            torch.nn.Linear(in_features, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASSES = checkpoint["classes"]
NUM_CLASSES = checkpoint["num_classes"]

model = FoodResNet50(num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se envio ninguna imagen."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Selecciona una imagen primero."}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "El archivo no es una imagen valida."}), 400

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = probs.max(1)

    pred_idx = pred.item()
    confidence = conf.item()

    top5_conf, top5_idx = probs.topk(5, dim=1)
    top5 = []
    for i in range(5):
        idx = top5_idx[0][i].item()
        top5.append(
            {
                "clase": CLASSES[idx],
                "confianza": round(top5_conf[0][i].item() * 100, 2),
            }
        )

    return jsonify(
        {
            "clase": CLASSES[pred_idx],
            "confianza": round(confidence * 100, 2),
            "top5": top5,
            "device": str(DEVICE),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
