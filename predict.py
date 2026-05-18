import os
import torch
from torchvision import transforms, models
from PIL import Image

# ==========================================
# CONFIGURACION
# ==========================================
MODEL_PATH = "/home/neri/escuela/proyectoma/modelo_final_comida.pth"
IMAGE_PATH = input("Ruta de la imagen a clasificar: ").strip()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Media y std del dataset
mean = [0.6288, 0.5124, 0.3893]
std  = [0.2264, 0.2395, 0.2512]

# ==========================================
# CARGAR MODELO
# ==========================================
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
            torch.nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASSES = checkpoint['classes']
NUM_CLASSES = checkpoint['num_classes']

model = FoodResNet50(num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"Modelo cargado exitosamente!")
print(f"Clases: {NUM_CLASSES}")
print(f"Mejor Val Acc: {checkpoint.get('val_acc', 'N/A')}")
print(f"Test Acc: {checkpoint.get('test_acc', 'N/A')}")

# ==========================================
# PREDICCION
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

if not os.path.exists(IMAGE_PATH):
    print(f"Error: No se encontro la imagen en {IMAGE_PATH}")
else:
    img = Image.open(IMAGE_PATH).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = probs.max(1)
        pred_idx = pred.item()
        confidence = conf.item()
    
    print("\n" + "="*40)
    print("RESULTADO DE LA PREDICCION")
    print("="*40)
    print(f"Clase predicha: {CLASSES[pred_idx]}")
    print(f"Confianza: {confidence*100:.2f}%")
    
    # Top 5 predicciones
    top5_conf, top5_idx = probs.topk(5, dim=1)
    print("\nTop 5 predicciones:")
    for i in range(5):
        idx = top5_idx[0][i].item()
        conf = top5_conf[0][i].item()
        print(f"  {i+1}. {CLASSES[idx]} - {conf*100:.2f}%")
