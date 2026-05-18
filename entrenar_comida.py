import os
import time
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ==========================================
# CONFIGURACION
# ==========================================
INPUT_ROOT = "/home/neri/escuela/proyectoma/Images"
OUTPUT_ROOT = "/home/neri/escuela/proyectoma/Images_128"
MODEL_PATH = "/home/neri/escuela/proyectoma/modelo_comida.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# REDIMENSIONAMIENTO DE IMAGENES
# ==========================================
def process_image(img):
    try:
        if img.mode == 'P':
            img = img.convert('RGBA')

        if img.mode in ('RGBA', 'LA'):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        else:
            img = img.convert("RGB")

        return img
    except Exception as e:
        print(f"Error en conversion de imagen: {e}")
        return None

def resize_images(input_dir, output_dir, size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(input_dir), desc=f"Redimensionando {os.path.basename(input_dir)}"):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        # Saltar si ya existe
        if os.path.exists(output_path):
            continue

        try:
            with Image.open(input_path) as img:
                img = process_image(img)
                if img is None:
                    continue
                img = ImageOps.fit(img, size, Image.LANCZOS)
                img.save(output_path, "JPEG", quality=95)
        except Exception as e:
            print(f"Error con {input_path}: {e}")

print("\n--- Verificando redimensionamiento ---")
if not os.path.exists(OUTPUT_ROOT) or len(os.listdir(OUTPUT_ROOT)) < 30:
    print("Creando imagenes 128x128...")
    start_resize = time.time()
    for class_name in os.listdir(INPUT_ROOT):
        input_class_path = os.path.join(INPUT_ROOT, class_name)
        output_class_path = os.path.join(OUTPUT_ROOT, class_name)
        if os.path.isdir(input_class_path):
            resize_images(input_class_path, output_class_path)
    print(f"Redimensionamiento completado en {time.time() - start_resize:.2f}s")
else:
    print("Imagenes 128x128 ya existen, saltando...")

# ==========================================
# MEDIA Y STD (precalculadas del notebook)
# ==========================================
mean = [0.6288, 0.5124, 0.3893]
std  = [0.2264, 0.2395, 0.2512]

# ==========================================
# DATASETS Y DATALOADERS
# ==========================================
print("\n--- Cargando datasets ---")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

dataset = datasets.ImageFolder(root=OUTPUT_ROOT, transform=transform)

NUM_CLASSES = len(dataset.classes)
print(f"Clases detectadas: {NUM_CLASSES}")
print(f"Nombres de clases: {dataset.classes}")

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=64, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=64, num_workers=4, pin_memory=True)

print(f"Total: {len(dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ==========================================
# MODELO
# ==========================================
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.AvgPool2d(4),  # 128 -> 32
            nn.Flatten(),
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# ENTRENAMIENTO
# ==========================================
def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

def train(model, train_loader, val_loader, epochs=10, lr=1e-3):
    model.to(DEVICE)  # <-- CORREGIDO: era modeel.to(dvice)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            outputs = model(x)
            loss = F.cross_entropy(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches if batches > 0 else 0
        val_acc = accuracy(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

# ==========================================
# EJECUCION PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("\n=== INICIANDO ENTRENAMIENTO ===")
    total_start = time.time()

    model = LinearModel()
    print(f"Parametros del modelo: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_loader, val_loader, epochs=5, lr=3e-4)

    test_acc = accuracy(model, test_loader)
    print(f"\nTest Acc: {test_acc:.4f}")

    total_time = time.time() - total_start
    print(f"\nTiempo total de ejecucion: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")

    # Guardar modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")
