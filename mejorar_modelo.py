import os
import time
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ==========================================
# CONFIGURACION
# ==========================================
INPUT_ROOT = "/home/neri/escuela/proyectoma/Images"
OUTPUT_ROOT = "/home/neri/escuela/proyectoma/Images_128"
MODEL_PATH = "/home/neri/escuela/proyectoma/mejor_modelo_comida.pth"

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
# DATASETS Y DATALOADERS (MEJOR AUGMENTATION)
# ==========================================
print("\n--- Cargando datasets ---")

# Media y std del dataset original
mean = [0.6288, 0.5124, 0.3893]
std  = [0.2264, 0.2395, 0.2512]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Dataset base sin transformacion
full_dataset = datasets.ImageFolder(root=OUTPUT_ROOT)
NUM_CLASSES = len(full_dataset.classes)
print(f"Clases detectadas: {NUM_CLASSES}")
print(f"Nombres de clases: {full_dataset.classes}")

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_indices, val_indices, test_indices = random_split(
    range(len(full_dataset)), [train_size, val_size, test_size]
)

# Subsets con diferentes transformaciones
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

train_ds = TransformSubset(full_dataset, train_indices.indices, train_transform)
val_ds   = TransformSubset(full_dataset, val_indices.indices, val_test_transform)
test_ds  = TransformSubset(full_dataset, test_indices.indices, val_test_transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=64, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=64, num_workers=4, pin_memory=True)

print(f"Total: {len(full_dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ==========================================
# MODELO: RESNET18 PRE-ENTRENADA (TRANSFER LEARNING)
# ==========================================
class FoodResNet(nn.Module):
    def __init__(self, num_classes=34):
        super().__init__()
        # Cargar ResNet18 pre-entrenada
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Congelar las primeras capas (opcional, pero ayuda a evitar overfitting)
        # Descongelamos las ultimas 2 capas para fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Descongelar layer4 y fc para fine-tuning
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        # Reemplazar la capa fully connected final
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# ENTRENAMIENTO CON EARLY STOPPING
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

def train(model, train_loader, val_loader, epochs=30, lr=1e-3, patience=5):
    model.to(DEVICE)
    
    # Solo optimizamos parametros que requieren gradientes
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_state = None
    
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
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"  -> Nuevo mejor modelo! (Val Acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping activado despues de {epoch+1} epocas")
            break
    
    # Cargar el mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_acc

# ==========================================
# EJECUCION PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("\n=== INICIANDO ENTRENAMIENTO MEJORADO (ResNet18 + Transfer Learning) ===")
    total_start = time.time()
    
    model = FoodResNet(num_classes=NUM_CLASSES)
    print(f"Parametros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Parametros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_acc = train(model, train_loader, val_loader, epochs=30, lr=1e-3, patience=5)
    
    test_acc = accuracy(model, test_loader)
    print(f"\nMejor Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    
    total_time = time.time() - total_start
    print(f"\nTiempo total de ejecucion: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    
    # Guardar modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")
