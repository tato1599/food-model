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
# CONFIGURACION FINAL
# ==========================================
INPUT_ROOT = "/home/neri/escuela/proyectoma/Images"
OUTPUT_ROOT = "/home/neri/escuela/proyectoma/Images_224"
MODEL_PATH = "/home/neri/escuela/proyectoma/modelo_final_comida.pth"
CHECKPOINT_PATH = "/home/neri/escuela/proyectoma/checkpoint_mejor.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# REDIMENSIONAMIENTO A 224x224 (si no existe)
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

def resize_images(input_dir, output_dir, size=(224, 224)):
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

if not os.path.exists(OUTPUT_ROOT) or len(os.listdir(OUTPUT_ROOT)) < 30:
    for class_name in os.listdir(INPUT_ROOT):
        input_class_path = os.path.join(INPUT_ROOT, class_name)
        output_class_path = os.path.join(OUTPUT_ROOT, class_name)
        if os.path.isdir(input_class_path):
            resize_images(input_class_path, output_class_path)
else:
    print("Imagenes 224x224 ya existen, saltando...")

# ==========================================
# DATASETS Y DATALOADERS
# ==========================================
print("\n--- Cargando datasets 224x224 ---")

mean = [0.6288, 0.5124, 0.3893]
std  = [0.2264, 0.2395, 0.2512]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_dataset = datasets.ImageFolder(root=OUTPUT_ROOT)
NUM_CLASSES = len(full_dataset.classes)
print(f"Clases: {NUM_CLASSES}")

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_indices, val_indices, test_indices = random_split(
    range(len(full_dataset)), [train_size, val_size, test_size]
)

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

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=32, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=32, num_workers=4, pin_memory=True)

print(f"Total: {len(full_dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ==========================================
# MODELO: RESNET50 PRE-ENTRENADA
# ==========================================
class FoodResNet50(nn.Module):
    def __init__(self, num_classes=34):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# ENTRENAMIENTO CON CHECKPOINTING
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

def train(model, train_loader, val_loader, epochs=20, lr=5e-4, patience=5, weight_decay=1e-4):
    model.to(DEVICE)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_state = None
    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        
        avg_loss = epoch_loss / batches if batches > 0 else 0
        val_acc = accuracy(model, val_loader)
        scheduler.step(val_acc)
        
        history.append((epoch+1, avg_loss, val_acc))
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"  -> Nuevo mejor modelo! (Val Acc: {val_acc:.4f})")
            # CHECKPOINTING: guardar inmediatamente el mejor modelo
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': best_state,
                'val_acc': best_val_acc,
                'classes': full_dataset.classes,
            }, CHECKPOINT_PATH)
            print(f"  -> Checkpoint guardado en {CHECKPOINT_PATH}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping despues de {epoch+1} epocas")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_acc, history

# ==========================================
# EJECUCION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODELO FINAL CON CHECKPOINTING (ResNet50 + 224x224)")
    print("="*60)
    total_start = time.time()
    
    model = FoodResNet50(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros totales: {total_params:,}")
    print(f"Parametros entrenables: {trainable_params:,}")
    
    best_val_acc, history = train(
        model, train_loader, val_loader, 
        epochs=20, lr=5e-4, patience=5, weight_decay=1e-4
    )
    
    test_acc = accuracy(model, test_loader)
    
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    print(f"Mejor Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test Accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    total_time = time.time() - total_start
    print(f"\nTiempo total: {total_time:.2f}s ({total_time/60:.1f} min)")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': full_dataset.classes,
        'num_classes': NUM_CLASSES,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'history': history
    }, MODEL_PATH)
    print(f"Modelo final guardado en: {MODEL_PATH}")
