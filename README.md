# Proyecto Final: Clasificación de Imágenes de Comida

**Integrantes:**  
- Lolita Alva Carreño (Lolita) - 21111241  
- Daniel Octavio Ramirez Neri - 21111240  
**Fecha:** Mayo 2026  
**Entorno:** Debian + NVIDIA GeForce RTX 4060  
**Framework:** PyTorch 2.5.1 + CUDA 12.1  

---

## 1. Resumen Ejecutivo

Este proyecto implementa un sistema de clasificación de imágenes de comida capaz de identificar **34 clases** diferentes (apple_pie, burger, sushi, tacos, etc.). El trabajo se migró desde Google Colab a un entorno local en WSL para aprovechar una GPU NVIDIA RTX 4060, obteniendo mejoras significativas en tiempos de ejecución y control del entorno.

Se experimentaron tres arquitecturas:
1. **Modelo base:** Red neuronal fully-connected simple.
2. **Modelo mejorado:** ResNet18 con Transfer Learning, alcanzando un **78.78% de accuracy**.
3. **Modelo final:** ResNet50 + 224×224 + Fine-tuning profundo, alcanzando un **91.23% de accuracy** en el conjunto de prueba.

---

## 2. Estructura del Proyecto

```
proyectoma/
├── Images/                          # Dataset original (23,873 imágenes)
│   ├── apple_pie/
│   ├── burger/
│   ├── sushi/
│   └── ... (34 carpetas)
│
├── Images_128/                      # Dataset preprocesado (128x128)
│   ├── apple_pie/
│   └── ...
│
├── Images_224/                      # Dataset preprocesado (224x224)
│   ├── apple_pie/
│   └── ...
│
├── entrenar_comida.py               # Script del modelo base
├── mejorar_modelo.py                # Script del modelo mejorado (ResNet18)
├── modelo_final_checkpoint.py       # Script del modelo final (ResNet50)
├── predict.py                       # Script para predecir una imagen
├── modelo_comida.pth                # Pesos del modelo base (~21% acc)
├── mejor_modelo_comida.pth          # Pesos del modelo mejorado (~79% acc)
├── modelo_final_comida.pth          # Pesos del modelo final (~91% acc)
├── checkpoint_mejor.pt              # Checkpoint intermedio del modelo final
├── README.md                        # Este documento
└── requirements.txt                 # Dependencias del proyecto
```

---

## 3. Dataset

### 3.1 Descripción
- **Total de imágenes:** 23,873
- **Número de clases:** 34
- **Origen:** [Kaggle - Food Image Classification Dataset](https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset?select=Food+Classification+dataset).
- **Formato:** Archivos JPG de diferentes tamaños originales.

### 3.4 Fuente de descarga
- Link oficial del dataset: https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset?select=Food+Classification+dataset

### 3.2 Distribución
| Conjunto | Cantidad | Porcentaje |
|----------|----------|------------|
| Entrenamiento (Train) | 16,711 | 70% |
| Validación (Val) | 3,580 | 15% |
| Prueba (Test) | 3,582 | 15% |

### 3.3 Clases detectadas
`Baked_Potato`, `Crispy_Chicken`, `Donut`, `Fries`, `Hot_Dog`, `Sandwich`, `Taco`, `Taquito`, `apple_pie`, `burger`, `butter_naan`, `chai`, `chapati`, `cheesecake`, `chicken_curry`, `chole_bhature`, `dal_makhani`, `dhokla`, `fried_rice`, `ice_cream`, `idli`, `jalebi`, `kaathi_rolls`, `kadai_paneer`, `kulfi`, `masala_dosa`, `momos`, `omelette`, `paani_puri`, `pakode`, `pav_bhaji`, `pizza`, `samosa`, `sushi`.

---

## 4. Preprocesamiento de Datos

### 4.1 Redimensionamiento
Las imágenes originales tenían tamaños variables. Se redimensionaron uniformemente a **128×128 píxeles** para estandarizar la entrada a la red neuronal.

- **Método:** `ImageOps.fit()` con interpolación `LANCZOS` (alta calidad).
- **Formato de salida:** RGB, JPEG calidad 95.
- **Manejo de transparencia:** Imágenes PNG/P con canal alpha se componen sobre fondo blanco antes de convertir a RGB.
- **Tiempo de procesamiento:** ~142 segundos (ejecución única).

### 4.2 Normalización
Los valores de media y desviación estándar se calcularon sobre el conjunto de entrenamiento:
- **Media:** `[0.6288, 0.5124, 0.3893]`
- **Std:** `[0.2264, 0.2395, 0.2512]`

### 4.3 Data Augmentation (Modelo Mejorado)
Para reducir el sobreajuste y mejorar la generalización, se aplicaron transformaciones aleatorias durante el entrenamiento:
- `RandomHorizontalFlip()`
- `RandomRotation(20°)`
- `RandomResizedCrop(128, scale=(0.8, 1.0))` — simula diferentes encuadres/zooms.
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` — simula cambios de iluminación.

Los conjuntos de validación y prueba **no** reciben augmentación, solo normalización.

---

## 5. Arquitecturas de Modelos

### 5.1 Modelo Base (entrenar_comida.py)

**Tipo:** Red neuronal fully-connected simple (perceptrón multicapa).

**Arquitectura:**
```
Input: 3 × 128 × 128
↓ AvgPool2d(4)          → 3 × 32 × 32
↓ Flatten               → 3072
↓ Linear(3072, 512)     → 512
↓ ReLU()
↓ Linear(512, 34)       → 34 clases
```

**Características:**
- Parámetros totales: **1,590,818**
- Optimizador: Adam (lr=3e-4)
- Función de pérdida: CrossEntropyLoss
- Épocas: 5

**Problema identificado en código original:**
El notebook original contenía un error tipográfico crítico en la función de entrenamiento:
```python
# ORIGINAL (erróneo):
modeel.to(dvice)

# CORREGIDO:
model.to(DEVICE)
```

**Resultados:**
- Test Accuracy: **21.16%**
- Tiempo de entrenamiento: ~30 segundos

**Análisis:** El modelo es demasiado simple para tareas de visión por computadora. Al aplanar la imagen directamente, se pierde completamente la información espacial y de textura que es crucial para distinguir objetos visuales.

---

### 5.2 Modelo Mejorado (mejorar_modelo.py)

**Tipo:** ResNet18 con Transfer Learning (Fine-tuning selectivo).

**Estrategia de Transfer Learning:**
Se utilizó una ResNet18 pre-entrenada en el dataset ImageNet (millones de imágenes de 1000 clases). Los pesos pre-entrenados permiten a la red reutilizar características visuales de bajo y medio nivel (bordes, texturas, formas) que son universales para cualquier imagen.

**Arquitectura adaptada:**
```
Backbone: ResNet18 (pre-entrenada en ImageNet)
  └─ Capas 1-3: Congeladas (no se entrenan)
  └─ Layer4: Descongelada para fine-tuning
  └─ Fully Connected (original): Reemplazada
      ↓ Dropout(0.5)
      ↓ Linear(512, 512)
      ↓ ReLU()
      ↓ Dropout(0.3)
      ↓ Linear(512, 34)  → 34 clases
```

**Características:**
- Parámetros totales: **11,456,610**
- Parámetros entrenables: **8,673,826**
- Optimizador: Adam (lr=1e-3) — solo sobre parámetros descongelados.
- Scheduler: `ReduceLROnPlateau` (reduce lr a la mitad si no mejora en 2 épocas).
- Regularización: Dropout 0.5 y 0.3.
- Early Stopping: paciencia de 5 épocas sin mejora.
- Épocas máximas: 30 (se detuvo automáticamente en 23).

**Resultados:**
- Mejor Validation Accuracy: **78.72%**
- Test Accuracy: **78.78%**
- Tiempo total de ejecución: ~4 minutos

---

### 5.3 Modelo Final (modelo_final_checkpoint.py)

**Tipo:** ResNet50 con Transfer Learning y Fine-tuning profundo.

**Arquitectura adaptada:**
```
Backbone: ResNet50 (pre-entrenada en ImageNet)
  ├─ Capas 1-2: Congeladas
  ├─ Layer3: Descongelada para fine-tuning
  ├─ Layer4: Descongelada para fine-tuning
  └─ Fully Connected (original): Reemplazada por clasificador profundo
      ↓ Dropout(0.6)
      ↓ Linear(2048, 1024)
      ↓ BatchNorm1d(1024)
      ↓ ReLU()
      ↓ Dropout(0.4)
      ↓ Linear(1024, 512)
      ↓ BatchNorm1d(512)
      ↓ ReLU()
      ↓ Dropout(0.3)
      ↓ Linear(512, 34)  → 34 clases
```

**Características:**
- Parámetros totales: **26,151,522**
- Parámetros entrenables: **24,706,594** (94.5%)
- Resolución de entrada: **224×224** (tamaño nativo de ResNet)
- Optimizador: AdamW (lr=5e-4, weight_decay=1e-4)
- Scheduler: `ReduceLROnPlateau` (factor 0.3, paciencia 3)
- Regularización: Dropout 0.6 / 0.4 / 0.3, Label Smoothing 0.1, Weight Decay 1e-4
- Early Stopping: paciencia de 5 épocas
- Checkpointing: Guarda el mejor modelo automáticamente en cada mejora
- Épocas ejecutadas: 20

**Data Augmentation avanzada:**
- RandomHorizontalFlip
- RandomRotation(25°)
- RandomResizedCrop(224, scale=0.7-1.0)
- RandomAffine(translate=10%)
- ColorJitter(brightness/contrast/saturation/hue)
- RandomPerspective(distortion_scale=0.2)

**Resultados:**
- Mejor Validation Accuracy: **91.01%**
- Test Accuracy: **91.23%**
- Tiempo total de ejecución: **~24 minutos**

---

## 6. Tabla Comparativa de Resultados

| Métrica | Modelo Base | Modelo Mejorado | **Modelo Final** |
|---------|-------------|-----------------|------------------|
| Arquitectura | MLP Simple | ResNet18 + Transfer Learning | **ResNet50 + Fine-tuning** |
| Resolución | 128×128 | 128×128 | **224×224** |
| Parámetros | 1.6M | 11.5M | **26.1M** |
| Test Accuracy | 21.16% | 78.78% | **91.23%** |
| Val Accuracy | 21.26% | 78.72% | **91.01%** |
| Tiempo entrenamiento | 30s | 4 min | **~24 min** |
| Épocas ejecutadas | 5 | 23 | **20** |
| Data Augmentation | Básica | Avanzada | **Máxima** |
| Regularización | Ninguna | Dropout + Early Stopping | **Dropout + Early Stopping + Label Smoothing + Weight Decay** |

---

## 7. Migración desde Google Colab

### 7.1 Problemas originales del notebook (comida.ipynb)
El notebook fue desarrollado originalmente en Google Colab, lo que generó las siguientes dependencias de infraestructura que debieron resolverse:

1. **Rutas hardcodeadas a Google Drive:**
   ```python
   # ORIGINAL:
   input_root = "/content/drive/MyDrive/Modelos_de_aprendizaje/..."
   
   # SOLUCIÓN:
   INPUT_ROOT = "/home/neri/escuela/proyectoma/Images"
   ```

2. **Entorno de ejecución dependiente de Colab:** Uso de rutas `/content/...` y estructura de Drive.

3. **Formato `.ipynb`:** Se convirtió a scripts `.py` ejecutables por terminal para mayor flexibilidad y facilidad de debugging.

### 7.2 Adaptaciones realizadas
| Aspecto | Colab (Original) | WSL Local (Adaptado) |
|---------|------------------|----------------------|
| Almacenamiento de imágenes | Google Drive (`/content/drive/`) | Disco WSL (`/home/neri/escuela/proyectoma/Images`) |
| GPU | Tesla T4 (Colab) | NVIDIA GeForce RTX 4060 (8GB) |
| Entorno de ejecución | Jupyter Notebook | Scripts Python + Terminal |
| Persistencia de datos | Dependiente de Drive | Local y persistente en WSL |
| Velocidad de I/O | Moderada (Drive virtual) | **Rápida** (filesystem ext4 nativo) |

### 7.3 Optimización de rendimiento
Se copiaron las imágenes desde el disco montado de Windows (`/mnt/c/Users/...`) al filesystem nativo de WSL (`/home/neri/...`) para eliminar el cuello de botella de I/O entre WSL y NTFS. Esto aceleró significativamente la carga de datos durante el entrenamiento.

---

## 8. Hardware y Entorno de Ejecución

### 8.1 Especificaciones técnicas
- **GPU:** NVIDIA GeForce RTX 4060 (8 GB GDDR6)
- **CUDA:** Versión 12.1
- **Driver:** 535.261.03
- **CPU:** Compatible con x64
- **RAM:** Suficiente para batch size de 64 imágenes.

### 8.2 Verificación de entorno
Para confirmar que CUDA está disponible, se ejecutó:
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

Resultado:
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 4060
```

---

## 9. Guía de Uso

### 9.1 Instalación de dependencias
```bash
pip install torch torchvision pillow tqdm
```

### 9.2 Ejecución del modelo base
```bash
python3 entrenar_comida.py
```

### 9.3 Ejecución del modelo mejorado
```bash
python3 mejorar_modelo.py
```

### 9.4 Ejecución del modelo final (ResNet50)
```bash
python3 modelo_final_checkpoint.py
```

### 9.5 Predicción con una imagen nueva
Se incluye el script `predict.py` que carga el modelo final y predice la clase de cualquier imagen:

```bash
python3 predict.py
# Te pedirá la ruta de la imagen a clasificar
```

**Ejemplo de salida:**
```
Ruta de la imagen a clasificar: /ruta/a/mi_foto.jpg
Modelo cargado exitosamente!
Clases: 34

========================================
RESULTADO DE LA PREDICCION
========================================
Clase predicha: sushi
Confianza: 97.45%

Top 5 predicciones:
  1. sushi - 97.45%
  2. sandwich - 1.23%
  3. burger - 0.89%
  4. pizza - 0.21%
  5. taco - 0.12%
```

### 9.6 Reutilizar el modelo final en código propio
```python
import torch
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir arquitectura (debe coincidir con el entrenamiento)
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

# Cargar checkpoint
checkpoint = torch.load("modelo_final_comida.pth", map_location=DEVICE)
CLASSES = checkpoint['classes']

model = FoodResNet50(num_classes=len(CLASSES))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Preprocesar (224x224 para el modelo final)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.6288, 0.5124, 0.3893], [0.2264, 0.2395, 0.2512])
])

img = Image.open("ruta/a/imagen.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    output = model(img_tensor)
    pred = output.argmax(1).item()
    print(f"Predicción: {CLASSES[pred]}")
```

---

## 10. Hallazgos y Lecciones Aprendidas

1. **El typo importa:** Un simple error tipográfico (`modeel.to(dvice)`) en el notebook original hubiera causado un fallo silencioso o un crash si no se hubiera revisado el código línea por línea durante la migración.

2. **Transfer Learning es clave:** Para datasets de tamaño moderado (~24k imágenes), utilizar una red pre-entrenada y hacer fine-tuning es drásticamente más efectivo que entrenar desde cero.

3. **El I/O importa:** Leer miles de imágenes desde un disco NTFS montado en WSL (`/mnt/c/...`) es significativamente más lento que tenerlas en el filesystem nativo de Linux.

4. **Early Stopping ahorra tiempo:** En lugar de entrenar 30 épocas fijas, el modelo se detuvo en la época 23 porque la validación no mejoró, ahorrando ~7 épocas innecesarias y previniendo sobreajuste.

5. **Checkpointing es indispensable:** Durante el entrenamiento del modelo final, el proceso fue interrumpido dos veces por timeouts. Implementar guardado de checkpoints en cada mejora permitió asegurar que el mejor modelo nunca se perdiera.

6. **La resolución importa:** Pasar de 128×128 a 224×224 (tamaño nativo de ResNet) mejoró significativamente el accuracy, ya que la red pudo aprovechar mejor los detalles finos de las imágenes.

7. **Label Smoothing y Weight Decay ayudan:** Estas técnicas de regularización fueron cruciales para alcanzar el 91% sin caer en sobreajuste, especialmente al entrenar millones de parámetros.

---

## 11. Próximos Pasos / Trabajo Futuro

El objetivo original de 85-90% fue **superado** (91.23% Test Accuracy). Para futuras iteraciones:

1. **Ensemble de modelos:** Combinar predicciones de ResNet50, ResNet18 y EfficientNet para llegar al 93-95%.
2. **Test Time Augmentation (TTA):** Aplicar augmentación también durante la inferencia y promediar resultados.
3. **Análisis de errores:** Revisar las clases con peor accuracy para identificar si hay confusiones sistemáticas (ej. "sushi" vs "rollos").
4. **Despliegue web:** Crear una API con FastAPI o Flask para que el modelo sea accesible desde una aplicación.
5. **Recolección de más datos:** Algunas clases tienen menos de 300 imágenes; aumentar su cantidad podría mejorar la generalización.

---

## 12. Referencias

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- torchvision.models: https://pytorch.org/vision/stable/models.html
- ResNet Paper: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- Dataset: Food Images (colección local de 34 categorías)


