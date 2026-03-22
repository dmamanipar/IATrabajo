# Fine-tuning YOLOv11 para Detección de Señales de Tránsito

Proyecto de detección de objetos basado en **YOLOv11s** entrenado con el dataset **TrafficRoadSignsYolov11** de Roboflow. El modelo identifica y clasifica **31 tipos de señales de tránsito** en imágenes mediante transfer learning sobre pesos pre-entrenados en COCO.

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Dataset](#dataset)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación y Configuración](#instalación-y-configuración)
- [Entrenamiento](#entrenamiento)
- [Métricas y Resultados](#métricas-y-resultados)
- [Inferencia](#inferencia)
- [Exportación del Modelo](#exportación-del-modelo)
- [Clases Detectadas](#clases-detectadas)
- [Autor](#autor)

---

## Descripción del Proyecto

El objetivo es completar el flujo de fine-tuning de principio a fin: descarga del dataset desde Roboflow, configuración del entrenamiento, análisis de métricas, evaluación formal, comparativa con el modelo baseline (COCO) y exportación a formato ONNX para despliegue en producción.

Todo el pipeline se ejecuta en **Google Colab** con los resultados persistidos en **Google Drive** para mantener checkpoints entre sesiones.

## Arquitectura del Modelo

| Propiedad | Valor |
|-----------|-------|
| **Modelo base** | YOLOv11s |
| **Parámetros** | 9.4 M |
| **Tamaño de entrada** | 640 × 640 px |
| **Framework** | Ultralytics (PyTorch) |
| **Pre-entrenamiento** | COCO (80 clases) |

Se seleccionó la variante **small (s)** por ser el punto de equilibrio entre capacidad y riesgo de overfitting para datasets de tamaño mediano (500–5000 imágenes).

## Dataset

- **Fuente:** Roboflow — TrafficRoadSignsYolov11
- **Formato:** YOLOv11 (bounding boxes normalizados)
- **Clases:** 31 señales de tránsito
- **Splits:** Train / Validation / Test
- **Augmentation offline:** Aplicado por Roboflow al set de train antes de la exportación

### Estructura del dataset

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

> **Nota sobre `data.yaml`:** Las rutas generadas por Roboflow son relativas y pueden fallar en Colab. El notebook genera un `data_fixed.yaml` con la clave `path` apuntando a la ruta absoluta del dataset para evitar errores de resolución de rutas.

## Estructura del Repositorio

```
DAVIDMP-2026/IA/
├── datasets/
│   └── TrafficRoadSignsYolov11/
│       ├── train/
│       ├── valid/
│       ├── test/
│       ├── data.yaml
│       └── data_fixed.yaml
├── models/
│   └── yolo11_trabajo/
│       ├── weights/
│       │   ├── best.pt          # Mejor checkpoint (mayor mAP en val)
│       │   └── last.pt          # Último epoch ejecutado
│       ├── results.csv
│       ├── confusion_matrix_normalized.png
│       ├── BoxP_curve.png
│       ├── BoxF1_curve.png
│       └── ...
├── outputs/
│   └── best.onnx               # Modelo exportado
└── sesion7_notebook.ipynb
```

## Requisitos

- Python 3.10+
- Google Colab (con GPU T4 recomendada)
- Google Drive para persistencia

### Dependencias principales

```
ultralytics
roboflow
torch
opencv-python
matplotlib
pandas
numpy
pyyaml
pillow
```

## Instalación y Configuración

### 1. Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Instalar dependencias

```bash
pip install -q ultralytics roboflow
```

### 3. Verificar GPU

```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
```

### 4. Cargar dataset

**Opción A — ZIP desde Roboflow (recomendado):**

1. Exportar dataset en formato YOLOv11 desde la web de Roboflow
2. Subir el ZIP a `Drive/DAVIDMP-2026/IA/datasets/`
3. El notebook lo descomprime automáticamente

**Opción B — API de Roboflow:**

```python
from roboflow import Roboflow
rf = Roboflow(api_key='TU_API_KEY')
project = rf.workspace('tu-workspace').project('tu-proyecto')
dataset = project.version(1).download('yolov11')
```

## Entrenamiento

### Configuración

| Hiperparámetro | Valor |
|----------------|-------|
| **Epochs** | 100 |
| **Patience (early stopping)** | 20 |
| **Batch size** | 16 |
| **Tamaño de imagen** | 640 px |
| **Workers** | 2 |

### Ejecutar entrenamiento

```python
from ultralytics import YOLO

model = YOLO('yolo11s.pt')
results = model.train(
    data    = 'data_fixed.yaml',
    epochs  = 100,
    patience= 20,
    batch   = 16,
    imgsz   = 640,
    device  = 'cuda',
    workers = 2,
    project = 'models/',
    name    = 'yolo11_trabajo',
    exist_ok= True,
    plots   = True,
)
```

### Reanudar entrenamiento interrumpido

Si la sesión de Colab se desconecta, el entrenamiento se puede retomar desde el último checkpoint:

```python
model = YOLO('models/yolo11_trabajo/weights/last.pt')
model.train(resume=True)
```

El archivo `last.pt` almacena el estado completo (época actual, pesos, optimizador, hiperparámetros), por lo que no es necesario re-especificar la configuración.

### Componentes del loss

- **Box loss:** Error de localización del bounding box
- **Cls loss:** Error de clasificación de la clase
- **DFL loss:** Distribución de probabilidad de los bordes del bounding box

## Métricas y Resultados

### Métricas principales

| Métrica | Descripción |
|---------|-------------|
| **mAP@50** | Mean Average Precision con IoU ≥ 0.50 |
| **mAP@50:95** | mAP promediado en umbrales IoU de 0.50 a 0.95 |
| **Precision** | Proporción de detecciones correctas sobre el total de detecciones |
| **Recall** | Proporción de objetos reales detectados |

### Matriz de confusión

La matriz de confusión normalizada muestra que la gran mayoría de las 31 clases se clasifican correctamente con valores cercanos a 1.0 en la diagonal. Las confusiones menores se concentran entre señales visualmente similares:

- **Speed Limit 20 KMPh** ↔ **Speed Limit 30 KMPh** (solo cambia el número)
- **50 mph speed limit** ↔ **Speed Limit 20 KMPh** (señales circulares con números)

### Evaluación formal

La evaluación se realiza con `best.pt` (checkpoint con mejor mAP en validación):

```python
best_model = YOLO('models/yolo11_trabajo/weights/best.pt')
metrics = best_model.val(data='data_fixed.yaml', split='val')
```

### Gráficos generados automáticamente

- `confusion_matrix_normalized.png` — Matriz de confusión normalizada
- `BoxP_curve.png` — Curva Precision-Recall
- `BoxF1_curve.png` — Curva F1 vs umbral de confianza
- `results.csv` — Métricas por época

## Inferencia

```python
from ultralytics import YOLO

model = YOLO('models/yolo11_trabajo/weights/best.pt')
results = model('imagen.jpg', conf=0.25, iou=0.45)

# Visualizar resultado
results[0].plot()
```

### Parámetros de inferencia

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `conf` | 0.25 | Umbral mínimo de confianza |
| `iou` | 0.45 | Umbral IoU para Non-Maximum Suppression |

Subir `conf` reduce falsos positivos pero puede perder detecciones de baja confianza. Bajar `iou` hace que NMS elimine más boxes solapadas.

## Exportación del Modelo

### Exportar a ONNX

```python
model = YOLO('models/yolo11_trabajo/weights/best.pt')
model.export(
    format  = 'onnx',
    imgsz   = 640,
    dynamic = True,
    simplify= True,
    opset   = 17
)
```

### Formatos de exportación disponibles

| Formato | Uso recomendado |
|---------|-----------------|
| **PyTorch (.pt)** | Desarrollo y experimentación |
| **ONNX (.onnx)** | Despliegue portable (cualquier framework/OS/hardware) |
| **TensorRT** | Servidores con GPU NVIDIA (menor latencia) |

## Clases Detectadas

El modelo reconoce las siguientes 31 señales de tránsito:

| # | Clase | # | Clase |
|---|-------|---|-------|
| 0 | Road narrows on right | 16 | Pedestrian Crossing |
| 1 | 50 mph speed limit | 17 | Round-About |
| 2 | Attention Please | 18 | Slippery Road Ahead |
| 3 | Beware of children | 19 | Speed Limit 20 KMPh |
| 4 | CYCLE ROUTE AHEAD WARNING | 20 | Speed Limit 30 KMPh |
| 5 | Dangerous Left Curve Ahead | 21 | Stop Sign |
| 6 | Dangerous Right Curve Ahead | 22 | Straight Ahead Only |
| 7 | End of all speed and passing limits | 23 | Traffic Signal |
| 8 | Give Way | 24 | Truck traffic is prohibited |
| 9 | Go Straight or Turn Right | 25 | Turn left ahead |
| 10 | Go straight or turn left | 26 | Turn right ahead |
| 11 | Keep-Left | 27 | Uneven Road |
| 12 | Keep-Right | 28 | background |
| 13 | Left Zig Zag Traffic | 29 | — |
| 14 | No Entry | 30 | — |
| 15 | No Over Taking | | |

## Notas Técnicas

- **Fine-tuning vs entrenamiento desde cero:** Los pesos pre-entrenados en COCO proporcionan features visuales generales. El modelo solo necesita aprender a asociar esos features con las nuevas clases, convergiendo en 50–100 epochs en lugar de 300+.
- **Augmentation offline (Roboflow) vs online (YOLO):** Roboflow genera imágenes fijas en disco antes del entrenamiento. YOLO aplica transformaciones aleatorias en cada epoch. Cuidado con superponer las mismas transformaciones en ambos.
- **Early stopping:** Con `patience=20`, el entrenamiento se detiene automáticamente si no hay mejora en mAP durante 20 epochs consecutivos.
- **best.pt vs last.pt:** `best.pt` corresponde al epoch con mejor mAP en validación; `last.pt` al último epoch ejecutado. Para producción usar siempre `best.pt`.

## Entorno de Ejecución

| Componente | Especificación |
|------------|---------------|
| **Plataforma** | Google Colab |
| **GPU** | NVIDIA T4 (16 GB VRAM) |
| **Framework** | Ultralytics / PyTorch |
| **Almacenamiento** | Google Drive |

## Autor

**David Mamani Pari**

Curso: Especialización en Computer Vision — Marzo 2026

---

## Licencia

Este proyecto fue desarrollado con fines educativos como parte del curso de especialización en Computer Vision.
