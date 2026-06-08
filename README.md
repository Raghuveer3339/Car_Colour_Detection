# 🚗 Task 5 — Car Colour Detection

> **Internship Task** | Machine Learning | Custom CNN from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Raghuveer3339/Car_Colour_Detection/blob/main/Car_Colour_Detection.ipynb)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org)

---

## 📌 Problem Statement

Build a machine learning model that:
- Detects **cars and people** in an image or video frame
- Classifies the **colour of each detected car** into 8 categories
- Applies colour-coded bounding boxes:
  - 🔴 **Red box** → Blue car detected
  - 🔵 **Blue box** → Other colour cars
  - 🟢 **Green box** → Person detected
- Includes an **interactive GUI** for image upload and video processing

---

## 🧠 Approach

### Two-Model Pipeline

```
Input Image / Video Frame
         │
         ▼
┌─────────────────────────────┐
│  Model 1: Detection CNN     │   128×128 input
│  4 Conv Blocks              │   3 classes: background / car / person
│  Sliding Window + Contours  │
└────────────┬────────────────┘
             │  Car detected?
             ▼
┌─────────────────────────────┐
│  Model 2: Colour CNN        │   64×64 car crop
│  3 Conv Blocks              │   8 colour classes
│  Classifies car ROI         │
└────────────┬────────────────┘
             │
             ▼
    Draw bounding box
    Label with colour
    Count cars per colour
```

---

## 🏗️ Model Architectures

### Model 1 — Detection CNN (4 Conv Blocks)
```
Input (128×128×3)
→ Conv Block 1: Conv2D(32) + BatchNorm + ReLU + MaxPool + Dropout
→ Conv Block 2: Conv2D(64) + BatchNorm + ReLU + MaxPool + Dropout
→ Conv Block 3: Conv2D(128) + BatchNorm + ReLU + MaxPool + Dropout
→ Conv Block 4: Conv2D(256) + BatchNorm + ReLU + MaxPool + Dropout
→ GlobalAvgPool → Dense(512) → Dense(256)
→ Dense(3, Softmax)   ← background / car / person
```

### Model 2 — Colour Classifier CNN (3 Conv Blocks)
```
Input (64×64×3) ← Car ROI crop
→ Conv Block 1: Conv2D(32) + BatchNorm + ReLU + MaxPool + Dropout
→ Conv Block 2: Conv2D(64) + BatchNorm + ReLU + MaxPool + Dropout
→ Conv Block 3: Conv2D(128) + BatchNorm + ReLU + MaxPool + Dropout
→ GlobalAvgPool → Dense(256) → Dropout
→ Dense(8, Softmax)   ← 8 colour classes
```

**Colour Classes:** Red · Blue · White · Black · Silver · Yellow · Green · Orange

---

## 📊 Dataset

### Colour Classification Dataset
- **Source:** Synthetic dataset generated from scratch using NumPy + OpenCV
- **300 images per class × 8 classes = 2,400 total images**
- Each image: 64×64 pixels with realistic colour gradients, surface noise, and lighting variation
- No external download required — auto-generated in notebook

### Detection Dataset
- **Source:** Synthetic traffic scene patches generated using NumPy + OpenCV
- **400 samples per class × 3 classes = 1,200 total samples** (128×128 pixels)
- Background: sky gradient + road texture
- Car: coloured body + windscreen + wheels (all drawn programmatically)
- Person: head + body + legs (drawn programmatically)

> **Why synthetic?** Kaggle API and fiftyone library had version conflicts with Python 3.12. Synthetic data allows full reproducibility with zero external dependencies.

---

## 🔬 Training Details

| | Detection CNN | Colour CNN |
|---|---|---|
| Input size | 128×128×3 | 64×64×3 |
| Classes | 3 (bg/car/person) | 8 colours |
| Epochs | 40 | 50 |
| Batch size | 32 | 32 |
| Optimizer | Adam (lr=1e-3) | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy | Categorical Crossentropy |
| Split | 70/15/15 train/val/test | 70/15/15 train/val/test |
| Callbacks | EarlyStopping · ReduceLROnPlateau · ModelCheckpoint | Same |

### Data Augmentation
- **Detection:** rotation, shift, horizontal flip, zoom, brightness
- **Colour:** rotation, shift, horizontal flip, zoom only
  *(No hue/brightness changes — preserves colour information)*

---

## 📁 Project Structure

```
Car_Colour_Detection/
│
├── Car_Colour_Detection.ipynb   # Main Colab notebook
│
└── README.md
```

**Google Drive folder** (auto-created by notebook):
```
Internship_Datasets/Car_Colour_Detection/
├── models/
│   ├── detector_model.h5
│   └── colour_model.h5
└── Outputs/
    ├── sample_colours.png
    ├── detection_samples.png
    ├── model_architectures.png
    ├── detect_training_history.png
    ├── colour_training_history.png
    ├── colour_confusion_matrix.png
    └── final_summary_dashboard.png
```

---

## 🚀 How to Run

1. Click **Open in Colab** badge above
2. `Runtime → Change runtime type → T4 GPU`
3. Run **Step 0 cell first** → then `Runtime → Restart Runtime`
4. Run all remaining cells top to bottom
5. Mount Google Drive when prompted
6. Everything else is **fully automatic** — no API keys, no downloads needed

---

## 🖥️ GUI Features

Built with **ipywidgets** inside Google Colab:
- 📂 Upload any image file
- ⚡ Click **Detect & Analyse** → bounding boxes drawn automatically
- 🎨 Colour label shown on each detected car
- 📊 Car count per colour displayed
- 💾 Result saved to Google Drive `Outputs/` folder
- 📹 Separate **video processing** module (Block 9)

---

## 📦 Requirements

All installed automatically in the notebook:
```
tensorflow>=2.21.0
opencv-python-headless
numpy
scikit-learn
matplotlib
seaborn
ipywidgets
pillow
tqdm
gdown
```

---

## ⚠️ Known Issues & Fixes Applied

| Issue | Fix |
|---|---|
| `ImportError: cannot import 'runtime_version' from google.protobuf` | Step 0 cell reinstalls compatible protobuf before TF loads |
| `tensorflow==2.12.0` not found on Python 3.12 | Use plain `tensorflow` (installs 2.21) |
| Kaggle API authentication failure | Replaced with synthetic dataset generator |
| `ModuleNotFoundError: fiftyone` | Replaced with synthetic traffic scene generator |
| `FileNotFoundError` on `plt.savefig` | `os.makedirs(..., exist_ok=True)` added before all saves |
| `IndentationError` on `plt.savefig` / `plt.show` | Fixed indentation in all plot cells |
| `ValueError: Invalid RGBA argument '#FFG8G8'` | Fixed to valid hex `#FF8888` |

---

## 👤 Author

**Raghuveer** | Internship Project — Task 5
Built with Python · TensorFlow · OpenCV · ipywidgets
