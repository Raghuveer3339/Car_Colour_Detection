# 🚗🎨 Task 5 — Car Colour Detection & Traffic Signal Analysis

A dual custom CNN system built from scratch that detects cars and people at a traffic signal, classifies car colours, and annotates results with colour-coded bounding boxes. Includes an ipywidgets GUI for image upload and a video processing pipeline.

---

## Problem Statement

At busy traffic signals, manually counting vehicles and identifying car colours is impractical. This system automates:
- Detecting cars and people in traffic images and video
- Classifying the colour of each detected car (8 colour classes)
- Drawing colour-coded bounding boxes:
  - 🔴 **Red rectangle** → Blue car
  - 🔵 **Blue rectangle** → Any other car colour
  - 🟢 **Green rectangle** → Person at the signal
- Counting and displaying the total number of cars and people

---

## Dataset

### Colour Classification Dataset
**Vehicle Color Recognition (VCoR)** — 8 classes

| Property | Details |
|---|---|
| Classes | red, blue, white, black, silver, yellow, green, orange |
| Images per class | up to 300 (subset used) |
| Input size | 64×64 px, RGB |
| Fallback | Synthetic colour-patch generator (if download fails) |

Folder name normalisation handled via `COLOUR_ALIASES` mapping (`gray→silver`, `cyan→blue`, `maroon→red`, etc.) and `map_folder_to_colour()` to handle numeric prefixes like `01_black`.

### Detection Dataset
**COCO Traffic Subset** — 3 classes: background, car, person

| Property | Details |
|---|---|
| Classes | background (0), car (1), person (2) |
| Input size | 128×128 px, RGB |
| Strategy | Patch extraction via sliding window for training |

---

## Methodology

### 1. Two-Model Pipeline Rationale

The system uses **two separate CNNs** rather than one overloaded network:

| Model | Task | Input | Classes |
|---|---|---|---|
| Detection CNN | Is this patch a car, person, or background? | 128×128 | 3 |
| Colour CNN | What colour is this car? | 64×64 | 8 |

Separating the tasks allows each model to be independently optimised and retrained without affecting the other.

### 2. Preprocessing & Augmentation

**Detection model augmentation:**
- Rotation ±15°, width/height shift 10%, zoom 10%, shear 10%
- Horizontal flip enabled (cars/people look same when mirrored)

**Colour model augmentation:**
- Rotation ±10°, width/height shift 5%, zoom 5%
- **No hue/saturation shift** — would corrupt colour ground-truth labels
- **No horizontal flip issues** — symmetric for colour

### 3. Model Architectures — Both Custom CNNs from Scratch

**Detection CNN (4 Conv Blocks)**
```
Input: 128×128×3
  ├── Conv Block 1: Conv2D(32)×2 → BN → MaxPool → Dropout(0.25)
  ├── Conv Block 2: Conv2D(64)×2 → BN → MaxPool → Dropout(0.25)
  ├── Conv Block 3: Conv2D(128)×2 → BN → MaxPool → Dropout(0.25)
  ├── Conv Block 4: Conv2D(256)×2 → BN → GlobalAvgPool → Dropout(0.4)
  ├── Dense(512) → BN → Dropout(0.5)
  └── Dense(3, softmax)  [background / car / person]
```

**Colour CNN (3 Conv Blocks)**
```
Input: 64×64×3
  ├── Conv Block 1: Conv2D(32)×2 → BN → MaxPool → Dropout(0.25)
  ├── Conv Block 2: Conv2D(64)×2 → BN → MaxPool → Dropout(0.25)
  ├── Conv Block 3: Conv2D(128)×2 → BN → GlobalAvgPool → Dropout(0.4)
  ├── Dense(256) → BN → Dropout(0.4)
  └── Dense(8, softmax)  [8 colour classes]
```

**Model Selection Justification:**

| Model | Params | Expected Accuracy | Notes |
|---|---|---|---|
| MLP patch classifier (baseline) | ~1M | ~45–60% | No spatial awareness |
| Simple CNN 2-block detector | ~0.3M | ~70–78% | Misses small objects |
| **Our Detection CNN (4-block)** | **~4M** | **~85–92%** | **Chosen** |
| Simple CNN 2-block colour | ~0.2M | ~75–82% | Insufficient for 8 classes |
| **Our Colour CNN (3-block)** | **~1.5M** | **~88–94%** | **Chosen** |
| VGG16/ResNet pretrained | 138M+ | ~97–99% | Not allowed (scratch constraint) |

### 4. Inference Pipeline

```
Input Image / Video Frame
        ↓
  Sliding Window (multi-scale patches)
        ↓
  Detection CNN → patch classification
        ↓
  Non-Maximum Suppression (NMS) → final bounding boxes
        ↓
  Car ROI crop → Colour CNN → colour label
        ↓
  Draw annotations:
      Red box   = Blue car
      Blue box  = Other car
      Green box = Person
        ↓
  Count display + save output
```

### 5. Training

| Parameter | Detection CNN | Colour CNN |
|---|---|---|
| Optimiser | Adam (lr=0.001) | Adam (lr=0.001) |
| Loss | Categorical Cross-entropy | Categorical Cross-entropy |
| Epochs | Up to 30 (EarlyStopping) | Up to 30 (EarlyStopping) |
| Batch size | 64 | 64 |
| EarlyStopping | patience=8, val_accuracy | patience=8, val_accuracy |
| ReduceLROnPlateau | factor=0.5, patience=4 | factor=0.5, patience=4 |

---

## Results

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Detection CNN | ~88–92% (update after training) | ~0.20–0.30 |
| Colour CNN | ~90–94% (update after training) | ~0.15–0.25 |

Full classification reports and confusion matrices are generated in Block 10 of the notebook.

---

## Visual Outputs

All plots are saved to `Task5_Car_Colour_Detection/Outputs/` on Google Drive:

| File | Description |
|---|---|
| `sample_colour_images.png` | Sample grid of all 8 colour classes |
| `detection_training_history.png` | Detection CNN accuracy & loss curves |
| `colour_training_history.png` | Colour CNN accuracy & loss curves + confusion matrix |
| `counts_<filename>.png` | Bar chart of cars/people/total for each analysed image |
| `annotated_<filename>.jpg` | Annotated output image with coloured bounding boxes |
| `annotated_traffic.mp4` | Annotated output video (Block 9) |
| `final_summary_dashboard.png` | Full dashboard: architecture, pipeline, training curves, confusion matrix |

---

## GUI — ipywidgets (Colab)

Block 8 launches an interactive dark-themed dashboard directly in Colab:

**Features:**
- **Confidence slider** — adjust detection threshold (0.30–0.95)
- **Upload Image button** — load any JPG/PNG traffic image
- **Analyse button** — runs the full detection + colour pipeline
- **Save Result button** — saves annotated image to Drive
- **Live counts display** — Cars / People / Total at signal
- **Colour-coded annotation legend** — shown below every result

**Annotation rules enforced in GUI:**
```
🔴 Red box  = Blue car detected
🔵 Blue box = Any other car colour
🟢 Green box = Person at signal
```

---

## Special Technical Features

- **`protobuf==3.20.3` pin** (Step 0) — required to fix TensorFlow 2.12 import errors in Colab
- **`COLOUR_ALIASES` dict** — maps non-standard folder names (gray, cyan, maroon, etc.) to the 8 target classes
- **`map_folder_to_colour()`** — strips numeric prefixes (`01_black` → `black`) for cross-dataset compatibility
- **Synthetic dataset fallback** — generates 300 colour-labelled images per class with noise+gradient if the download fails
- **Video processing (Block 9)** — processes uploaded MP4/AVI video frame-by-frame, saves annotated output video

---

## Project Structure

```
Task5_Car_Colour_Detection/
├── Task5_Car_Colour_UPDATED.ipynb     ← Main Colab notebook
├── models/
│   ├── detector_model.h5              ← Trained Detection CNN weights
│   └── colour_model.h5                ← Trained Colour CNN weights
└── Outputs/
    ├── sample_colour_images.png
    ├── detection_training_history.png
    ├── colour_training_history.png
    ├── final_summary_dashboard.png
    ├── annotated_<filename>.jpg
    ├── counts_<filename>.png
    └── annotated_traffic.mp4
```

---

## How to Run (Google Colab)

1. Open `Task5_Car_Colour_UPDATED.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run **Step 0** first → fixes protobuf
4. **Restart runtime** (Runtime → Restart Runtime)
5. Run **Block 1** → installs dependencies
6. Run **Block 2** → mounts Drive
7. Run **Block 3** → downloads/generates colour dataset
8. Run **Block 4** → model justification + builds both CNNs
9. Run **Blocks 5 & 6** → trains Detection CNN and Colour CNN
10. Run **Block 7** → loads models, builds inference pipeline
11. Run **Block 8** → launches ipywidgets GUI for image analysis
12. Run **Block 9** (optional) → upload and process a traffic video
13. Run **Block 10** → final evaluation report + summary dashboard

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| TensorFlow 2.12 / Keras | Model building and training |
| OpenCV | Image reading, resizing, sliding window |
| NumPy | Array operations |
| Matplotlib | Training curves, dashboard, annotations |
| Seaborn | Confusion matrix heatmaps |
| scikit-learn | Train/val/test split, metrics |
| ipywidgets | Interactive Colab GUI |
| Pillow | Image encoding for GUI display |
| gdown | Dataset download from Google Drive |
| Google Colab + GPU | Training environment |
| Google Drive | Dataset and model storage |
