# 📘 Aztec Glyph Element Detection System

**Complete Documentation for Training and Using Faster R-CNN to Detect Elements in Aztec Glyphs**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [What This Project Does](#what-this-project-does)
3. [Complete Workflow Diagram](#complete-workflow-diagram)
4. [Technical Architecture](#technical-architecture)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [Understanding the Results](#understanding-the-results)
7. [Performance and Timing](#performance-and-timing)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project trains deep learning models to automatically detect and classify individual elements within Aztec glyphs.

**The Problem:**
- Aztec glyphs contain multiple symbolic elements combined together
- Each element has meaning (e.g., "acatl" = reed, "pantli" = flag, "calli" = house)
- Manually identifying and cataloging these elements is time-consuming

**The Solution:**
- Train AI models to automatically detect elements in glyph images
- Identify which elements are present and where they're located
- Output: Element names, positions, and confidence scores

---

## 🔍 What This Project Does

### Input → Process → Output

```
INPUT:
══════════════════════════════════════════════════════
Individual Element Images (Training Data)
──────────────────────────────────────────────────────
Main_Elements/
├── acatl-element/          "Reed" symbol
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── pantli-element/         "Flag" symbol
│   ├── image1.png
│   └── ...
└── ... (31 element types, 1095 images total)

         ↓ ↓ ↓ ↓ ↓

PROCESS:
══════════════════════════════════════════════════════
1. Data Preparation (prepare_training_data.py)
   - Auto-detect bounding boxes around elements
   - Apply data augmentation (rotation, flip, brightness)
   - Balance dataset (ensure each class has enough samples)
   - Split into train/validation sets (80/20)

2. Model Training (train_faster_rcnn.py or train_yolo.py)
   - Load pre-trained model (transfer learning from COCO dataset)
   - Train on your 31 element types
   - Learn to recognize and locate elements
   - Save trained model checkpoints

         ↓ ↓ ↓ ↓ ↓

OUTPUT (Trained Model):
══════════════════════════════════════════════════════
output_faster_rcnn/
└── model_final.pth         Your trained "brain"


DEPLOYMENT:
══════════════════════════════════════════════════════
Input: Complete Glyph (contains multiple elements)
──────────────────────────────────────────────────────
     ┌─────────────────────┐
     │  ╔═══╗              │
     │  ║   ║ ← pantli     │
     │  ╚═══╝              │
     │                     │
     │   │││   ← acatl     │
     │   │││               │
     │                     │
     │  ┌───┐             │
     │  │   │ ← calli     │
     │  └───┘              │
     └─────────────────────┘

         ↓ ↓ ↓ ↓ ↓

Model Detection (detect_elements_in_glyph.py)
──────────────────────────────────────────────────────
✅ Detected 3 elements:
1. pantli-element  | 95% confidence | Location: [50, 20, 110, 60]
2. acatl-element   | 89% confidence | Location: [70, 80, 100, 130]
3. calli-element   | 92% confidence | Location: [45, 140, 100, 185]

Output Files:
├── detected_glyph.jpg      (visualization with bounding boxes)
└── detected_glyph.json     (structured detection data)
```

---

## 📊 Complete Workflow Diagram

### Phase 1: Data Preparation

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION PHASE                        │
└─────────────────────────────────────────────────────────────────┘

Step 1: Organize Training Data
────────────────────────────────────────────────────────────────
Main_Elements/
├── acatl-element/      (32 images)
├── ahuitzotl-element/  (32 images)
├── atl-element/        (55 images)
├── calli-element/      (39 images)
└── ... (31 element types)

                    ↓

Step 2: Run prepare_training_data.py
────────────────────────────────────────────────────────────────
For each image:
  1. Auto-detect bounding box (find non-white pixels)
  2. Convert to Detectron2 format: {
       'file_name': 'path/to/image.png',
       'image_id': 123,
       'height': 200,
       'width': 200,
       'annotations': [{
         'bbox': [x_min, y_min, x_max, y_max],
         'category_id': 5  # Element class ID
       }]
     }

For classes with < 40 images:
  3. Apply data augmentation:
     ├── Rotation: ±15°, ±30°
     ├── Flipping: horizontal, vertical
     ├── Brightness: ×0.8, ×1.2
     └── Noise: Gaussian noise

                    ↓

Step 3: Output Prepared Datasets
────────────────────────────────────────────────────────────────
detectron2_dataset/
├── train_dataset.json      (1095 training images)
├── val_dataset.json        (278 validation images)
└── metadata.json           (31 class names)

yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/

✅ Dataset ready for training!
```

### Phase 2: Model Training (Faster R-CNN)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING PHASE                          │
└─────────────────────────────────────────────────────────────────┘

Step 1: Initialize Model (train_faster_rcnn.py)
────────────────────────────────────────────────────────────────
Load Pre-trained Weights (COCO Dataset)
  ↓
  Model Architecture:

  ┌─────────────────────────────────────────────────────────┐
  │                   Faster R-CNN Model                     │
  ├─────────────────────────────────────────────────────────┤
  │                                                          │
  │  INPUT IMAGE (H × W × 3)                                │
  │         ↓                                                │
  │  ┌──────────────────────────────────────┐              │
  │  │ BACKBONE: ResNet-50 + FPN            │              │
  │  │ - Extracts visual features           │              │
  │  │ - 25 million parameters              │              │
  │  │ - Multi-scale feature pyramid        │              │
  │  └──────────────────────────────────────┘              │
  │         ↓                                                │
  │  ┌──────────────────────────────────────┐              │
  │  │ RPN (Region Proposal Network)        │              │
  │  │ - Proposes ~1000 object locations    │              │
  │  │ - "Where might elements be?"         │              │
  │  └──────────────────────────────────────┘              │
  │         ↓                                                │
  │  ┌──────────────────────────────────────┐              │
  │  │ ROI Align                            │              │
  │  │ - Extract features for each proposal │              │
  │  └──────────────────────────────────────┘              │
  │         ↓                                                │
  │  ┌──────────────────────────────────────┐              │
  │  │ BOX HEAD                             │              │
  │  │ - Classify: "Is this acatl/pantli?"  │              │
  │  │ - Refine: "Adjust bounding box"      │              │
  │  └──────────────────────────────────────┘              │
  │         ↓                                                │
  │  OUTPUT:                                                 │
  │  - Bounding boxes: [x1, y1, x2, y2]                    │
  │  - Class labels: "acatl-element", etc.                 │
  │  - Confidence scores: 0.0 - 1.0                        │
  └─────────────────────────────────────────────────────────┘

Modify for 31 Classes:
  - Change output layer: 80 classes (COCO) → 31 classes (your elements)
  - Re-initialize final classification layer

                    ↓

Step 2: Training Loop (5000 iterations)
────────────────────────────────────────────────────────────────
Each Iteration (~40 seconds on Apple Silicon M-series):

  1. Load batch of 2 images

  2. Forward pass:
     Image → ResNet-50 → FPN → RPN → ROI Align → Box Head

  3. Calculate losses:
     ├── loss_rpn_cls:  "Is this region an object?"
     ├── loss_rpn_loc:  "Where exactly is it?"
     ├── loss_cls:      "Which element type is it?"
     └── loss_box_reg:  "How to adjust the box?"

     total_loss = sum of all losses

  4. Backpropagation:
     - Update 25 million parameters
     - Make model better at detecting elements

  5. Every 500 iterations:
     - Save checkpoint: model_0000499.pth
     - Evaluate on validation set

  6. Repeat for next batch

Training Progress Example:
──────────────────────────────────────────────────────────────
iter:    0  | loss: 3.79  | ← Model making random guesses
iter:   59  | loss: 1.001 | ← Already learning!
iter:  500  | loss: 0.45  | ← Getting good (checkpoint saved)
iter: 1000  | loss: 0.32  | ← Very good
iter: 2000  | loss: 0.25  | ← Excellent
iter: 5000  | loss: 0.18  | ← Best performance

Loss decreasing = Model getting better! ✅

                    ↓

Step 3: Save Trained Model
────────────────────────────────────────────────────────────────
output_faster_rcnn/
├── model_0000499.pth   (checkpoint at 500 iterations)
├── model_0000999.pth   (checkpoint at 1000 iterations)
├── model_0001999.pth   (checkpoint at 2000 iterations)
└── model_final.pth     (final model at 5000 iterations)

✅ Training complete!
```

### Phase 3: Detection/Inference

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION/INFERENCE PHASE                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Load Trained Model
────────────────────────────────────────────────────────────────
python3 detect_elements_in_glyph.py \
  --image "Glyphs/my_glyph/image.png" \
  --model "output_faster_rcnn/model_final.pth"

                    ↓

Step 2: Process Glyph Image
────────────────────────────────────────────────────────────────
Input: Complete Glyph Image
  ┌────────────────────────────┐
  │                            │
  │    ╔═══╗                   │  Element A
  │    ║   ║                   │
  │    ╚═══╝                   │
  │                            │
  │      │││                   │  Element B
  │      │││                   │
  │                            │
  │    ┌───┐                   │  Element C
  │    │   │                   │
  │    └───┘                   │
  │                            │
  └────────────────────────────┘

Model processes image:
  1. Extract features (ResNet-50 backbone)
  2. Propose candidate locations (RPN)
  3. Classify each candidate (Box Head)
  4. Filter by confidence threshold (default: 50%)

                    ↓

Step 3: Output Detections
────────────────────────────────────────────────────────────────
Console Output:
──────────────────────────────────────────────────────────────
🔍 ANALYZING GLYPH: my_glyph.png
Image size: 400x600 pixels

✅ Detected 3 elements:
1. pantli-element  | Confidence: 95.23% | Location: [50, 20, 110, 60]
2. acatl-element   | Confidence: 89.47% | Location: [70, 80, 100, 130]
3. calli-element   | Confidence: 92.15% | Location: [45, 140, 100, 185]

💾 Saved visualization to: detected_my_glyph.jpg
💾 Saved detection data to: detected_my_glyph.json

Visualization (detected_my_glyph.jpg):
──────────────────────────────────────────────────────────────
  ┌────────────────────────────┐
  │  ┌─────────────┐           │
  │  │ pantli 95%  │           │  ← Bounding box + label
  │  │  ╔═══╗      │           │
  │  │  ║   ║      │           │
  │  │  ╚═══╝      │           │
  │  └─────────────┘           │
  │                            │
  │    ┌─────────────┐         │
  │    │ acatl 89%   │         │
  │    │   │││       │         │
  │    │   │││       │         │
  │    └─────────────┘         │
  │                            │
  │  ┌─────────────┐           │
  │  │ calli 92%   │           │
  │  │  ┌───┐      │           │
  │  │  │   │      │           │
  │  │  └───┘      │           │
  │  └─────────────┘           │
  └────────────────────────────┘

JSON Output (detected_my_glyph.json):
──────────────────────────────────────────────────────────────
{
  "image": "Glyphs/my_glyph/image.png",
  "num_detections": 3,
  "detections": [
    {
      "element": "pantli-element",
      "confidence": 0.9523,
      "bbox": {
        "x_min": 50,
        "y_min": 20,
        "x_max": 110,
        "y_max": 60,
        "width": 60,
        "height": 40
      }
    },
    {
      "element": "acatl-element",
      "confidence": 0.8947,
      "bbox": {
        "x_min": 70,
        "y_min": 80,
        "x_max": 100,
        "y_max": 130,
        "width": 30,
        "height": 50
      }
    },
    {
      "element": "calli-element",
      "confidence": 0.9215,
      "bbox": {
        "x_min": 45,
        "y_min": 140,
        "x_max": 100,
        "y_max": 185,
        "width": 55,
        "height": 45
      }
    }
  ]
}

✅ Detection complete!
```

### Batch Processing

```
┌─────────────────────────────────────────────────────────────────┐
│                    BATCH PROCESSING                              │
└─────────────────────────────────────────────────────────────────┘

Process Multiple Glyphs at Once:
────────────────────────────────────────────────────────────────
python3 detect_elements_batch.py \
  --input "Glyphs" \
  --output "detected_glyphs" \
  --model "output_faster_rcnn/model_final.pth"

Input Folder:
──────────────────────────────────────────────────────────────
Glyphs/
├── 0010-cuahuacalli/
│   └── glyph.png
├── 0010-acolhuaca/
│   └── glyph.png
├── 0010-apanecatl/
│   └── glyph.png
└── ... (hundreds of glyphs)

                    ↓

Processing:
──────────────────────────────────────────────────────────────
Processing glyphs: 100%|██████████| 150/150 [2:30<00:00, 1.0s/glyph]

                    ↓

Output:
──────────────────────────────────────────────────────────────
detected_glyphs/
├── 0010-cuahuacalli/
│   ├── glyph_detected.jpg    (visualization)
│   └── glyph_detected.json   (detection data)
├── 0010-acolhuaca/
│   ├── glyph_detected.jpg
│   └── glyph_detected.json
├── ...
└── detection_summary.json    (overall summary)

Summary (detection_summary.json):
──────────────────────────────────────────────────────────────
{
  "total_images": 150,
  "total_detections": 487,
  "average_detections_per_glyph": 3.24,
  "results": [...]
}

✅ Batch processing complete!
```

---

## 🏗️ Technical Architecture

### Model: Faster R-CNN

**What is Faster R-CNN?**
- **R-CNN** = Region-based Convolutional Neural Network
- **Purpose**: Object detection (find and classify multiple objects in images)
- **Two-stage detector**:
  1. **Stage 1 (RPN)**: Propose candidate object locations
  2. **Stage 2 (Box Head)**: Classify and refine each proposal

**Architecture Components:**

```
┌──────────────────────────────────────────────────────────┐
│                    Faster R-CNN                           │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. BACKBONE: ResNet-50 + FPN                            │
│     ├── ResNet-50: Deep CNN with 50 layers              │
│     ├── Parameters: ~25 million                          │
│     └── FPN: Feature Pyramid Network (multi-scale)      │
│                                                           │
│  2. RPN (Region Proposal Network)                        │
│     ├── Scans image at multiple scales                  │
│     ├── Proposes ~1000 candidate object locations       │
│     ├── Output: Boxes + "objectness" scores             │
│     └── Losses: loss_rpn_cls + loss_rpn_loc             │
│                                                           │
│  3. ROI ALIGN                                            │
│     ├── Extract fixed-size features for each proposal   │
│     ├── 7×7 feature map per proposal                    │
│     └── Maintains spatial alignment                     │
│                                                           │
│  4. BOX HEAD                                             │
│     ├── 2 fully connected layers (1024 neurons each)    │
│     ├── Classification: Which of 31 element types?      │
│     ├── Box regression: Refine bounding box             │
│     └── Losses: loss_cls + loss_box_reg                 │
│                                                           │
│  TOTAL LOSS = loss_rpn_cls + loss_rpn_loc +             │
│               loss_cls + loss_box_reg                    │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**Training Strategy:**

1. **Transfer Learning**:
   - Start with weights pre-trained on COCO dataset (80 object classes)
   - Fine-tune on your 31 element types
   - Advantages: Faster training, better accuracy with limited data

2. **Data Augmentation**:
   - Rotation: ±15°, ±30°
   - Flipping: horizontal, vertical
   - Brightness: ×0.8, ×1.2
   - Noise: Gaussian noise
   - Purpose: Increase dataset diversity, prevent overfitting

3. **Multi-scale Training**:
   - Input sizes: 200, 240, 280, 320, 360, 400 pixels
   - Purpose: Handle elements at different scales

4. **Learning Rate Schedule**:
   - Base LR: 0.00025
   - Decay at: 2000, 4000 iterations
   - Optimizer: SGD with momentum

### Why Faster R-CNN?

| Feature | Faster R-CNN | YOLO | Notes |
|---------|--------------|------|-------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Better for small objects |
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | YOLO faster for inference |
| **Small Objects** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Better feature pyramid |
| **Training Time** | Similar | Similar | Both use transfer learning |
| **Precision** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | More accurate boxes |

**For this project:**
- Faster R-CNN chosen for better accuracy on small glyph elements
- YOLO also available (train_yolo.py) for faster inference if needed

---

## 📖 Step-by-Step Guide

### Prerequisites

**Hardware:**
- CPU: Any modern processor (training will be slow)
- GPU: Apple M-series (MPS), NVIDIA GPU (CUDA), or CPU
- RAM: 8GB minimum, 16GB recommended
- Storage: 5GB free space

**Software:**
- macOS, Linux, or Windows
- Python 3.11-3.13
- Git (optional)

### Step 1: Setup Environment

```bash
# Navigate to project directory
cd /path/to/AI_clinic

# Activate virtual environment
source myenv/bin/activate  # macOS/Linux
# or
myenv\Scripts\activate     # Windows

# Verify installations
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'GPU (MPS): {torch.backends.mps.is_available()}')"
python3 -c "import detectron2; print('Detectron2: OK')"
```

**Expected output:**
```
PyTorch: 2.9.1
GPU (MPS): True
Detectron2: OK
```

### Step 2: Prepare Training Data

**Your data structure:**
```
Main_Elements/
├── acatl-element/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── pantli-element/
└── ... (31 element folders)
```

**Run data preparation:**
```bash
python3 prepare_training_data.py
```

**What happens:**
- Scans all element folders
- Auto-detects bounding boxes
- Applies data augmentation to classes with < 40 images
- Splits into train (80%) and validation (20%)
- Creates datasets for both Faster R-CNN and YOLO

**Output:**
```
detectron2_dataset/
├── train_dataset.json  (1095 images)
├── val_dataset.json    (278 images)
└── metadata.json

yolo_dataset/
├── train/
└── val/

✅ Dataset preparation complete!
Time: 2-5 minutes
```

### Step 3: Train Model

**Option A: Faster R-CNN (Recommended)**

```bash
python3 train_faster_rcnn.py
```

**Configuration options** (edit train_faster_rcnn.py line 382-390):
```python
trainer.setup_config(
    train_dataset_name="glyph_train",
    val_dataset_name="glyph_val",
    base_lr=0.00025,       # Learning rate
    max_iter=1000,         # Training iterations (adjust based on time)
    batch_size=2,          # Images per iteration
    num_workers=2,         # Data loading workers
    checkpoint_period=500, # Save checkpoint every 500 iterations
    eval_period=500        # Evaluate every 500 iterations
)
```

**Training time estimates:**

| Iterations | Time (Apple M-series) | Expected Accuracy |
|-----------|----------------------|-------------------|
| 500 | ~5.5 hours | 60-70% (quick test) |
| 1000 | ~11 hours | 70-80% (good) |
| 2000 | ~22 hours | 80-85% (very good) |
| 5000 | ~55 hours | 85-90% (best) |

**Monitor training:**
```
[11/13 10:04:39] eta: 1 day, 22:30:19  iter: 19  total_loss: 3.79
[11/13 10:18:23] eta: 2 days, 6:37:26  iter: 39  total_loss: 1.201
[11/13 10:31:43] eta: 2 days, 7:04:45  iter: 59  total_loss: 1.001

Loss decreasing = Model learning! ✅
```

**Checkpoints saved at:**
```
output_faster_rcnn/
├── model_0000499.pth   (500 iterations)
├── model_0000999.pth   (1000 iterations)
└── model_final.pth     (final model)
```

**Option B: YOLO (Faster inference)**

```bash
python3 train_yolo.py
```

Similar training process, saves to `runs/detect/train/weights/best.pt`

### Step 4: Test Model on Glyphs

**Single glyph detection:**
```bash
python3 detect_elements_in_glyph.py \
  --image "Glyphs/my_glyph/image.png" \
  --model "output_faster_rcnn/model_final.pth" \
  --confidence 0.5
```

**Options:**
- `--image`: Path to glyph image
- `--model`: Path to trained model (use checkpoints for early testing)
- `--confidence`: Threshold (0.0-1.0, default 0.5)
- `--save-json`: Save detection data as JSON

**Batch processing:**
```bash
python3 detect_elements_batch.py \
  --input "Glyphs" \
  --output "detected_glyphs" \
  --model "output_faster_rcnn/model_final.pth"
```

### Step 5: Analyze Results

**Visualization (detected_glyph.jpg):**
- Bounding boxes around detected elements
- Element names as labels
- Confidence scores

**JSON data (detected_glyph.json):**
```json
{
  "detections": [
    {
      "element": "pantli-element",
      "confidence": 0.95,
      "bbox": {"x_min": 50, "y_min": 20, ...}
    }
  ]
}
```

**Use cases:**
- Catalog elements in large glyph collections
- Statistical analysis of element frequency
- Automated element extraction
- Research and documentation

---

## 🎯 Understanding the Results

### Detection Output Explained

**Bounding Box:**
```
bbox: [x_min, y_min, x_max, y_max]
      [50,    20,    110,   60   ]
```
- `x_min, y_min`: Top-left corner (in pixels)
- `x_max, y_max`: Bottom-right corner
- Width = x_max - x_min
- Height = y_max - y_min

**Confidence Score:**
- Range: 0.0 to 1.0 (0% to 100%)
- Interpretation:
  - 0.95+ : Very confident (likely correct)
  - 0.70-0.95: Confident (usually correct)
  - 0.50-0.70: Uncertain (may be false positive)
  - < 0.50: Not shown (filtered out)

**Element Names:**
- Format: "{element}-element"
- Example: "acatl-element", "pantli-element", "calli-element"
- Total: 31 different element types

### Quality Metrics

**mAP (mean Average Precision):**
- Industry standard metric for object detection
- Range: 0-100%
- Interpretation:
  - 50-60%: Basic detection working
  - 60-75%: Good performance
  - 75-85%: Very good performance
  - 85-90%: Excellent performance
  - 90%+: State-of-the-art (rare)

**Training Loss vs. Iterations:**
```
Loss
  │
4 │●
  │
3 │  ●
  │    ●
2 │      ●
  │        ●●
1 │          ●●●●
  │              ●●●●●●●●●●●●●
0 │________________________●●●●●
  └────────────────────────────> Iterations
  0   500  1000 1500 2000  5000

Lower loss = Better model
```

### Common Detection Scenarios

**Scenario 1: Perfect Detection**
```
Input:  Glyph with 3 clear elements
Output: 3 detections, all correct, confidence > 90%
Result: ✅ Model working perfectly
```

**Scenario 2: Missed Detection**
```
Input:  Glyph with 3 elements
Output: Only 2 detected
Reason: Element too small, occluded, or rare
Fix:    Train longer, add more training data for that element
```

**Scenario 3: False Positive**
```
Input:  Glyph with 2 elements
Output: 3 detections (1 incorrect)
Reason: Model seeing patterns that look like elements
Fix:    Increase confidence threshold (--confidence 0.7)
```

**Scenario 4: Wrong Classification**
```
Input:  Contains "acatl" element
Output: Detected as "pantli" element
Reason: Similar visual appearance, needs more training
Fix:    Train longer, add more diverse training examples
```

---

## ⏱️ Performance and Timing

### Training Performance

**Hardware Impact:**

| Hardware | Iterations/Hour | Time to 1000 iters | Time to 5000 iters |
|----------|----------------|-------------------|-------------------|
| **Apple M1/M2 (MPS)** | ~90 | 11 hours | 55 hours |
| **NVIDIA RTX 3080** | ~300 | 3.3 hours | 16 hours |
| **NVIDIA A100** | ~600 | 1.7 hours | 8 hours |
| **CPU Only** | ~20 | 50 hours | 250 hours |

**Your current setup (Apple M-series):**
- Time per iteration: ~40 seconds
- 1000 iterations: ~11 hours
- 5000 iterations: ~55 hours (2.3 days)

**Optimization tips:**
1. Reduce `max_iter`: 1000-2000 often sufficient
2. Reduce `batch_size`: If running out of memory
3. Use checkpoints: Test at 500 iterations, continue if needed
4. Multi-GPU: Not applicable for single M-series Mac

### Inference Performance

**Detection speed (per glyph):**

| Hardware | Time per Glyph | Glyphs per Hour |
|----------|---------------|-----------------|
| **Apple M1/M2 (MPS)** | 1-2 seconds | 1800-3600 |
| **NVIDIA GPU** | 0.5-1 second | 3600-7200 |
| **CPU** | 3-5 seconds | 720-1200 |

**Batch processing:**
- 100 glyphs: ~2-3 minutes
- 1000 glyphs: ~20-30 minutes
- 10000 glyphs: ~3-5 hours

---

## 🔧 Troubleshooting

### Installation Issues

**Problem: SSL Certificate Error**
```
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```

**Solution:**
```bash
# Install SSL certificates
/Applications/Python\ 3.13/Install\ Certificates.command

# Or manually:
pip3 install --upgrade certifi
```

---

**Problem: Detectron2 Installation Fails**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Install PyTorch FIRST, then Detectron2
pip3 install torch torchvision torchaudio
pip3 install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

---

**Problem: MPS (GPU) Not Available**
```
MPS Available: False
```

**Solution:**
- Ensure you have Apple Silicon (M1/M2/M3)
- Update macOS to latest version
- Update PyTorch: `pip3 install --upgrade torch`
- If on Intel Mac, GPU won't be available (use CPU)

---

### Training Issues

**Problem: Training Very Slow**
```
iter: 19  time: 300 seconds
```

**Solution:**
- Check GPU is being used: Look for "Training on Apple Silicon GPU (MPS)"
- Reduce batch size if out of memory
- Expected: ~40 seconds per iteration on M-series Mac

---

**Problem: Loss Not Decreasing**
```
iter: 500  loss: 3.50
iter: 1000 loss: 3.48
```

**Solution:**
- Check data: Verify prepared datasets are correct
- Reduce learning rate: Change `base_lr=0.00025` to `base_lr=0.0001`
- Check data augmentation: May be too aggressive
- Verify class balance: All elements should have ~30+ images

---

**Problem: Loss = NaN**
```
iter: 150  loss: nan
```

**Solution:**
- Reduce learning rate (try 0.0001)
- Check data: Corrupted images or invalid annotations
- Reduce batch size
- Restart training from scratch

---

**Problem: Out of Memory**
```
RuntimeError: MPS backend out of memory
```

**Solution:**
```python
# Edit train_faster_rcnn.py, line 387
batch_size=1,  # Reduce from 2 to 1
```

---

### Detection Issues

**Problem: No Elements Detected**
```
✅ Detected 0 elements
```

**Solution:**
- Lower confidence threshold: `--confidence 0.3`
- Check model path is correct
- Verify training completed successfully
- Test with training images first (should detect perfectly)

---

**Problem: Too Many False Positives**
```
✅ Detected 15 elements (expected 3)
```

**Solution:**
- Increase confidence threshold: `--confidence 0.7`
- Train longer for better discrimination
- Check if elements are too similar visually

---

**Problem: Wrong Classifications**
```
Expected: acatl-element
Detected: pantli-element
```

**Solution:**
- Train longer (more iterations)
- Add more diverse training examples
- Check if elements are visually similar
- Verify training data labels are correct

---

### Data Issues

**Problem: prepare_training_data.py Fails**
```
FileNotFoundError: Main_Elements not found
```

**Solution:**
- Verify folder structure:
  ```
  Main_Elements/
  ├── acatl-element/
  ├── pantli-element/
  └── ...
  ```
- Check folder names end with "-element"
- Ensure images are .png, .jpg, or .bmp format

---

**Problem: Class Imbalance Warning**
```
Warning: Class 'rare-element' has only 5 images
```

**Solution:**
- Data augmentation will automatically create more
- Target: 40+ images per class
- If < 5 images, consider removing class or collecting more data

---

## 📚 Additional Resources

### Element Reference

**31 Aztec Glyph Elements:**
1. acatl-element (reed)
2. ahuitzotl-element (water creature)
3. atl-element (water)
4. calli-element (house)
5. chimalli-element (shield)
6. cohuatl-element (serpent)
7. cuauhtli-element (eagle)
8. huehuetl-element (drum)
9. huitzilin-element (hummingbird)
10. icpalli-element (seat)
11. ihuitl-element (feather)
12. ilhuitl-element (day)
13. maitl_1-element (hand)
14. micqui-element (death)
15. mitl-element (arrow)
16. nochtli-element (prickly pear)
17. nopalli-element (cactus)
18. ocelotl-element (jaguar)
19. pantli-element (flag)
20. petlatl-element (mat)
21. piqui-element (tobacco)
22. popoca-element (smoke)
23. tecpan-element (palace)
24. tecpatl-element (flint)
25. tepotzoicpalli-element (hunchback seat)
26. tetl-element (stone)
27. tilmatli-element (cloak)
28. tlatoa-element (speak)
29. tochtli-element (rabbit)
30. xayacatl-element (face)
31. xiuhuitzollin-element (turquoise diadem)

### File Reference

**Key Scripts:**
- `prepare_training_data.py`: Data preparation and augmentation
- `train_faster_rcnn.py`: Train Faster R-CNN model
- `train_yolo.py`: Train YOLO model (alternative)
- `detect_elements_in_glyph.py`: Single glyph detection
- `detect_elements_batch.py`: Batch glyph processing
- `benchmark_models.py`: Compare model performance

**Key Folders:**
- `Main_Elements/`: Training data (31 element folders)
- `Glyphs/`: Complete glyphs for detection
- `detectron2_dataset/`: Prepared training data
- `output_faster_rcnn/`: Trained models and checkpoints
- `detected_glyphs/`: Detection results

**Configuration:**
- `requirements.txt`: Python dependencies
- `QUICK_START.md`: Quick reference guide

---

## 📞 Support and Contact

**Common Questions:**

Q: **How accurate will my model be?**
A: With 1000-2000 iterations, expect 70-85% mAP. Accuracy depends on training data quality and quantity.

Q: **Can I add more element types?**
A: Yes! Add new folders to `Main_Elements/`, re-run `prepare_training_data.py`, and train again.

Q: **How do I resume training?**
A: Edit `train_faster_rcnn.py` line 228: Change `resume=False` to `resume=True`.

Q: **Can I use this on Windows?**
A: Yes! GPU support requires NVIDIA GPU with CUDA. Otherwise, use CPU (slower).

Q: **How much training data do I need?**
A: Minimum: 20-30 images per element type. Recommended: 40-100 images per type.

---

## 🎓 Understanding Deep Learning Concepts

### What is Transfer Learning?

**Traditional approach:**
```
Train from scratch → Requires millions of images → Months of training
```

**Transfer learning approach:**
```
Pre-trained model (COCO) → Fine-tune on your data → Hours of training
```

**Benefits:**
- Faster training (hours vs. days/weeks)
- Better accuracy with limited data
- Learns general features (edges, shapes) from COCO
- Only needs to learn specific element patterns

### What is Data Augmentation?

**Problem:** Limited training data (30-50 images per element)

**Solution:** Create variations of existing images

**Augmentations applied:**
1. **Rotation**: ±15°, ±30° → Learn elements at different angles
2. **Flipping**: Horizontal, vertical → Learn mirrored elements
3. **Brightness**: ×0.8, ×1.2 → Handle different lighting
4. **Noise**: Gaussian noise → Handle image quality variations

**Result:** 32 images → 40 images per class

### What are Bounding Boxes?

**Bounding box** = Rectangle around an object

```
Format: [x_min, y_min, x_max, y_max]

Visual:
  (x_min, y_min)
      ┌─────────────┐
      │             │
      │   ELEMENT   │
      │             │
      └─────────────┘
                (x_max, y_max)

Coordinates are in pixels from top-left corner of image
```

**Why bounding boxes?**
- Simple and efficient
- Industry standard for object detection
- Sufficient for identifying element locations

### Loss Function Explained

**Loss** = How wrong the model is

**Components:**

1. **loss_rpn_cls**: "Is this region an object or background?"
   - High = Model can't distinguish objects from background
   - Low = Model accurately identifies object regions

2. **loss_rpn_loc**: "Where exactly is the object?"
   - High = Proposed boxes are far from true locations
   - Low = Accurate box proposals

3. **loss_cls**: "Which element type is this?"
   - High = Model confusing element types
   - Low = Accurate classification

4. **loss_box_reg**: "How to refine the bounding box?"
   - High = Boxes don't fit elements tightly
   - Low = Precise bounding boxes

**Training goal:** Minimize total_loss

```
Iteration 0:    total_loss = 3.79  (random guesses)
Iteration 500:  total_loss = 0.45  (learning)
Iteration 2000: total_loss = 0.25  (good)
Iteration 5000: total_loss = 0.18  (excellent)
```

---

## ✅ Summary

**What you have:**
- 31 Aztec glyph element types
- ~1095 training images (after augmentation)
- Faster R-CNN model architecture
- Apple Silicon GPU (MPS) for training

**What you're building:**
- AI model to detect elements in complete glyphs
- Automatic element identification and localization
- JSON data for further analysis

**Complete workflow:**
```
1. Prepare data (2-5 min)
   ↓
2. Train model (5-55 hours depending on iterations)
   ↓
3. Test on glyphs (1-2 sec per glyph)
   ↓
4. Analyze results (element names, locations, confidence)
```

**Expected results:**
- 70-85% detection accuracy with 1000-2000 iterations
- ~2 seconds per glyph for detection
- Automated processing of large glyph collections

**Next steps:**
1. Let training complete (or stop at 500-1000 iterations for testing)
2. Test model on real glyphs
3. Evaluate accuracy
4. Resume training if needed
5. Process your glyph collection!

---

**Documentation Version:** 1.0
**Last Updated:** November 13, 2025
**Project:** Aztec Glyph Element Detection
**Model:** Faster R-CNN with ResNet-50 + FPN
