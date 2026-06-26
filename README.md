# DPAF-YOLO: A Detail-Preserving Asymptotic Feature Fusion Network for Small UAV Detection in Low-Altitude Optical Remote Sensing

## Project Structure

```
DPAF_YOLO/
├── ultralytics/
│   ├── ultralytics/
│   │   ├── cfg/
│   │   │   ├── models/
│   │   │   │   └── 11/
│   │   │   │       ├── yolo11s.yaml          # Original YOLO11s model
│   │   │   │       ├── yolo11s_DASI.yaml     # DASI module version
│   │   │   │       └── yolo11s_PPA_DASI.yaml # PPA + DASI full version (core model)
│   │   │   └── datasets/
│   │   │       └── DUT.yaml                  # Dataset configuration file
│   │   ├── runs/                             # Training results directory
│   │   │   ├── weights/
│   │   │   │   ├── best.pt                   # Best weights
│   │   │   │   └── last.pt                   # Last training weights
│   │   │   └── args.yaml                     # Training parameters record
│   │   └── ...
│   ├── train.py                              # Training entry
│   ├── predict.py                            # Inference entry
│   └── pyproject.toml                        # Project dependencies
└── README.md                                 # Project documentation
```

## 🛠️ Environment Requirements

```bash
# Install dependencies
cd ultralytics
pip install -e .

# Or install ultralytics directly
pip install ultralytics>=8.2.0
```

## Dataset Preparation

### Dataset Structure

```
ultralytics/datasets/DUT/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Create DUT.yaml Configuration

Create `DUT.yaml` in `ultralytics/ultralytics/cfg/datasets/`:

```yaml
path: ../datasets/DUT  # Dataset root directory
train: train/images     # Training set images path
val: val/images         # Validation set images path
test: test/images       # Test set images path

# Class definitions
names:
  0: small_uav  # Small UAV class

# Number of classes
nc: 1
```

## Training and Evaluation

### Training Commands

```bash
# Train with default parameters
cd ultralytics
python train.py --model ultralytics/cfg/models/11/yolo11s_PPA_DASI.yaml \
                --data ultralytics/cfg/datasets/DUT.yaml \
                --epochs 150 \
                --batch 32 \
                --imgsz 640 \
                --optimizer Adam \
                --device 0

# Resume training
python train.py --resume runs/detect/train89/weights/last.pt
```

### Key Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--epochs` | 150 | Number of training epochs |
| `--batch` | 32 | Batch size |
| `--imgsz` | 640 | Input image size |
| `--optimizer` | Adam | Optimizer type |
| `--device` | 0 | GPU device index |
| `--lr0` | 0.001 | Initial learning rate |
| `--close_mosaic` | 10 | Close mosaic augmentation in last 10 epochs |

### Evaluation Commands

```bash
# Validate model
python val.py --model runs/detect/train89/weights/best.pt \
              --data ultralytics/cfg/datasets/DUT.yaml

# Generate confusion matrix and metrics report
python val.py --model runs/detect/train89/weights/best.pt \
              --data ultralytics/cfg/datasets/DUT.yaml \
              --save_json
```

## Inference and Prediction

```bash
# Single image inference
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source test.jpg \
                  --save-crop

# Batch image inference
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source test_images/ \
                  --save-txt

# Video inference
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source input.mp4 \
                  --save-video
```

## ⚠️ Important Notes

### Dataset Configuration

Ensure that `ultralytics/ultralytics/cfg/datasets/DUT.yaml` is created with the correct paths to your dataset before training.

### Weight Files

Training results are saved in `ultralytics/runs/weights/`:
- `best.pt`: Best performing weights on validation set
- `last.pt`: Last epoch training weights

### Training Parameters Record

The `ultralytics/runs/args.yaml` file records all training parameters. Paths have been converted to relative paths for easy sharing.

## Model Architecture

```
Input Image (640x640)
        ↓
    Backbone
        ├── Conv + PPA (P1/2)
        ├── C3k2 + PPA (P2/4)
        ├── C3k2 + PPA (P3/8)
        ├── C3k2 (P4/16)
        ├── C3k2 + SPPF + C2PSA (P5/32)
        ↓
    Neck (ASFF + DASI)
        ├── ASFF2 (P3-P4 Fusion)
        ├── ASFF3 (P3-P5 Fusion)
        └── DASI (Dynamic Scale Interaction)
        ↓
    Head (Detect)
        └── Output Detection Results
```

## Training Results

Visualization results are saved in `ultralytics/runs/`:
- `results.png`: Training loss curves
- `F1_curve.png`: F1 score curves
- `PR_curve.png`: Precision-Recall curves
- `train_batch*.jpg`: Training batch visualizations
- `val_batch*_labels.jpg`: Validation set label visualizations
- `val_batch*_pred.jpg`: Validation set prediction visualizations


## License

This project is licensed under the [AGPL-3.0](ultralytics/LICENSE) license. A Detail-Preserving Asymptotic Feature Fusion Network for Small UAV Detection in Low-Altitude Optical Remote Sensing
