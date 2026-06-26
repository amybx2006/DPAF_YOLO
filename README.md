# DPAF-YOLO: 小目标检测优化方案

基于 YOLO11 的小目标检测改进模型，集成了 PPA (Position-Priority Attention) 和 DASI (Dynamic Adaptive Scale Interaction) 模块，显著提升小目标检测性能。

## �� 核心亮点

- **PPA (Position-Priority Attention)**：位置优先级注意力机制，增强对小目标特征的捕捉能力
- **DASI (Dynamic Adaptive Scale Interaction)**：动态自适应尺度交互模块，优化多尺度特征融合
- **ASFF (Adaptively Spatial Feature Fusion)**：自适应空间特征融合，提升特征表达能力
- **单类别检测优化**：针对小目标检测场景专门优化

## �� 项目结构

```
DPAF_YOLO/
├── ultralytics/
│   ├── ultralytics/
│   │   ├── cfg/
│   │   │   ├── models/
│   │   │   │   └── 11/
│   │   │   │       ├── yolo11s.yaml          # 原始 YOLO11s 模型
│   │   │   │       ├── yolo11s_DASI.yaml     # DASI 模块版本
│   │   │   │       └── yolo11s_PPA_DASI.yaml # PPA + DASI 完整版本 (核心模型)
│   │   │   └── datasets/
│   │   │       └── DUT.yaml                  # 数据集配置文件
│   │   ├── runs/                             # 训练结果目录
│   │   │   ├── weights/
│   │   │   │   ├── best.pt                   # 最佳权重
│   │   │   │   └── last.pt                   # 最后训练权重
│   │   │   └── args.yaml                     # 训练参数记录
│   │   └── ...
│   ├── train.py                              # 训练入口
│   ├── predict.py                            # 推理入口
│   └── pyproject.toml                        # 项目依赖
└── README.md                                 # 项目说明文档
```

## ��️ 环境要求

```bash
# 安装依赖
cd ultralytics
pip install -e .

# 或直接安装 ultralytics
pip install ultralytics>=8.2.0
```

## �� 数据集准备

### 数据集结构

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

### 创建 DUT.yaml 配置文件

在 `ultralytics/ultralytics/cfg/datasets/` 目录下创建 `DUT.yaml`：

```yaml
path: ../datasets/DUT  # 数据集根目录
train: train/images     # 训练集图片路径
val: val/images         # 验证集图片路径
test: test/images       # 测试集图片路径

# 类别定义
names:
  0: small_object  # 小目标类别

# 类别数量
nc: 1
```

## �� 训练与评估

### 训练命令

```bash
# 使用默认参数训练
cd ultralytics
python train.py --model ultralytics/cfg/models/11/yolo11s_PPA_DASI.yaml \
                --data ultralytics/cfg/datasets/DUT.yaml \
                --epochs 150 \
                --batch 32 \
                --imgsz 640 \
                --optimizer Adam \
                --device 0

# 恢复训练
python train.py --resume runs/detect/train89/weights/last.pt
```

### 关键训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--epochs` | 150 | 训练轮数 |
| `--batch` | 32 | 批次大小 |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--optimizer` | Adam | 优化器 |
| `--device` | 0 | GPU 设备编号 |
| `--lr0` | 0.001 | 初始学习率 |
| `--close_mosaic` | 10 | 最后 10 轮关闭 Mosaic 增强 |

### 评估命令

```bash
# 验证模型
python val.py --model runs/detect/train89/weights/best.pt \
              --data ultralytics/cfg/datasets/DUT.yaml

# 生成混淆矩阵和指标报告
python val.py --model runs/detect/train89/weights/best.pt \
              --data ultralytics/cfg/datasets/DUT.yaml \
              --save_json
```

## �� 推理预测

```bash
# 单张图片推理
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source test.jpg \
                  --save-crop

# 批量图片推理
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source test_images/ \
                  --save-txt

# 视频推理
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source input.mp4 \
                  --save-video
```

## ⚠️ 重要说明

### 数据集配置

使用前请确保已创建 `ultralytics/ultralytics/cfg/datasets/DUT.yaml` 数据集配置文件，并根据实际数据集路径进行修改。

### 权重文件

训练结果权重保存在 `ultralytics/runs/weights/` 目录：
- `best.pt`: 验证集性能最佳的权重
- `last.pt`: 最后一轮训练的权重

### 训练参数记录

`ultralytics/runs/args.yaml` 文件记录了训练时的所有参数配置，路径已修改为相对路径便于共享。

## �� 模型架构

```
输入图像 (640x640)
        \u2193
    Backbone
        \u251c\u2500\u2500 Conv + PPA (P1/2)
        \u251c\u2500\u2500 C3k2 + PPA (P2/4)
        \u251c\u2500\u2500 C3k2 + PPA (P3/8)
        \u251c\u2500\u2500 C3k2 (P4/16)
        \u2514\u2500\u2500 C3k2 + SPPF + C2PSA (P5/32)
        \u2193
    Neck (ASFF + DASI)
        \u251c\u2500\u2500 ASFF2 (P3-P4 融合)
        \u251c\u2500\u2500 ASFF3 (P3-P5 融合)
        \u2514\u2500\u2500 DASI (动态尺度交互)
        \u2193
    Head (Detect)
        \u2514\u2500\u2500 输出检测结果
```

## �� 训练结果

训练过程中生成的可视化结果保存在 `ultralytics/runs/` 目录：
- `results.png`: 训练损失曲线
- `F1_curve.png`: F1 曲线
- `PR_curve.png`: PR 曲线
- `train_batch*.jpg`: 训练批次可视化
- `val_batch*_labels.jpg`: 验证集标签可视化
- `val_batch*_pred.jpg`: 验证集预测结果可视化

## �� 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{DPAF-YOLO,
  title={DPAF-YOLO: Position-Priority Attention with Dynamic Adaptive Scale Interaction for Small Object Detection},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## �� 许可证

本项目基于 [AGPL-3.0](ultralytics/LICENSE) 许可证。# DPAF-YOLO: 小目标检测优化方案

基于 YOLO11 的小目标检测改进模型，集成了 PPA (Position-Priority Attention) 和 DASI (Dynamic Adaptive Scale Interaction) 模块，显著提升小目标检测性能。

## �� 核心亮点

- **PPA (Position-Priority Attention)**：位置优先级注意力机制，增强对小目标特征的捕捉能力
- **DASI (Dynamic Adaptive Scale Interaction)**：动态自适应尺度交互模块，优化多尺度特征融合
- **ASFF (Adaptively Spatial Feature Fusion)**：自适应空间特征融合，提升特征表达能力
- **单类别检测优化**：针对小目标检测场景专门优化

## �� 项目结构

```
DPAF_YOLO/
├── ultralytics/
│   ├── ultralytics/
│   │   ├── cfg/
│   │   │   ├── models/
│   │   │   │   └── 11/
│   │   │   │       ├── yolo11s.yaml          # 原始 YOLO11s 模型
│   │   │   │       ├── yolo11s_DASI.yaml     # DASI 模块版本
│   │   │   │       └── yolo11s_PPA_DASI.yaml # PPA + DASI 完整版本 (核心模型)
│   │   │   └── datasets/
│   │   │       └── DUT.yaml                  # 数据集配置文件
│   │   ├── runs/                             # 训练结果目录
│   │   │   ├── weights/
│   │   │   │   ├── best.pt                   # 最佳权重
│   │   │   │   └── last.pt                   # 最后训练权重
│   │   │   └── args.yaml                     # 训练参数记录
│   │   └── ...
│   ├── train.py                              # 训练入口
│   ├── predict.py                            # 推理入口
│   └── pyproject.toml                        # 项目依赖
└── README.md                                 # 项目说明文档
```

## ��️ 环境要求

```bash
# 安装依赖
cd ultralytics
pip install -e .

# 或直接安装 ultralytics
pip install ultralytics>=8.2.0
```

## �� 数据集准备

### 数据集结构

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

### 创建 DUT.yaml 配置文件

在 `ultralytics/ultralytics/cfg/datasets/` 目录下创建 `DUT.yaml`：

```yaml
path: ../datasets/DUT  # 数据集根目录
train: train/images     # 训练集图片路径
val: val/images         # 验证集图片路径
test: test/images       # 测试集图片路径

# 类别定义
names:
  0: small_object  # 小目标类别

# 类别数量
nc: 1
```

## �� 训练与评估

### 训练命令

```bash
# 使用默认参数训练
cd ultralytics
python train.py --model ultralytics/cfg/models/11/yolo11s_PPA_DASI.yaml \
                --data ultralytics/cfg/datasets/DUT.yaml \
                --epochs 150 \
                --batch 32 \
                --imgsz 640 \
                --optimizer Adam \
                --device 0

# 恢复训练
python train.py --resume runs/detect/train89/weights/last.pt
```

### 关键训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--epochs` | 150 | 训练轮数 |
| `--batch` | 32 | 批次大小 |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--optimizer` | Adam | 优化器 |
| `--device` | 0 | GPU 设备编号 |
| `--lr0` | 0.001 | 初始学习率 |
| `--close_mosaic` | 10 | 最后 10 轮关闭 Mosaic 增强 |

### 评估命令

```bash
# 验证模型
python val.py --model runs/detect/train89/weights/best.pt \
              --data ultralytics/cfg/datasets/DUT.yaml

# 生成混淆矩阵和指标报告
python val.py --model runs/detect/train89/weights/best.pt \
              --data ultralytics/cfg/datasets/DUT.yaml \
              --save_json
```

## �� 推理预测

```bash
# 单张图片推理
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source test.jpg \
                  --save-crop

# 批量图片推理
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source test_images/ \
                  --save-txt

# 视频推理
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source input.mp4 \
                  --save-video
```

## ⚠️ 重要说明

### 数据集配置

使用前请确保已创建 `ultralytics/ultralytics/cfg/datasets/DUT.yaml` 数据集配置文件，并根据实际数据集路径进行修改。

### 权重文件

训练结果权重保存在 `ultralytics/runs/weights/` 目录：
- `best.pt`: 验证集性能最佳的权重
- `last.pt`: 最后一轮训练的权重

### 训练参数记录

`ultralytics/runs/args.yaml` 文件记录了训练时的所有参数配置，路径已修改为相对路径便于共享。

## �� 模型架构

```
输入图像 (640x640)
        \u2193
    Backbone
        \u251c\u2500\u2500 Conv + PPA (P1/2)
        \u251c\u2500\u2500 C3k2 + PPA (P2/4)
        \u251c\u2500\u2500 C3k2 + PPA (P3/8)
        \u251c\u2500\u2500 C3k2 (P4/16)
        \u2514\u2500\u2500 C3k2 + SPPF + C2PSA (P5/32)
        \u2193
    Neck (ASFF + DASI)
        \u251c\u2500\u2500 ASFF2 (P3-P4 融合)
        \u251c\u2500\u2500 ASFF3 (P3-P5 融合)
        \u2514\u2500\u2500 DASI (动态尺度交互)
        \u2193
    Head (Detect)
        \u2514\u2500\u2500 输出检测结果
```

## �� 训练结果

训练过程中生成的可视化结果保存在 `ultralytics/runs/` 目录：
- `results.png`: 训练损失曲线
- `F1_curve.png`: F1 曲线
- `PR_curve.png`: PR 曲线
- `train_batch*.jpg`: 训练批次可视化
- `val_batch*_labels.jpg`: 验证集标签可视化
- `val_batch*_pred.jpg`: 验证集预测结果可视化

## �� 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{DPAF-YOLO,
  title={DPAF-YOLO: Position-Priority Attention with Dynamic Adaptive Scale Interaction for Small Object Detection},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## �� 许可证

本项目基于 [AGPL-3.0](ultralytics/LICENSE) 许可证。# DPAF-YOLO: 小目标检测优化方案

基于 YOLO11 的小目标检测改进模型，集成了 PPA (Position-Priority Attention) 和 DASI (Dynamic Adaptive Scale Interaction) 模块，显著提升小目标检测性能。

## �� 核心亮点

- **PPA (Position-Priority Attention)**：位置优先级注意力机制，增强对小目标特征的捕捉能力
- **DASI (Dynamic Adaptive Scale Interaction)**：动态自适应尺度交互模块，优化多尺度特征融合
- **ASFF (Adaptively Spatial Feature Fusion)**：自适应空间特征融合，提升特征表达能力
- **单类别检测优化**：针对小目标检测场景专门优化

## �� 项目结构

```
DPAF_YOLO/
├── ultralytics/
│   ├── ultralytics/
│   │   ├── cfg/
│   │   │   ├── models/
│   │   │   │   └── 11/
│   │   │   │       ├── yolo11s.yaml          # 原始 YOLO11s 模型
│   │   │   │       ├── yolo11s_DASI.yaml     # DASI 模块版本
│   │   │   │       └── yolo11s_PPA_DASI.yaml # PPA + DASI 完整版本 (核心模型)
│   │   │   └── datasets/
│   │   │       └── DUT.yaml                  # 数据集配置文件
│   │   ├── runs/                             # 训练结果目录
│   │   │   ├── weights/
│   │   │   │   ├── best.pt                   # 最佳权重
│   │   │   │   └── last.pt                   # 最后训练权重
│   │   │   └── args.yaml                     # 训练参数记录
│   │   └── ...
│   ├── train.py                              # 训练入口
│   ├── predict.py                            # 推理入口
│   └── pyproject.toml                        # 项目依赖
└── README.md                                 # 项目说明文档
```

## ��️ 环境要求

```bash
# 安装依赖
cd ultralytics
pip install -e .

# 或直接安装 ultralytics
pip install ultralytics>=8.2.0
```

## �� 数据集准备

### 数据集结构

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

### 创建 DUT.yaml 配置文件

在 `ultralytics/ultralytics/cfg/datasets/` 目录下创建 `DUT.yaml`：

```yaml
path: ../datasets/DUT  # 数据集根目录
train: train/images     # 训练集图片路径
val: val/images         # 验证集图片路径
test: test/images       # 测试集图片路径

# 类别定义
names:
  0: small_object  # 小目标类别

# 类别数量
nc: 1
```

## �� 训练与评估

### 训练命令

```bash
# 使用默认参数训练
cd ultralytics
python train.py --model ultralytics/cfg/models/11/yolo11s_PPA_DASI.yaml \
                --data ultralytics/cfg/datasets/DUT.yaml \
                --epochs 150 \
                --batch 32 \
                --imgsz 640 \
                --optimizer Adam \
                --device 0

# 恢复训练
python train.py --resume runs/detect/train89/weights/last.pt
```

### 关键训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--epochs` | 150 | 训练轮数 |
| `--batch` | 32 | 批次大小 |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--optimizer` | Adam | 优化器 |
| `--device` | 0 | GPU 设备编号 |
| `--lr0` | 0.001 | 初始学习率 |
| `--close_mosaic` | 10 | 最后 10 轮关闭 Mosaic 增强 |

### 评估命令

```bash
# 验证模型
python val.py --model runs/detect/train89/weights/best.pt \
              --data ultralytics/cfg/datasets/DUT.yaml

# 生成混淆矩阵和指标报告
python val.py --model runs/detect/train89/weights/best.pt \
              --data ultralytics/cfg/datasets/DUT.yaml \
              --save_json
```

## �� 推理预测

```bash
# 单张图片推理
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source test.jpg \
                  --save-crop

# 批量图片推理
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source test_images/ \
                  --save-txt

# 视频推理
python predict.py --model runs/detect/train89/weights/best.pt \
                  --source input.mp4 \
                  --save-video
```

## ⚠️ 重要说明

### 数据集配置

使用前请确保已创建 `ultralytics/ultralytics/cfg/datasets/DUT.yaml` 数据集配置文件，并根据实际数据集路径进行修改。

### 权重文件

训练结果权重保存在 `ultralytics/runs/weights/` 目录：
- `best.pt`: 验证集性能最佳的权重
- `last.pt`: 最后一轮训练的权重

### 训练参数记录

`ultralytics/runs/args.yaml` 文件记录了训练时的所有参数配置，路径已修改为相对路径便于共享。

## �� 模型架构

```
输入图像 (640x640)
        \u2193
    Backbone
        \u251c\u2500\u2500 Conv + PPA (P1/2)
        \u251c\u2500\u2500 C3k2 + PPA (P2/4)
        \u251c\u2500\u2500 C3k2 + PPA (P3/8)
        \u251c\u2500\u2500 C3k2 (P4/16)
        \u2514\u2500\u2500 C3k2 + SPPF + C2PSA (P5/32)
        \u2193
    Neck (ASFF + DASI)
        \u251c\u2500\u2500 ASFF2 (P3-P4 融合)
        \u251c\u2500\u2500 ASFF3 (P3-P5 融合)
        \u2514\u2500\u2500 DASI (动态尺度交互)
        \u2193
    Head (Detect)
        \u2514\u2500\u2500 输出检测结果
```

## �� 训练结果

训练过程中生成的可视化结果保存在 `ultralytics/runs/` 目录：
- `results.png`: 训练损失曲线
- `F1_curve.png`: F1 曲线
- `PR_curve.png`: PR 曲线
- `train_batch*.jpg`: 训练批次可视化
- `val_batch*_labels.jpg`: 验证集标签可视化
- `val_batch*_pred.jpg`: 验证集预测结果可视化

## �� 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{DPAF-YOLO,
  title={DPAF-YOLO: Position-Priority Attention with Dynamic Adaptive Scale Interaction for Small Object Detection},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## �� 许可证

本项目基于 [AGPL-3.0](ultralytics/LICENSE) 许可证。