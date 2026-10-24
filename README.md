自己标注的夜间道路车辆检测数据集的训练及应用 夜间自动驾驶检测数据集 低光照度度下道路车辆检测数据集的训练及应用 YOLO模型训练
@[toc]


数据集描述：
夜间车辆检测数据集，夜间道路车辆检测数据集 自制数据集。标注类别为car，yolo格式，约4万帧，0.1秒一帧，使用labelimg工具标注![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/155a5c02a375485b802a62a77bda1b34.png)


使用 **YOLOv8** 训练和推理一个自制的夜间车辆检测数据集（4万帧，标注类别为car），我们将从环境搭建、数据准备、模型训练、推理到评估提供完整的技术流程与代码。由于这是一个专门针对夜间车辆的目标检测任务，如何利用 YOLOv8 的强大能力来准确识别并定位车辆。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f04824534f8b4c7591d58ad94cbe3481.png)

### 数据集信息

- **类别数量**: 1 (car)
- **图像总数**: 约40,000张
- **标注格式**: YOLO 格式
- **帧率**: 0.1秒一帧
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0e593dd8197c4c1ab7be6b830dcfac6f.png)

### 目标
1. 搭建 YOLOv8 环境。
2. 配置数据集以适应 YOLOv8 训练。
3. 使用预训练模型进行迁移学习。
4. 完成模型训练、推理、评估全流程。
5. 导出模型用于部署。

---

## ✅ 一、环境配置

### 1. 安装 CUDA 驱动（GPU 加速）

确保你的 NVIDIA 显卡驱动已安装：

```bash
nvidia-smi
```

输出应显示 GPU 信息。推荐使用 **CUDA 11.8 或 12.1**。

---

### 2. 安装 Anaconda

下载并安装 [Anaconda](https://www.anaconda.com/products/distribution)（Python 3.9+ 推荐）。

---

### 3. 创建 Python 虚拟环境

```bash
conda create -n night_vehicle_yolo python=3.9
conda activate night_vehicle_yolo
```

---

### 4. 安装必要依赖

```bash
# 安装 PyTorch（以 CUDA 11.8 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 YOLOv8 官方库
pip install ultralytics opencv-python numpy matplotlib tqdm scikit-learn pandas pillow

# 可选：TensorBoard 可视化
pip install tensorboard

# 如果需要处理 VOC 格式
pip install lxml  # 用于解析 XML (VOC)
```

验证 GPU 是否可用：

```python
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

---

## ✅ 二、数据集结构与 `data.yaml` 配置

假设你将使用 YOLO 格式的标注文件，并且数据集已经按照 8:1:1 的比例划分（即训练集32,000张，验证集4,000张，测试集4,000张）。

### 数据目录结构

```
night_vehicle_dataset/
├── images/
│   ├── train/      # 训练集图像
│   ├── val/        # 验证集图像
│   └── test/       # 测试集图像
├── labels/
│   ├── train/      # 对应的 .txt 文件 (YOLO 格式)
│   ├── val/
│   └── test/
└── data.yaml       # 数据配置文件
```

### 创建 `data.yaml`

在项目根目录创建 `data.yaml` 文件：

```yaml
# data.yaml
train: ./night_vehicle_dataset/images/train
val: ./night_vehicle_dataset/images/val
test: ./night_vehicle_dataset/images/test  # 支持独立测试集评估

# 类别数量
nc: 1

# 英文类别名称
names:
  - car
```

---

## ✅ 三、调用 YOLOv8 官方预训练模型进行训练

考虑到该数据集是专门为夜间车辆检测设计的，建议使用 **YOLOv8m 或 YOLOv8l** 提升分类精度。

```python
from ultralytics import YOLO

# 加载官方预训练模型（推荐使用 yolov8m.pt）
model = YOLO('yolov8m.pt')  # 自动下载 ImageNet 预训练权重

# 开始训练
results = model.train(
    data='data.yaml',           # 数据配置文件
    epochs=100,                 # 训练轮数
    batch=64,                   # 根据显存调整（RTX 3090 可用 64）
    imgsz=640,                  # 输入尺寸（可尝试 800 提升小目标检测）
    optimizer='AdamW',          # 更稳定优化器
    lr0=0.001,                  # 初始学习率
    weight_decay=0.0005,
    momentum=0.937,

    # 数据增强（提升泛化能力）
    augment=True,
    hsv_h=0.015,                # 色调扰动
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,               # 随机旋转
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,                 # 左右翻转
    mosaic=1.0,                 # Mosaic 增强
    mixup=0.1,                  # MixUp
    copy_paste=0.1,             # Copy-Paste Augmentation

    # 正则化
    dropout=0.2,
    label_smoothing=0.1,        # 标签平滑，防止过拟合

    # 学习率调度
    cos_lr=True,                # 余弦退火学习率

    # 保存与日志
    project='runs/train',
    name='night_vehicle_yolov8m',
    save=True,
    save_period=10,
    exist_ok=False,

    # 缓存（内存充足时开启）
    cache=True
)
```

> **注意**：
> - 若显存不足，考虑改用 `yolov8s.pt` 或降低 `batch` 大小。
> - 尝试 `imgsz=800` 来提高对远处或较小车辆的检测性能。
> - `copy_paste` 和 `mixup` 增强技术有助于提高模型的鲁棒性。

---

## ✅ 四、推理代码（单图 / 批量 / 视频）

### 1. 单张图像推理

```python
from ultralytics import YOLO
from PIL import Image

# 加载最佳模型
model = YOLO('runs/train/night_vehicle_yolov8m/weights/best.pt')

# 推理
results = model('test_night_vehicle.jpg', conf=0.4)  # 设置置信度阈值

# 显示结果
for r in results:
    im_array = r.plot()  # 绘制边界框和标签
    im = Image.fromarray(im_array[..., ::-1])  # BGR to RGB
    im.show()
```

---

### 2. 批量图像推理

```python
results = model.predict(
    source='night_vehicle_dataset/images/test',
    save=True,
    project='runs/detect',
    name='night_vehicle_test_results',
    conf=0.4,
    imgsz=640
)
print("批量推理完成")
```

---

### 3. 视频检测（如监控视频）

```python
results = model.predict(
    source='night_vehicle_monitoring.mp4',
    save=True,
    project='runs/detect/video',
    name='night_vehicle_video_result',
    show=False,
    conf=0.4
)
print("视频检测完成")
```

---

### 4. 实时摄像头检测（OpenCV）

```python
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated_frame = results[0].plot()

    cv2.imshow('Night Vehicle Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✅ 五、模型评估（验证集 + 测试集）

### 1. 在验证集上评估

```python
metrics = model.val(
    data='data.yaml',
    split='val',
    batch=64,
    imgsz=640,
    save_json=False,
    project='runs/val',
    name='night_vehicle_eval'
)

print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")

# 每类 AP
for i, name in enumerate(model.names):
    print(f"{name} (AP@0.5): {metrics.box.ap[i]:.4f}")
```

---

### 2. 在测试集上独立评估（最终性能）

```python
# 使用 test split 进行最终性能评估
test_metrics = model.val(
    data='data.yaml',
    split='test',
    save_json=True,
    name='test_final'
)

print(f"Test mAP@0.5: {test_metrics.box.map50:.4f}")
print(f"Test Recall: {test_metrics.box.r:.4f}")
```

---

## ✅ 六、可视化与分析

### 1. 查看训练曲线

```bash
tensorboard --logdir runs/train/night_vehicle_yolov8m
```

### 2. 结果展示

可以使用 Matplotlib 或其他可视化工具进一步分析预测结果，包括但不限于绘制 PR 曲线、混淆矩阵等。

---

## ✅ 七、模型导出（ONNX / TensorRT / TFLite）

```python
# 导出为 ONNX（通用部署）
model.export(format='onnx', dynamic=True, opset=13, imgsz=640)

# 导出为 TensorRT（NVIDIA GPU 加速）
model.export(format='engine', half=True, dynamic=True)

# 导出为 TFLite（移动端）
model.export(format='tflite', int8=True)

# 导出为 CoreML（iOS）
model.export(format='coreml')
```

---


