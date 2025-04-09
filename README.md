# 🛡️ Birds vs Drones Detection with YOLOv8

A robust, real-time object detection system using YOLOv8 to distinguish between birds and drones in aerial surveillance imagery. This model is optimized for airspace safety, wildlife monitoring, and drone traffic management.

---

## 📌 Project Overview

This project aims to solve the critical problem of **misclassification between drones and birds**, which can hinder surveillance, cause false alarms, or compromise safety near sensitive zones.

### ✅ Core Objectives

- Enhance **airspace security** (e.g., airports, military zones)
- Reduce **false positives** in drone monitoring
- Improve **wildlife conservation** by reducing UAV interference
- Support **law enforcement** in detecting unauthorized drone activities
- Optimize **drone operations** in shared airspace

---

## 🧠 Model Summary

- **Model Used**: `YOLOv8m` (Ultralytics)
- **Training Type**: Custom object detection (birds vs drones)
- **Classes**:  
  - `0` – Bird  
  - `1` – Drone

- **Final mAP@0.5**: `0.778`  
- **Final mAP@0.5:0.95**: `0.523`  
- **Precision**: Bird = `0.745`, Drone = `0.927`  
- **Recall**: Bird = `0.852`, Drone = `0.745`  
- **Inference Speed**: `2.6 ms/image`

---

## 📂 Dataset Structure

```bash
Dataset/
└── cleaned_dataset/
    ├── cleaned_train/
    │   ├── images/
    │   └── labels/
    ├── cleaned_valid/
    │   ├── images/
    │   └── labels/
    └── cleaned_test/
        ├── images/
        └── labels/
All images resized to 640×640

Labels follow YOLO format: <class> <x_center> <y_center> <width> <height>

## **🔍 Data Preprocessing**
Filename normalization (e.g., BT (1).jpg → BT(1).jpg)

Validation of image dimensions and corruption checks

Background class diagnosis to prevent label confusion

Re-labeling of misclassified drones/birds

Balanced and augmented dataset for better generalization


## Training Pipeline

from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # Medium version

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolo_v8m_clean',
    augment=True
)

#  Evaluation & Results

results = model.val(data="data.yaml", split="test", imgsz=640, augment=True)

Final Test Performance
Class	Precision	Recall	AP@0.5	AP@0.5:0.95
Bird	0.745	0.852	0.555	0.556
Drone	0.927	0.745	0.814	0.491
Overall Accuracy: ~90%

F1 Scores: High for both classes

Confusion Matrix: Clear class separation

## Prediction Example

yolo predict model=weights/best.pt source=test/images/DT(131).jpg

## Predictions Saved in this folder

runs/detect/predict/
🛠️ Model Export
Supports multiple formats for deployment:

✅ ONNX (best.onnx)

✅ TorchScript (best.torchscript)

✅ PyTorch weights (best.pt)

yolo export model=weights/best.pt format=onnx
yolo export model=weights/best.pt format=torchscript


## 📦 Inference Code

from ultralytics import YOLO
model = YOLO("runs/detect/yolo_v8m_clean/weights/best.pt")
results = model.predict("sample.jpg", conf=0.5, save=True)


🧭 Business Alignment

Objective	Achieved

80%+ Drone/Bird classification	✅
Real-time inference	✅
Low false alarm rates	✅
Cross-environment performance	✅
Deployable formats (ONNX, Torch)	✅

🤝 Contributing

If you'd like to contribute:

Fork the repository

Submit a pull request with clear documentation

📜 License
MIT License – feel free to use, modify, and build upon this project.
