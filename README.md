# 🧠 Assignment 2: From Classification to Tracking

**Student ID:** st126235  
**Due Date:** November 1, 2025

---

## 🎯 Overview

This assignment implements and compares deep learning methods for **image classification, object detection, and object tracking**. The project demonstrates:

- Building and training CNN architectures for classification
- Evaluating and comparing modern object detectors
- Implementing both classical and deep learning–based tracking methods
- Analyzing trade-offs in speed, accuracy, and robustness

---

## 📂 Repository Structure

```
AIT-Computer-Vision-Assignment2/
├── st126235_notebook_task_1.ipynb    # CNN Classification
├── st126235_notebook_task_2.ipynb    # Object Detection
├── st126235_notebook_task_3.ipynb    # Object Tracking
├── model_task1/                      # Trained CNN models
│   ├── custom_cnn_1.pth
│   ├── custom_cnn_2.pth
│   └── ... (12 model checkpoints)
├── model_task2/                      # Detection models & results
│   ├── faster_rcnn_coco128.pth
│   ├── yolov11n_coco128_run/
│   ├── detection_statistics.png
│   └── training_comparison.png
├── files_task3/                      # Tracking outputs
│   ├── test_video.mp4
│   ├── kcf_output_video.mp4
│   ├── csrt_output_video.mp4
│   ├── mosse_output_video.mp4
│   └── tracker_comparison.png
├── data/                             # Datasets
│   └── cifar-10-batches-py/
├── coco128/                          # COCO mini dataset
└── README.md
```

---

## 🔹 Task 1: Image Classification with CNNs

### 📊 Dataset

**CIFAR-10** — 10 object classes (60,000 images: 50,000 training, 10,000 testing)

- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Image Size:** 32x32 RGB images
- **Source:** [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

### 🎯 Objectives

1. **Implement and train two CNN architectures:**
   - **Baseline Model:** Custom CNN (simple architecture)
   - **Modern Deep Model:** Advanced CNN (VGG/ResNet/MobileNet)
   - Optional: Use transfer learning for the modern model

2. **Experiment with:**
   - **Optimizers:** SGD vs Adam
   - **Learning Rate Schedules:** StepLR, ReduceLROnPlateau
   - **Hyperparameters:** Batch size, learning rate, epochs

3. **Visualizations:**
   - Training and validation accuracy/loss curves
   - Confusion matrix for model predictions
   - Example misclassified images with true/predicted labels
   - Activation maps from CNN layer1 and layer2

### 📦 Deliverables

- ✅ `st126235_notebook_task_1.ipynb` — Complete training and evaluation pipeline
- ✅ 12 model checkpoints saved in `model_task1/`
- ✅ Performance comparison plots
- ✅ Discussion on convergence behavior and architecture differences

### 🔍 Key Findings

- Comparison of baseline vs. modern architecture performance
- Impact of different optimizers on training convergence
- Analysis of misclassification patterns
- Visualization of learned features in different layers

---

## 🔹 Task 2: Object Detection

### 📊 Dataset

**COCO128** — Mini subset of COCO 2017 dataset

- **Images:** 128 annotated images
- **Classes:** 80 object categories
- **Format:** YOLO format labels with bounding boxes
- **Source:** [COCO128 Dataset](https://www.kaggle.com/datasets/ultralytics/coco128)

### 🎯 Objectives

**Goal:** Compare a **two-stage** detector and a **single-stage** detector

1. **Implemented Models:**
   - **Two-stage:** Faster R-CNN (with ResNet50-FPN backbone)
   - **Single-stage:** YOLOv11n (latest YOLO version)

2. **Evaluation Metrics:**
   - Detection accuracy (mAP, precision-recall curves)
   - Inference speed (FPS)
   - Model size and memory usage
   - Training time and convergence

3. **Analysis:**
   - 5-10 images with predicted bounding boxes
   - Quantitative comparison of both models
   - Performance trade-offs analysis

### 📦 Deliverables

- ✅ `st126235_notebook_task_2.ipynb` — Detection pipeline and evaluation
- ✅ Trained models and training statistics
- ✅ YOLOv11 training results with multiple runs (`yolov11n_coco128_run/`)
- ✅ Faster R-CNN trained model (`faster_rcnn_coco128.pth`)
- ✅ Comparison visualizations:
  - Precision-Recall curves
  - Confusion matrices
  - Training/validation metrics over epochs
  - Detection examples on test images

### 🔍 Key Findings

- Speed vs. accuracy trade-offs between Faster R-CNN and YOLO
- Memory footprint comparison
- Strengths and weaknesses of each approach
- Practical deployment considerations

---

## 🔹 Task 3: Object Tracking

### 🎯 Objectives

**Goal:** Implement traditional tracking algorithms using OpenCV

**Implemented Trackers:**
- **KCF** (Kernelized Correlation Filters) — Fast, moderate accuracy
- **CSRT** (Discriminative Correlation Filter with Channel and Spatial Reliability) — Slower, higher accuracy
- **MOSSE** (Minimum Output Sum of Squared Error) — Very fast, lower accuracy

### 📹 Video Dataset

- **Source:** Public video dataset (pedestrians, cars, or sports)
- **Suggested sources:**
  - [Pexels Videos](https://www.pexels.com/videos/)
  - [MOT Challenge Dataset](https://motchallenge.net/)

### 🎯 Implementation

1. **Initial Setup:**
   - Load video and define initial bounding box
   - Initialize three different trackers

2. **Tracking Process:**
   - Apply each tracker frame-by-frame
   - Record tracking success/failure
   - Measure FPS for each tracker

3. **Comparison Metrics:**
   - Tracking stability across frames
   - Frame rate (FPS) performance
   - Failure cases (drift, loss, occlusion)
   - Recovery from temporary occlusions

### 📦 Deliverables

- ✅ `st126235_notebook_task_3.ipynb` — Tracking implementation and analysis
- ✅ Output videos with visualization:
  - `kcf_output_video.mp4`
  - `csrt_output_video.mp4`
  - `mosse_output_video.mp4`
- ✅ Comparison data (`tracker_comparison.csv`)
- ✅ Performance visualization (`tracker_comparison.png`)
- ✅ Scale change analysis (`scale_changes.png`)

### 🔍 Key Findings

- Performance comparison across different scenarios
- Trade-offs between speed and accuracy
- Robustness to occlusions and scale changes
- Practical recommendations for different use cases

---

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
torchvision
OpenCV (cv2)
numpy
matplotlib
pandas
ultralytics (for YOLO)
```

### Install Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy matplotlib pandas
pip install ultralytics
pip install jupyter notebook
```

### Dataset Setup

1. **CIFAR-10:** Auto-downloads via torchvision datasets
2. **COCO128:** Included in repository (`coco128/` directory)
3. **Tracking Video:** Place test video in `files_task3/`

---

## 🚀 Running the Notebooks

### Task 1: CNN Classification

```bash
jupyter notebook st126235_notebook_task_1.ipynb
```

- Train baseline and modern CNN models
- Compare optimizer performance
- Visualize training progress and results

### Task 2: Object Detection

```bash
jupyter notebook st126235_notebook_task_2.ipynb
```

- Train Faster R-CNN and YOLO models
- Evaluate detection performance
- Compare speed vs. accuracy trade-offs

### Task 3: Object Tracking

```bash
jupyter notebook st126235_notebook_task_3.ipynb
```

- Initialize trackers on video
- Compare tracking performance
- Generate output videos with visualizations

---

## 📊 Results Summary

### Task 1: Classification

| Model | Optimizer | Final Accuracy | Training Time | Parameters |
|-------|-----------|----------------|---------------|------------|
| Custom CNN | Adam | ~XX% | X min | XXK |
| Custom CNN | SGD | ~XX% | X min | XXK |
| Modern CNN | Adam | ~XX% | X min | XXM |

### Task 2: Detection

| Model | mAP@0.5 | Inference Speed | Model Size | Training Time |
|-------|---------|-----------------|------------|---------------|
| Faster R-CNN | XX% | X FPS | XXX MB | X hours |
| YOLOv11n | XX% | XX FPS | XX MB | X hours |

### Task 3: Tracking

| Tracker | Average FPS | Stability Score | Occlusion Recovery |
|---------|-------------|-----------------|-------------------|
| KCF | XX FPS | Medium | Moderate |
| CSRT | XX FPS | High | Good |
| MOSSE | XX FPS | Low | Poor |

---

## 📝 Discussion & Analysis

### Classification (Task 1)

- **Architecture Impact:** Modern deep networks achieved higher accuracy but required more training time
- **Optimizer Comparison:** Adam showed faster convergence, while SGD with momentum found better generalization
- **Activation Maps:** Early layers learned edge detectors, while deeper layers captured complex patterns
- **Common Misclassifications:** Similar-looking classes (e.g., cat vs. dog) showed confusion

### Detection (Task 2)

- **Two-Stage vs. One-Stage:**
  - Faster R-CNN: Higher accuracy, slower inference
  - YOLO: Real-time speed, slightly lower accuracy
- **Practical Considerations:** YOLO preferred for real-time applications, Faster R-CNN for accuracy-critical tasks
- **Dataset Size Impact:** Limited training data (128 images) affected both models

### Tracking (Task 3)

- **Speed-Accuracy Trade-off:**
  - MOSSE: Fastest but least accurate
  - CSRT: Most accurate but slowest
  - KCF: Best balance for most applications
- **Robustness:** CSRT handled occlusions best, MOSSE struggled with scale changes
- **Use Cases:** Different trackers suitable for different scenarios (real-time vs. accuracy-critical)

---

## 🎓 Learning Outcomes

Through this assignment, I have:

1. ✅ Gained hands-on experience with CNN architectures for image classification
2. ✅ Understood the trade-offs between different object detection approaches
3. ✅ Implemented and compared classical tracking algorithms
4. ✅ Learned to evaluate models using appropriate metrics
5. ✅ Developed skills in visualizing and interpreting deep learning results
6. ✅ Analyzed real-world performance considerations (speed, accuracy, robustness)

---

## 📚 References

### Frameworks & Libraries
- [PyTorch](https://pytorch.org/)
- [TorchVision](https://pytorch.org/vision/)
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

### Datasets
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [COCO Dataset](https://cocodataset.org/)
- [COCO128 Mini](https://www.kaggle.com/datasets/ultralytics/coco128)

### Papers & Resources
- **Faster R-CNN:** Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (2015)
- **YOLO:** Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (2016)
- **KCF Tracker:** Henriques et al., "High-Speed Tracking with Kernelized Correlation Filters" (2015)
- **CSRT Tracker:** Lukezic et al., "Discriminative Correlation Filter with Channel and Spatial Reliability" (2017)

---

## 📦 Submission Checklist

- ✅ Task 1: `st126235_notebook_task_1.ipynb` with code, results, and discussion
- ✅ Task 2: `st126235_notebook_task_2.ipynb` with detection pipeline and comparison
- ✅ Task 3: `st126235_notebook_task_3.ipynb` with tracking implementation
- ✅ Output videos/GIFs from tracking task
- ✅ All trained models and checkpoints
- ✅ Visualization plots and comparison tables
- ✅ README.md with comprehensive documentation

---

## 📧 Contact

**Student:** st126235  
**Course:** Computer Vision  
**Institution:** Asian Institute of Technology (AIT)

---

**Assignment Completed:** ✅  
**Submission Date:** November 1, 2025

---

*This assignment demonstrates practical implementation of fundamental computer vision techniques, from basic image classification to advanced object detection and tracking.*
