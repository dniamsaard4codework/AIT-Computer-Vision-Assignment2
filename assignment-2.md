# 🧠 Assignment 2: From Classification to Tracking

**Due Date:** Nov 1, 2025

---

## 🎯 Overview

In this assignment, you will implement and compare deep learning methods for **image classification, object detection, and object tracking**.  
By the end of this assignment, you should be able to:

- Build and train CNN architectures for classification
- Evaluate and compare modern object detectors
- Implement both classical and deep learning–based tracking methods
- Analyze trade-offs in speed, accuracy, and robustness

---

## 🔹 Image Classification with CNNs

### Task 1: Implement and Compare CNN Architectures

**Dataset Options:**

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 10 object classes (60,000 images)
- [Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet) — 200 object classes (64x64 images)

**Instructions:**

1. Implement and train **two CNN architectures** using `Pytorch`:
   - A simple baseline (e.g., custom CNN or LeNet)
   - A modern deep model (e.g., VGG, ResNet, or MobileNet (optional: use `transfer` learning)
2. Experiment with:
   - Optimizers: **SGD vs Adam**
   - Learning rate schedules (e.g., `StepLR`, `ReduceLROnPlateau`)
3. Visualize:
   - Training and validation accuracy/loss curves
   - Confusion matrix
   - Example misclassified images
   - Example activation maps from CNN layer1 and layer2 of each model.

**Deliverables:**

- Notebook with model training and evaluation
- Plots and visualizations
- Short discussion comparing performance and convergence behavior

---

## 🔹 Object Detection

### Task 2: Apply and Analyze Object Detection Models

**Goal:** Compare a **two-stage** detector and a **single-stage** detector.

**Recommended Models:**

- Two-stage: [Faster R-CNN (TorchVision)](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
- Single-stage: YOLO v`x` (`x` = version of your choice)

**Dataset Options:**

- [COCO 2017 (Mini subset)](https://www.kaggle.com/datasets/ultralytics/coco128) — 128 annotated images
- [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)

**Instructions:**

1. Run both detectors on the same dataset.
2. Measure:
   - Detection accuracy (mAP or precision-recall curves)
   - Inference speed (FPS)
   - Model size and memory usage
3. Visualize 5–10 images with predicted bounding boxes.
4. Optionally, test reduced input resolutions or lightweight model variants.

**Deliverables:**

- Notebook with detection and evaluation pipeline
- Quantitative comparison table
- Example detection results
- Short discussion on performance trade-offs

---

## 🔹 Object Tracking

### Task 3: Classical Object Tracking using OpenCV

**Goal:** Implement traditional tracking algorithms using OpenCV.

**Suggested Trackers:**

- KCF
- CSRT
- MOSSE

**Instructions:**

1. Use a **short video** (e.g., pedestrians, cars, or sports).
   - You can download public videos from [Pexels Videos](https://www.pexels.com/videos/) or [MOT Challenge dataset](https://motchallenge.net/).
2. Define an initial bounding box.
3. Apply **two different OpenCV trackers** (e.g., KCF and CSRT).
4. Compare:
   - Tracking stability
   - Frame rate (FPS)
   - Failure cases (drift, loss, occlusion)

**Deliverables:**

- Output video or GIF with tracking visualization
- Comparison table (FPS, success rate, drift cases)
- Short discussion on performance trade-offs

---

## 📦 Submission Instructions

- Submit a folder containing:

  1. 3 notebooks for each task, and name it as`<your_studentID>_notebook_task_{#}.ipynb` where `# = task number`. Each notebook should contain your code, example results, comparison table, and short discussion.
  2. Output videos or GIFs

- Zip the folder and submit on TEAL

**End of Assignment**
