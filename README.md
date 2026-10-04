# Bridge Detection over Water Bodies using YOLOv8-OBB

## Deep Learning-Based Aerial Bridge Detection under Real-World Computational Constraints

Remote Sensing • Computer Vision • Oriented Object Detection • Deep Learning • Aerial AI

---

# Overview

This project presents an end-to-end deep learning pipeline for detecting bridges over water bodies in high-resolution aerial and satellite imagery using YOLOv8-Oriented Bounding Boxes (YOLOv8-OBB).

The project initially explored the GLH-Bridge dataset, a very high-resolution bridge-specific dataset containing images up to 16,384×16,384 pixels. However, large-scale preprocessing and training proved infeasible under GPU and storage limitations on Kaggle and Google Colab environments.

The project therefore pivoted to DOTA v1.0, a benchmark aerial object detection dataset with pre-tiled 1024×1024 crops optimized for GPU-constrained training.

The repository has two notebooks:

* `dota_yolov8_obb.ipynb`: YOLOv8n-OBB and YOLOv8s-OBB on DOTA v1.0 (oriented boxes, all 15 classes, bridge AP reported separately)
* `glh_water_bridges_yolov8s.ipynb`: water-aware filtering of GLH-Bridge annotations, then an axis-aligned YOLOv8s trained on the filtered bridges, downscaled to 640 px

The report (`Bridge_Detection_Report.pdf`) and slides (`Bridge_Detection_Presentation.pptx`) cover both.

---

# Problem Statement

Detecting bridges in aerial imagery is a difficult computer vision problem because:

* Bridges are thin and elongated structures
* Bridges appear at arbitrary orientations
* Bridges occupy very few pixels in large aerial images
* Bridges are visually similar to roads
* Bridge instances are rare compared to vehicles and ships

Additionally, very large remote sensing datasets introduce practical constraints:

* GPU memory exhaustion
* CUDA runtime failures
* Disk quota limitations
* Expensive preprocessing pipelines

This project investigates how oriented object detection and augmentation strategies can improve bridge detection performance under realistic hardware limitations.

---

# Dataset Journey: GLH-Bridge → DOTA v1.0

## Initial Dataset: GLH-Bridge

The project initially selected the GLH-Bridge dataset because of its bridge-specific focus and extremely high-resolution imagery.

### GLH Dataset Characteristics

* ~6,000 bridge images
* Resolution range:

  * 2,048×2,048
  * up to 16,384×16,384
* Approximate dataset size:

  * ~50 GB

---

## Practical Challenges Encountered

### 1. GPU Memory Constraints

Training on 16K images caused repeated:

* CUDA Out-of-Memory (OOM) errors
* Runtime crashes
* VRAM exhaustion on Tesla T4 GPUs

Even batch size 1 was unstable.

---

### 2. Disk Space Limitations

Kaggle and Colab environments provide limited storage.

Issues encountered:

* Dataset exceeded working directory quotas
* Intermediate preprocessing files increased storage pressure
* Tiling and downsampling pipelines became infeasible

---

### 3. Preprocessing Bottleneck

Downsampling extremely large images:

* is computationally expensive
* risks losing fine bridge structures
* requires large preprocessing infrastructure

---

## What the GLH run did

Full-resolution training was not feasible, so `glh_water_bridges_yolov8s.ipynb` trains on GLH-Bridge downscaled to 640 px with an axis-aligned YOLOv8s, after keeping only bridges over water (see [Water-Aware Filtering](#water-aware-filtering)): 3,463 train and 1,145 val images. Validation mAP@50 was still rising (0.10 at epoch 1, about 0.33 by epoch 60) when the notebook's output ends, partway through epoch 61. The saved output has no final evaluation.

## Pivot to DOTA v1.0

For oriented boxes on pre-tiled imagery, the project moved to DOTA v1.0 because it provides:

* Pre-tiled 1024×1024 crops
* GPU-ready dataset structure
* Oriented Bounding Box (OBB) annotations
* Benchmark-quality aerial imagery
* Manageable preprocessing requirements

---

# DOTA v1.0 Dataset

## Dataset Overview

| Property         | Value                   |
| ---------------- | ----------------------- |
| Total Images     | 2,806                   |
| Categories       | 15                      |
| Annotation Type  | Oriented Bounding Boxes |
| Tile Size        | 1024×1024               |
| Train Split      | 1,411                   |
| Validation Split | 458                     |

---

## Bridge Class Challenge

Bridges are heavily underrepresented:

| Split      | Bridge Instances |
| ---------- | ---------------- |
| Training   | 2,047            |
| Validation | 464              |

Only ~15% of training tiles contain bridges (210 of 1,411).

This severe class imbalance significantly impacts bridge AP.

---

# Methodology

## Overall Pipeline

```text
GLH Exploration / DOTA Dataset
                ↓
Dataset Engineering
                ↓
Water-Aware Annotation Filtering
                ↓
OBB Label Processing
                ↓
YOLOv8n / YOLOv8s Training
                ↓
Augmentation Pipeline
                ↓
Inference & Confidence Analysis
                ↓
Bridge Detection Evaluation
```

---

# Water-Aware Filtering

Used on GLH-Bridge (`glh_water_bridges_yolov8s.ipynb`). A custom filtering strategy retains only bridge-over-water annotations.

## Process

Bounding box regions were sampled using RGB heuristics:

* Blue dominance
* Dark neutral tones
* Teal water signatures

Annotations were retained only if:

```text
Water Pixels ≥ 28%
```

On GLH-Bridge this kept 26,460 of 32,873 training bridges (3,463 images) and 8,995 of 11,722 validation bridges (1,145 images).

---

# Data Augmentation

Because bridge instances are rare, the DOTA runs use heavy augmentation (the YOLOv8s run adds copy-paste).

## Techniques Used

### Rotation Augmentation

* Full 360° rotational robustness

### Mosaic Augmentation

* Combines multiple scenes into one training sample

### Copy-Paste Augmentation

* Artificially increases bridge frequency

### Flipping

* Horizontal and vertical orientation invariance

### Scaling & Translation

* Simulates altitude and positional variations

---

# Model Architecture

## YOLOv8-OBB

The project uses YOLOv8's Oriented Bounding Box variant.

---

## Why OBB?

Normal bounding boxes are axis-aligned.

Bridges are:

* rotated
* elongated
* directionally variable

OBB predicts:

```text
(x_center, y_center, width, height, angle)
```

This enables tighter localization for aerial structures.

---

## Architecture Components

### Backbone — CSPDarkNet

Extracts:

* edges
* textures
* spatial features

---

### Neck — PAN/FPN

Combines:

* low-level features
* high-level semantic information

Enables:

* multi-scale detection

---

### Detection Head

Predicts:

* object class
* confidence score
* rotated bounding boxes
* orientation angle

---

# Model Variants

| Model       | Parameters | Role     |
| ----------- | ---------- | -------- |
| YOLOv8n-OBB | 3.1M       | Baseline |
| YOLOv8s-OBB | 11.4M      | Improved |

---

# Training Configuration

| Parameter  | YOLOv8n   | YOLOv8s   |
| ---------- | --------- | --------- |
| Epochs     | 50        | 100       |
| Batch Size | 8         | 8         |
| Image Size | 1024×1024 | 1024×1024 |
| Copy-Paste | 0.0       | 0.3       |
| Mosaic     | 1.0       | 1.0       |

---

# Results

## Quantitative Performance (DOTA v1.0 val, all classes)

From the final validation pass of each run in `dota_yolov8_obb.ipynb`:

| Metric    | YOLOv8n | YOLOv8s |
| --------- | ------- | ------- |
| mAP@50    | 51.6%   | 54.9%   |
| mAP@50-95 | 34.1%   | 37.1%   |
| Precision | 73.8%   | 77.7%   |
| Recall    | 50.4%   | 53.3%   |

**What these numbers measure.** Both runs start from Ultralytics' `yolov8n-obb.pt` / `yolov8s-obb.pt`, which are already trained on DOTA v1.0. In both, the best validation epoch was epoch 1, and early stopping ended training after 11 (n) and 21 (s) epochs. So the reported weights are the pretrained checkpoints after one epoch of fine-tuning, and further training with these augmentations didn't improve them. The notebook doesn't evaluate the untouched checkpoints, so how much of the score that one epoch adds isn't measured; the n → s gap is mostly model size.

---

# Bridge-Specific Performance

## Bridge AP@50

| YOLOv8n | YOLOv8s |
| ------- | ------- |
| 16.1%   | 19.2%   |

---

# Why is Bridge AP Low?

Several reasons contribute:

## 1. Severe Class Imbalance

Bridges represent <2% of all DOTA instances.

---

## 2. Thin Object Geometry

Small localization errors drastically reduce IoU.

---

## 3. Visual Similarity to Roads

Roads and bridges appear structurally similar in aerial imagery.

---

## 4. Limited Feature Resolution

Bridges occupy very few CNN feature cells.

---

## 5. IoU Sensitivity

For tiny elongated objects:

$$IoU = \frac{Area\ of\ Overlap}{Area\ of\ Union}$$

Even slight misalignment causes large IoU drops.

---

# Confidence Analysis

The improved YOLOv8s model produced:

* fewer total detections
* higher confidence detections
* fewer false positives

## Key Insight

The larger backbone becomes more selective and produces higher-quality detections.

---

# Key Research Insights

## Dataset Engineering Matters

Dataset selection itself became a core methodological challenge.

---

## OBB Suits Elongated Structures

Rotated boxes fit thin, diagonal bridges far more tightly than axis-aligned boxes. The two notebooks use different datasets, so they don't measure this directly.

---

## Training Stopped Improving After Epoch 1

On DOTA, neither run improved after epoch 1. Testing whether augmentation helps the rare bridge class needs a model that isn't already trained on DOTA, or a held-out bridge set.

---

## Computational Constraints Affect Model Performance

Model performance depends not only on architecture, but also on:

* GPU availability
* preprocessing feasibility
* storage infrastructure

---

# Future Work

## Planned Improvements

* Fine-tune from COCO weights (or freeze the backbone) so the DOTA runs measure the training choices, not the pretrained checkpoint
* Finish the GLH run and evaluate it
* YOLOv8m / YOLOv8l upgrade
* SegFormer-based water segmentation
* Multi-class infrastructure detection
* Multi-temporal change detection
* Supplementary dataset integration
* HPC-based GLH training
* Multi-scale inference pipelines

---

# Technologies Used

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Kaggle Notebooks
* CUDA

---

# Hardware

| Hardware         | Usage                     |
| ---------------- | ------------------------- |
| Tesla T4 GPU     | Training                  |
| Kaggle Notebooks | Primary Environment       |
| Google Colab     | Supplementary Experiments |

---

# Repository Structure

```text
dota_yolov8_obb.ipynb               # DOTA v1.0: YOLOv8n-OBB vs YOLOv8s-OBB, bridge AP, plots
glh_water_bridges_yolov8s.ipynb     # GLH-Bridge: water-aware filtering + YOLOv8s training
Bridge_Detection_Report.pdf         # project report
Bridge_Detection_Presentation.pptx  # slides
README.md
```

Datasets, weights and run outputs aren't in the repo. The notebooks download DOTA v1.0 through Ultralytics and read GLH-Bridge from a Kaggle dataset.

---

# Citation

```bibtex
@misc{bridge_detection_yolov8_obb,
  title={Bridge Detection over Water Bodies using YOLOv8-OBB},
  author={Arnav Deshpande, Apoorv Singh, Yash Dodiya},
  year={2026}
}
```

---

# Authors

Arnav Deshpande
Indian Institute of Technology Indore

Apoorv Singh
Indian Institute of Technology Indore

Yash Dodiya
Indian Institute of Technology Indore

---

# Research Areas

* Computer Vision
* Remote Sensing
* Aerial Object Detection
* Oriented Bounding Boxes
* Deep Learning
* Infrastructure Detection
* Geospatial AI

---

# Objective

The primary objective of this project is to develop a robust deep learning pipeline for bridge detection in aerial imagery while simultaneously studying the impact of dataset engineering, computational constraints, and oriented object detection on real-world deployment feasibility.
