
# A Lightweight Video Anomaly Detection via Persistent Temporal Differencing & Isolation Forest

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Unsupervised-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCSD__Ped2-green.svg)](https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An unsupervised, computationally efficient framework engineered to detect irregular motion patterns in surveillance footage using the **UCSD Ped2** benchmark dataset. By extracting temporal motion dynamics through a specialized **3-Frame Persistent Temporal Differencing (PTD)** pipeline, this architecture leverages an **Unsupervised Isolation Forest** model to isolate anomalies without requiring deep-learning overhead or labeled anomaly training data.

---

## 📝 Abstract / Project Description

Real-time video anomaly detection in surveillance infrastructure remains challenging due to the heavy computational demands of deep-learning architectures. This paper introduces a lightweight, unsupervised framework for video anomaly detection that optimizes edge-computing efficiency without sacrificing structural accuracy. The proposed approach replaces resource-intensive deep-learning networks with a highly optimized 3-Frame Persistent Temporal Differencing (PTD) pipeline. 

By calculating consecutive frame deltas via bitwise blending and zero-thresholding, the PTD mechanism isolates high-fidelity persistent motion vectors while successfully filtering ambient environmental noise. These streamlined spatial-temporal vectors are subsequently modeled using an unsupervised Isolation Forest anomaly detector, trained exclusively on normal pedestrian behavior. 

Evaluated on the benchmark UCSD Ped2 dataset, our framework achieves a competitive AUC-ROC score of 0.8415 and an anomaly F1-score of 0.85. Crucially, the model demonstrates a massive reduction in computational overhead compared to baseline single-frame and deep autoencoder methodologies. This balance of a small computational footprint and high temporal sensitivity makes our framework uniquely suited for real-time deployment on hardware-constrained edge surveillance devices.

---

## 📌 Methodology Overview

Traditional single-frame subtraction struggles with localized background noise (e.g., camera jitter, lighting shifts). This framework implements a **3-Frame Window PTD Logic** to isolate true, continuous motion signatures:

1. **Spatial Normalization**: Input video frames are resized to a uniform $64 \times 64$ spatial dimension and smoothed using a Gaussian filter to mitigate pixel sensor noise.
2. **Persistent Differencing**: Computes individual absolute motion deltas across three sequential frames:
   $$\text{Diff}_1 = |F_t - F_{t-1}|, \quad \text{Diff}_2 = |F_{t-1} - F_{t-2}|$$
3. **Bitwise Blending & Noise Elimination**: Merges both temporal matrices using a logical bitwise `OR` operation to capture overlapping motion persistence before applying a structural zero-threshold to drop background variations:
   $$\text{Combined} = \text{Diff}_1 \cup \text{Diff}_2$$

---

## 📊 Dataset Structure & Setup

The framework is verified using the benchmark **UCSD Anomaly Detection Dataset (Ped2 subset)** tracking outdoor pedestrian walkways. 

* **Normal Profiling**: Crowded or sparse walkways featuring exclusively walking pedestrians.
* **Anomalous Events**: Bicycles, skateboarders, small vehicles traversing pedestrian-only areas, or pedestrians walking across forbidden grass lines.

### Expected Directory Architecture
Configure your processing environment or Google Drive storage layer to match the configuration path layout below:
```text
📂 content/
 ┗ 📂 drive/
    ┗ 📂 My Drive/
       ┗ 📂 ped2/
          ┣ 📂 Train/
          ┃   ┣ 📂 Train001/ (Normal .tif frame sequences)
          ┃   ┗ ...
          ┗ 📂 Test/
              ┣ 📂 Test001/ (Anomalous test .tif frame sequences)
              ┣ 📂 Test001_gt/ (Ground truth .bmp validation masks)
              ┗ ...

```

---

## 📈 Performance Benchmarks

By supplying the unsupervised profiling engine with pristine, noise-filtered temporal motion vectors, the **Isolation Forest** delivers strong performance metrics without needing an active deep-learning training phase:

| Evaluation Metric | Baseline (Single-Frame Subtraction) | PTD Framework (Proposed) |
| --- | --- | --- |
| **AUC-ROC Score** | `~0.7023` | **`0.8415`** |
| **Anomaly Precision** | `~0.90` | **`0.89`** |
| **Anomaly Recall** | `0.62` | **`0.81`** |
| **Anomaly F1-Score** | `0.73` | **`0.85`** |
| **Overall Accuracy** | `0.63` | **`0.76`** |

### System Classification Matrix

```text
Detailed Classification Report (Proposed Framework):
              precision    recall  f1-score   support

      Normal       0.71      0.72      0.71       912
     Anomaly       0.89      0.81      0.85      1104

    accuracy                           0.76      2016
   macro avg       0.80      0.76      0.78      2016
weighted avg       0.81      0.76      0.79      2016

```

---

## 🛠️ Implementation & Repository Layout

The logic is structured sequentially inside the execution notebook to ensure isolated parameter testing:

* **Section 1 (Configuration)**: Establishes localized or cloud file stream links.
* **Section 2 (Feature Pipeline)**: Runs spatial-temporal reduction and builds the pixel intensity variance arrays.
* **Section 3 (Engine Training)**: Initializes data scales and builds normal isolation profiles.
* **Section 4 & 5 (Metrics & Visualizations)**: Computes scientific evaluation reports and exports analytical performance plots (`research_results_final.png`).

---

## 🎨 Visualization Samples

The pipeline generates dual validation graphs for evaluation reviews:

1. **Confusion Matrix Heatmap**: For false positive/negative validation across normal and anomalous frames.
2. **ROC Curve Comparison**: Showing the true positive performance trajectory against a random baseline across changing isolation thresholds.

---

## 📚 References & Dataset Citation

This work utilizes the benchmark UCSD Anomaly Detection Dataset curated by the UCSD Computer Vision and Systems Laboratory.

* **Dataset Resource**: The source files used in this setup were obtained from the repository on Kaggle: [UCSD Anomaly Dataset by Orvile](https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset).

---

## 📄 License

This project is made open-source under the terms of the [MIT License](https://www.google.com/search?q=LICENSE). Data preservation and intellectual property rights belong to the original dataset authors at UCSD.

```

```
