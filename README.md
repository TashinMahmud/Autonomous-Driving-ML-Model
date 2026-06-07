# 🚗 Autonomous Driving Object Classification via Classical ML & HOG

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TashinMahmud/Autonomous-Driving-ML-Model/blob/main/Code/CSE_445_Project_(Complete).ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)

---

An autonomous driving object recognition model trained using classical Computer Vision and Machine Learning algorithms. The system processes road-scene images, extracts structural edges using **Histogram of Oriented Gradients (HOG)**, applies **PCA** dimensionality reduction, and compares different classification heads for real-world driving environments.

</div>

---

## 🛠️ Image Pre-Processing & Feature Pipeline

The vision pipeline is structured to extract high-contrast gradients and compress them for classifiers:

```
+-----------------------------------------------------+
|                     INPUT IMAGE                     |
|  Loads RGB road frames from dataset (640x640px)     |
+--------------------------+--------------------------+
                           | (BGR Channels)
                           v
+-----------------------------------------------------+
|                  PRE-PROCESSING                     |
|  - Grayscale conversion (0.299R + 0.587G + 0.114B)  |
|  - Resizing to standard grid (128x128px)            |
+--------------------------+--------------------------+
                           | (Normalized Grays)
                           v
+-----------------------------------------------------+
|                 FEATURE EXTRACTION                  |
|  - Computes pixel gradient magnitude and direction  |
|  - Generates HOG feature descriptor vectors         |
+--------------------------+--------------------------+
                           | (HOG Vector)
                           v
+-----------------------------------------------------+
|             DIMENSIONALITY REDUCTION                |
|  - Principal Component Analysis (PCA) projection    |
|  - Retains 95% variance of features                 |
+--------------------------+--------------------------+
                           | (PCA Components)
                           v
+-----------------------------------------------------+
|                 CLASSIFIER HEADS                    |
|  - Support Vector Machine (SVC)                     |
|  - Random Forest Classifier                         |
|  - K-Nearest Neighbors (KNN)                        |
+-----------------------------------------------------+
```

---

## 🔬 Mathematical Formulas

### 1. Gradient Computations
For each pixel in the normalized image $I(x,y)$, we compute horizontal and vertical gradients $I_x$ and $I_y$:
$$I_x(x,y) = I(x+1, y) - I(x-1, y)$$
$$I_y(x,y) = I(x, y+1) - I(x, y-1)$$

### 2. Magnitude & Orientation
The gradient magnitude $G(x,y)$ and orientation angle $\theta(x,y)$ are computed as follows:
$$G(x,y) = \sqrt{I_x(x,y)^2 + I_y(x,y)^2}$$
$$\theta(x,y) = \arctan\left(\frac{I_y(x,y)}{I_x(x,y)}\right)$$
*These orientations are accumulated into 9-bin histograms over spatial cells.*

---

## 📊 Benchmark Results

| Model Classifier | Accuracy | F1-Score |
| :--- | :---: | :---: |
| **Support Vector Classifier (SVC)** | **88.2%** | **87.9%** |
| **Random Forest** | 84.5% | 83.8% |
| **K-Nearest Neighbors (KNN)** | 79.1% | 78.4% |

---

## 📂 Deliverables & Project Folders

*   **[`Code/CSE_445_Project_(Complete).ipynb`](Code/CSE_445_Project_(Complete).ipynb)**: The main Jupyter Notebook implementing the image pre-processing, HOG descriptors, PCA, model evaluations, and confusion matrices.
*   **[`Dataset/`](Dataset/)**: Annotations and sample image directories used for model validation.
*   **[`Project Report/FINAL REPORT.pdf`](Project%20Report/FINAL%20REPORT.pdf)**: Comprehensive academic report summarizing findings and validation graphs.
*   **[`Presentation/`](Presentation/)**: Project slide decks summarizing performance and future enhancements.

---

## 🚀 Local Run Guide

### 1. Requirements
Install the required vision and machine learning libraries:
```bash
pip install opencv-python scikit-learn pandas numpy matplotlib jupyter
```

### 2. Execute Training
Run the training notebook locally:
```bash
cd Code/
jupyter notebook CSE_445_Project_(Complete).ipynb
```
Evaluate performance metrics by executing the notebook cells sequentially.

---

## 👥 Authors

*   **Tashin Mahmud Khan**
*   **Md. Mynul Islam**
*   **Sheikh Shamiul Haque**
*   **S. M. Mahbub-Ul-Islam**

---

## 📜 License

Licensed under the MIT License.
