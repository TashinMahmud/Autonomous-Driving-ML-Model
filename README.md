# 🚗 Machine Learning Model for Autonomous Driving

[![Python](https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)](https://jupyter.org/)
[![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Model-F7931E?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Processing-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)

An autonomous driving object classification and annotation framework built using classical machine learning and computer vision techniques. The model processes real-world road camera frames, extracts **Histogram of Oriented Gradients (HOG)** features, applies **Principal Component Analysis (PCA)**, and benchmarks multiple classification algorithms to detect road obstacles and markers.

* **Target Classes**: Bikers, Cars, Pedestrians, Traffic Lights, and Trucks.
* **Classical ML Benchmark**: Evaluates and compares Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).

---

## 🏗️ Modeling & Feature Engineering Pipeline

The model utilizes structural feature descriptors and dimensionality reduction before running classification heads.

```
                      [ Image Dataset (512x512) ]
                                   │
                      [ Resized & Grayscale (64x64) ]
                                   │
                         [ HOG Feature Vector ]
                                   │
                       [ Dimensionality Reduction ]
                                 (PCA)
                                   │
                 [ Supervised Action Classifier ]
                    (Random Forest / SVM / KNN)
```

### Process Breakdown
1. **Pre-processing**: Normalizes image pixel matrices to `[0.0, 1.0]` range and downsamples them to `64x64` grayscale to minimize computation overhead.
2. **Feature Extraction**: Extracts HOG descriptors to capture edge directions and gradient structures.
3. **Dimensionality Reduction**: Fits PCA to compress sparse HOG vectors into the top 50 principal components.
4. **Classification**: Evaluates classifier models using confusion matrices, classification reports, and macro F1 scores.

---

## ⚡ Tech Stack & Core Libraries

* **Development Environment**: Google Colab / Jupyter Notebooks.
* **Feature Engineering**: [scikit-image](https://scikit-image.org/) — HOG descriptors.
* **Dimensionality Reduction & Modeling**: Scikit-Learn (PCA, KMeans, Random Forest, SVC, KNN).
* **Image Processing**: OpenCV (cv2) and PIL.
* **Visualizations**: Matplotlib, Seaborn.
* **Dataframes**: Pandas, NumPy.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have the required packages installed:
```bash
pip install numpy pandas scikit-learn scikit-image matplotlib seaborn opencv-python pillow
```

### 2. Execution
1. Open the Jupyter Notebook located at `Code/CSE_445_Project_(Complete).ipynb` inside Google Colab or your local Jupyter environment.
2. Configure the dataset paths:
   - By default, the notebook mounts Google Drive to retrieve the dataset zip.
   - For local runs, modify the `dataset_path` variable to point directly to `Dataset/Images` and load annotations from `Dataset/_annotations.csv`.
3. Execute the cells sequentially to visualize preprocessed frames, compute features, perform model training, and generate benchmark reports.

---

## 🧭 Project Directory Layout

```
Autonomous-Driving-ML-Model/
├── Code/                                      # Jupyter notebook project code
│   └── CSE_445_Project_(Complete).ipynb
├── Dataset/                                   # Model dataset
│   ├── Images/                                # Raw road camera frame images
│   └── _annotations.csv                       # Object class annotations and bounding boxes
├── Presentation/                              # Thesis and slide decks
│   └── A-Machine-Learning-Model-for-Autonomous-Driving.pdf
├── Project Report/                            # Final thesis documentation
│   └── FINAL REPORT.pdf
├── Project Progress Report/                   # Intermediate milestones and update notes
└── README.md                                  # Project documentation
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.
