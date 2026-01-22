# ML2526 - Machine Learning Assignments

This repository contains a collection of projects and assignments completed for the **2025/26 Machine Learning course** at the Faculty of Mathematics, **Wrocław University of Science and Technology**.

The projects range from fundamental algorithms implemented from scratch in NumPy to advanced deep learning models using PyTorch.

## 🛠 Project Details

### 📈 Linear Regression Comparison
A foundational exercise comparing a manual implementation of Linear Regression using **NumPy**'s linear algebra capabilities against the standard `scikit-learn` implementation. This project serves as a verification of mathematical understanding versus optimized library performance.

### 🔢 MNIST DNN (from scratch)
Deep learning without high-level frameworks. This directory contains a **Dense Neural Network** (Multi-Layer Perceptron) implemented purely in NumPy, including the manual implementation of forward propagation, backpropagation, and weight updates.

### 🧠 Brain Tumor MRI Classification
This project focuses on medical image analysis. It utilizes **Convolutional Neural Networks (CNNs)** to classify MRI scans into specific tumor categories. I experimented with custom architectures and **Inception** modules to evaluate how different structural approaches affect feature extraction and accuracy.

### 🎮 Connect Four MCTS
An implementation of the **Monte Carlo Tree Search (MCTS)** algorithm designed for the game Connect Four. 
* **Evaluation:** The agent was tested against an Alpha-Beta Pruning baseline.
* **Results:** After extensive hyperparameter tuning, the MCTS agent achieved an **87% win ratio**.

### ⚡ Epilepsy Seizure Detection
A time-series classification task using publicly available data from the **Epilepsy Research Center at Bonn University**. 
* **Data:** Single-electrode EEG signals segmented into 1-second windows.
* **Model:** Utilizes **Recurrent Neural Networks (RNNs)** to capture the temporal dependencies in the signal to determine the presence of a seizure (binary classification).
