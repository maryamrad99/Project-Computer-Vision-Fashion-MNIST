# Fashion-MNIST Image Classification using CNN and Random Forest

## Overview

This project compares a **Deep Learning (DL)** approach and a **Traditional Machine Learning (ML)** approach for automated fashion image classification using the **Fashion-MNIST** dataset.

The study evaluates:

* **Convolutional Neural Network (CNN)** implemented with TensorFlow/Keras
* **Random Forest (RF)** classifier implemented with Scikit-learn

The goal of the project is to analyze and compare both models in terms of:

* Classification accuracy
* Generalization performance
* Precision, Recall, and F1-Score
* Overfitting behavior
* Computational performance
* Category-level prediction reliability

---

## Dataset

The project uses the Fashion-MNIST dataset introduced by Han Xiao.

### Dataset Information

* **Total Images:** 70,000 grayscale images
* **Training Images:** 60,000
* **Testing Images:** 10,000
* **Image Size:** 28 × 28 pixels
* **Classes:** 10 fashion categories

### Categories

| Label | Category    |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

---

## Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* TensorFlow / Keras
* Scikit-learn
* NumPy
* Matplotlib
* Seaborn
* Pandas

---

## CNN Architecture

The CNN model was designed for spatial feature extraction using convolutional operations.

```python
cnn_model = keras.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

### CNN Features

* ReLU activation
* Max-pooling layer
* Softmax output layer
* Adam optimizer
* Sparse categorical crossentropy loss

---

## Random Forest Configuration

The Random Forest classifier was implemented as a traditional ML baseline.

### RF Parameters

* 100 Decision Trees
* Entropy splitting criterion
* Maximum depth = 100
* Multi-core parallel processing
* Fixed random state for reproducibility

---

## Data Preprocessing

### CNN Preprocessing

* Pixel normalization (`0–255 → 0–1`)
* Reshaped to `28×28×1`
* Spatial structure preserved

### RF Preprocessing

* Pixel normalization
* Flattened into `784-dimensional vectors`

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrices
* Training Time
* Inference Time

---

## Results Summary

| Metric            | CNN      | Random Forest |
| ----------------- | -------- | ------------- |
| Training Accuracy | 96.77%   | 100%          |
| Test Accuracy     | 91.31%   | 87.73%        |
| Training Time     | ~302 sec | ~23 sec       |
| Generalization    | Strong   | Moderate      |
| Overfitting       | Low      | High          |

---

## Key Findings

### CNN Strengths

* Better generalization performance
* Stronger classification of visually similar categories
* Preserved spatial relationships
* Learned hierarchical image features automatically

### RF Strengths

* Faster training time
* Lower computational cost
* Easier implementation

### Common Difficult Categories

Both models struggled with:

* Shirt
* Pullover
* Coat
* T-shirt/top

These categories share very similar grayscale patterns and shapes.

---

## Overfitting Analysis

The Random Forest model showed stronger overfitting behavior:

* RF Training Accuracy: **100%**
* RF Test Accuracy: **87.73%**

The CNN maintained a smaller training-testing gap, indicating better generalization.

---

## Conclusion

The CNN significantly outperformed the Random Forest classifier for Fashion-MNIST image classification.

Although the RF classifier was computationally faster, the CNN achieved:

* Higher classification accuracy
* Better generalization
* Improved handling of visually similar categories

The results demonstrate the importance of spatial feature extraction in computer vision tasks and highlight the effectiveness of convolutional architectures for image classification problems.

---

## Future Improvements

Possible future enhancements include:

* Deeper CNN architectures
* Transfer learning
* Data augmentation
* Hyperparameter optimization
* Real-world color fashion datasets
* GPU acceleration

---


## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/maryamrad99/Project-CV-Fashion-MNIST.git
cd Project-CV-Fashion-MNIST
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebook

Open:

```bash
CV_MNIST_FASHION.ipynb
```

using Jupyter Notebook or VS Code.

---
