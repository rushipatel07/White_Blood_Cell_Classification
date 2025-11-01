#  White Blood Cell Classification using CNN

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Framework-red.svg)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A deep learning project implementing a Convolutional Neural Network (CNN) for automated classification of white blood cells from microscopic images, achieving 95% accuracy across five cell types.

## Rushi Patel (12202120601038)
## Tanish Patel (12202120601048)
## Vivek chanchlani (12202120601060)


## 🛠️ Technologies Used

### Core Technologies
- **[Python](https://www.python.org/)** - Primary programming language
- **[TensorFlow](https://tensorflow.org/)** - Deep learning framework
- **[Keras](https://keras.io/)** - Neural network API
- **[Google Colab](https://colab.research.google.com/)** - Development environment

### Data Processing & Visualization
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation
- **[PIL (Pillow)](https://pillow.readthedocs.io/)** - Image processing
- **[Matplotlib](https://matplotlib.org/)** - Data visualization
- **[Seaborn](https://seaborn.pydata.org/)** - Statistical visualization
- **[Scikit-learn](https://scikit-learn.org/)** - Model evaluation
- **[KaggleHub](https://github.com/Kaggle/kagglehub)** - Dataset management

## 🚀 Features

- **Advanced Image Processing**
  - Automatic image resizing (128x128)
  - Pixel normalization
  - Data augmentation with rotations and flips
  - Balanced class distribution

- **Robust CNN Architecture**
  - 4-stage convolutional network
  - Batch normalization
  - Dropout regularization
  - MaxPooling for feature selection

- **Performance Optimization**
  - Early stopping
  - Learning rate scheduling
  - Model checkpointing
  - GPU acceleration

- **Comprehensive Evaluation**
  - Confusion matrix analysis
  - ROC curve generation
  - Per-class accuracy metrics
  - Real-time prediction visualization

## 🏗️ Model Architecture

```
Input (128x128x3)
│
├── Conv Block 1 (64 filters) → BatchNorm → MaxPool → Dropout(0.25)
│
├── Conv Block 2 (128 filters) → BatchNorm → MaxPool → Dropout(0.25)
│
├── Conv Block 3 (256 filters) → BatchNorm → MaxPool → Dropout(0.25)
│
├── Conv Block 4 (512 filters) → BatchNorm → MaxPool → Dropout(0.25)
│
├── Dense(1024) → Dropout(0.5)
│
├── Dense(512) → Dropout(0.5)
│
└── Output(5) → Softmax
```

## 📊 Dataset

- **Source**: [White Blood Cells Dataset](https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset)
- **Classes**: 5 WBC types
  - Neutrophils (6,231 training images)
  - Lymphocytes (2,427 training images)
  - Monocytes (561 training images)
  - Eosinophils (744 training images)
  - Basophils (212 training images)
- **Volume**: 14,514 images
  - Training: 10,175 total
    - Training split: 8,140 (80%)
    - Validation split: 2,035 (20%)
  - Test: 4,339 (Test-A)

## 📈 Performance

| Metric | Score |
|--------|--------|
| Training Accuracy | ~94% |
| Validation Accuracy | ~95% |
| Test Accuracy | ~95% |
| Inference Time | <100ms |
| Model Size | ~50MB |

## 🚀 Quick Start

### 1. Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/M-Husnain-Ali/White-Blood-Cell-Classification/blob/main/code.ipynb)

### 2. Install Dependencies
```python
!pip install kagglehub tensorflow numpy pandas matplotlib seaborn pillow scikit-learn tqdm
```

### 3. Download Dataset
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/white-blood-cells-dataset")
print("Path to dataset files:", path)
```

### 4. Run the Notebook
Simply execute all cells in `code.ipynb` sequentially. The notebook will:
- Load and preprocess the dataset
- Display data distribution and sample images
- Build and train the CNN model
- Evaluate performance with metrics and visualizations
- Save the trained model as `wbc_classification_model.keras`

## 📦 Requirements

```
tensorflow>=2.8.0
numpy>=1.19.5
pandas>=1.3.0
matplotlib>=3.4.3
seaborn>=0.11.0
pillow>=8.3.0
scikit-learn>=1.0.2
tqdm>=4.62.0
kagglehub>=0.1.0
```

## 💡 Usage Tips

### Training
- Use GPU runtime in Colab for faster training (Runtime → Change runtime type → GPU)
- The notebook includes data augmentation by default
- Training typically takes 10 epochs with early stopping
- Monitor validation metrics to avoid overfitting

### Making Predictions
```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('wbc_classification_model.keras')

# Load and preprocess image
img = Image.open('path/to/image.jpg').convert('RGB')
img = img.resize((128, 128))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
classes = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction)

print(f"Predicted: {predicted_class} with {confidence:.2%} confidence")
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request


## 🙏 Acknowledgments

- Kaggle for the comprehensive dataset
- TensorFlow team for the framework
- Google Colab for free GPU resources
- All contributors to this project

---

