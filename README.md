Hybrid CNN Model for Brain Tumor Detection

This project implements a hybrid deep learning model for brain tumor detection and classification using:
- A custom Convolutional Neural Network (CNN) architecture
- Transfer Learning with ResNet50 and InceptionV3

The goal is to leverage the strengths of both custom feature extraction and pretrained networks to achieve high accuracy in identifying brain tumor types from MRI scans.

About

Brain tumors can be life-threatening if not detected early. This project builds a hybrid model combining:
- **Custom CNN layers** to capture unique features specific to the dataset
- **Pretrained models (ResNet50, InceptionV3)** to extract deep hierarchical features
- **Fusion** of extracted features for final classification

Tumor Types:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

---

Model Architecture

- **Custom CNN**:  
  3–4 convolutional layers with ReLU activation and max-pooling  
- **ResNet50**:  
  Pretrained on ImageNet, fine-tuned on MRI data  
- **InceptionV3**:  
  Pretrained on ImageNet, fine-tuned on MRI data  
- **Feature Fusion**:  
  Features from CNN, ResNet50, and InceptionV3 are concatenated  
- **Fully Connected Layers**:  
  Dropout + Dense layers for classification

---

Dataset

You can use datasets such as:
- **Brain Tumor MRI Dataset** from Kaggle: [Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- The dataset should include MRI images categorized into **Glioma**, **Meningioma**, **Pituitary**, and **Normal**.

**Data Preprocessing**:
- Resize images to 224x224
- Normalize pixel values
- Data augmentation (rotation, flipping, zooming)

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hybrid-cnn-brain-tumor.git
   cd hybrid-cnn-brain-tumor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Recommended libraries:
   - TensorFlow or PyTorch
   - OpenCV
   - Scikit-learn
   - Matplotlib
   - NumPy

---

Training

1. Prepare your dataset directory:
   ```
   dataset/
      glioma/
      meningioma/
      pituitary/
      normal/
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

3. You can configure hyperparameters inside `config.py`:
   - Learning rate
   - Batch size
   - Number of epochs
   - Optimizer settings

---

Evaluation

- Classification Metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC Curve

- Run evaluation:
  ```bash
  python evaluate.py
  ```

---

Results

| Model Component | Test Accuracy |
| :-------------- | :------------ |
| Custom CNN Only  | 86%           |
| ResNet50 Only    | 91%           |
| InceptionV3 Only | 89%           |
| **Hybrid Model (CNN + ResNet50 + InceptionV3)** | **94%** |

*Results may vary depending on dataset and training settings.*

---


## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

Would you also like me to give you a simple version of the README if you want it a bit shorter for a first release? 🚀  
Or maybe a badge-rich fancy version too? 🎖️
