# PCB Defect Detection and Classification System

## 📌 Overview

This project focuses on detecting and classifying defects in Printed Circuit Boards (PCB) using image processing and deep learning techniques. The system processes PCB images, identifies defects, and provides a user-friendly interface for visualization.

---

## 🚀 Features

* Image preprocessing and defect highlighting
* Contour detection and ROI extraction
* Deep learning-based defect classification (EfficientNet)
* Web interface for uploading and analyzing images
* Annotated output with defect labels
* Downloadable results and logs

---

## 🗂️ Project Structure

```
PCB_PROJECT/
│
├── infosys_milestone1/
├── infosys_milestone2/
├── infosys_milestone3/
├── infosys_milestone4/
├── README.md
└── .gitignore
```

---

## 🧩 Milestones

### 🔹 Milestone 1: Dataset Preparation and Image Processing

* Dataset setup (DeepPCB)
* Image alignment and preprocessing
* Image subtraction and thresholding (Otsu)
* Contour detection and ROI extraction

**Outputs:**

* Defect-highlighted images
* ROI datasets
* Contour visualizations

---

### 🔹 Milestone 2: Model Training and Evaluation

* Model: EfficientNet (PyTorch/TensorFlow)
* Image resizing (128x128)
* Data augmentation
* Training with Adam optimizer

**Outputs:**

* Trained model (.pth)
* Accuracy and loss graphs
* Confusion matrix

---

### 🔹 Milestone 3: Frontend & Backend Integration

* Web UI using Streamlit
* Image upload (template & test)
* Backend inference pipeline
* Display predictions with bounding boxes

**Outputs:**

* Functional web app
* Real-time predictions

---

### 🔹 Milestone 4: Finalization and Delivery

* Export results (images + CSV logs)
* Performance optimization
* Documentation and demo

**Outputs:**

* Final web app
* Documentation
* Presentation/demo

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/PCB-Defect-Detection-and-Classification-System.git
cd PCB-Defect-Detection-and-Classification-System
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Training:

```bash
python train.py
```

### Run Inference:

```bash
python inference.py
```

### Run Web App:

```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics

* Accuracy ≥ 95% (target)
* Confusion Matrix
* Precision & Recall
* Low false positives/negatives

---

## 🛠️ Technologies Used

* Python
* OpenCV
* PyTorch / TensorFlow
* Streamlit
* NumPy, Matplotlib

---

## 📸 Sample Output

* Defect highlighted images
* Bounding boxes around defects
* Classification labels

---

## 📄 Documentation

* Technical documentation included
* User guide for running the system
* Demo video / presentation (if available)

---

## 👩‍💻 Author

**Sanjana Penneru**

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgment

Dataset: DeepPCB
Guidance: Infosys Springboard Mentor
