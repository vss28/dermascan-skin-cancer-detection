# 🔬 DermaScan — Skin Cancer Detection using Machine Learning

## 📌 Overview

DermaScan is a machine learning-based web application that detects **7 types of skin diseases** from images using advanced feature engineering and ensemble learning techniques.

The system allows users to upload an image or capture one using a camera, and provides predictions along with confidence scores and risk levels.

---

## 🧠 Features

* ✅ 7-class skin disease classification
* ✅ Ensemble Learning (XGBoost + Random Forest + HistGradientBoosting)
* ✅ Soft Voting Classifier
* ✅ Feature Engineering (HOG, LBP, GLCM, Color Histograms, etc.)
* ✅ PCA-based feature analysis visualization
* ✅ Real-time prediction using Streamlit
* ✅ Image upload + camera input support

---

## 🧬 Diseases Detected

* Actinic Keratoses (akiec)
* Basal Cell Carcinoma (bcc)
* Benign Keratosis (bkl)
* Dermatofibroma (df)
* Melanoma (mel)
* Melanocytic Nevi (nv)
* Vascular Lesions (vasc)

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* XGBoost
* OpenCV
* Scikit-image
* Streamlit
* Pandas, NumPy

---

## 📊 Machine Learning Approach

* Extensive feature extraction (color, texture, shape-based features)
* Data augmentation for minority classes
* SMOTE for class balancing
* Training multiple models:

  * XGBoost
  * Random Forest
  * HistGradientBoosting
* Final prediction using **Soft Voting Ensemble**

---

## 📈 Model Performance

* Ensemble model achieves high accuracy on test data
* Balanced performance across all 7 classes

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/vss28/dermascan-skin-cancer-detection.git
cd dermascan-skin-cancer-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the model (IMPORTANT)

Due to GitHub file size limits, the trained model is not included.

👉 Download from:
https://drive.google.com/file/d/1HIQpHaO8BxxW_DUveAEA2ZpMIrl8-LvJ/view?usp=drive_link

After downloading, place it in:

```
models/skin_cancer_model.pkl
```

---

### 4. Run the application

```bash
streamlit run app.py
```

---

## 📸 Screenshots

<img width="1903" height="902" alt="image" src="https://github.com/user-attachments/assets/fc213910-ad6c-46e7-9b15-b42f7c9313c3" />

<img width="902" height="761" alt="image" src="https://github.com/user-attachments/assets/f4b94646-124e-4576-98e9-c5d9cc870fa7" />

<img width="900" height="601" alt="image" src="https://github.com/user-attachments/assets/9b48b890-8643-4a91-b2dd-6be723130f22" />




---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.
It is **not a substitute for professional medical diagnosis**. Always consult a qualified dermatologist.

---

## 👨‍💻 Author

Vedant Shelke

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
