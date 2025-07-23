# 💼 Employee Salary Prediction using Machine Learning and Deep Learning

This project is an interactive Streamlit web application that predicts whether an individual's annual income exceeds $50K based on demographic and employment features from the **UCI Adult Income Dataset**. It combines machine learning and deep learning models for comparison, emphasizing clean preprocessing, outlier handling, and model accuracy.

---

## 📁 Dataset

- **Source**: [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **File**: `adult.csv`
- **Description**: Contains 14 attributes including age, education, occupation, and hours-per-week. Target variable is `<=50K` or `>50K`.

---

## 🧠 Features

- Data cleaning and preprocessing
- Outlier detection and removal with boxplot visualization
- Feature encoding and scaling
- Model training with multiple ML algorithms and a DL model
- Model evaluation and accuracy comparison
- Streamlit UI for real-time prediction and comparison

---

---

## ✅ System Requirements

**Python Libraries Used**:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `tensorflow`, `keras`
- `joblib`
- `streamlit`

---

## 🚀 How to Run the App Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SANTHOSHPETCHIMUTHU/EMPLOYEE-SALARY-PREDICTION.git
   cd EMPLOYEE-SALARY-PREDICTION

2. Install Requirements
    pip install -r requirements.txt
    Run Streamlit App
    streamlit run app.py
   
📊 Model Performance (Accuracy)
Model	Accuracy
Logistic Regression	~0.85
Random Forest	~0.88
K-Nearest Neighbors	~0.84
SVM	~0.86
Deep Learning	>0.91 ✅
Best model used for final prediction is the deep learning model.


📌 Features in Streamlit App
📂 View Dataset and Preprocessing Summary

📉 Visualize Outliers (Before & After)

🤖 Predict income using input form

🔀 Compare predictions across multiple models

📈 Bar chart of model comparison

⚡ Fast loading with caching via @st.cache_data

✨ Final Outcome
A clean, responsive, and accurate salary prediction web app with:

✅ >90% accuracy

✅ Outlier-robust ML pipeline

✅ Streamlit frontend for real-time user interaction


