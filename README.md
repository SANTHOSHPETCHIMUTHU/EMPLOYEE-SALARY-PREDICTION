Employee Salary Prediction
Predict whether an individual's annual income exceeds $50K based on personal and job-related features using multiple machine learning and deep learning models.

Table of Contents
Project Overview

Features

Technologies Used

System Requirements

Installation & Setup

Project Workflow

Model Development & Evaluation

Deployment

Results

Future Scope

References

Project Overview
This project leverages data from the UCI Adult Income Dataset to build an accurate and interpretable income classification model. It includes extensive data cleaning, feature engineering, comparative model training (classical ML and deep learning), result visualization, and a Streamlit-based application for user interaction.

Features
Data preprocessing: outlier removal, missing value imputation, encoding, normalization.

Model training: Logistic Regression, Random Forest, KNN, SVM, and Deep Neural Network.

Performance comparison with visualizations.

Interactive web app for real-time predictions.

Model and data visualization dashboards.

Deployment-ready with Streamlit support.

Technologies Used
Library	Purpose
pandas	Data loading and manipulation
numpy	Numerical operations
matplotlib	Visualization
seaborn	Visualization
scikit-learn	Preprocessing, models, evaluation
tensorflow/keras	Deep learning implementation
joblib	Model serialization
streamlit	App deployment
System Requirements
Processor: Intel i5 or higher

RAM: 8GB minimum (16GB recommended)

Storage: 1GB+ free disk space

GPU: Optional (for DL model training acceleration)

Software: Python 3.8+, Chrome/Firefox for app

Installation & Setup
Clone the repository:

bash
git clone https://github.com/SANTHOSHPETCHIMUTHU/EMPLOYEE-SALARY-PREDICTION.git
cd EMPLOYEE-SALARY-PREDICTION
Install requirements:

bash
pip install -r requirements.txt
Run the Streamlit app:

bash
streamlit run app.py
Open the provided localhost link in your browser.

Project Workflow
1. Data Collection
Source: UCI Adult Income Dataset.

Post-cleaning size: 32,000+ rows, 15 columns.

2. Data Preprocessing
Replace missing values ('?') with NaN, drop incomplete records.

Remove outliers from numeric columns using the IQR method.

Encode categorical columns using LabelEncoder.

Normalize numerical features with StandardScaler.

Train-test split (80/20, stratified).

3. Model Training
Classical ML: Logistic Regression, Random Forest, KNN, SVM.

Deep Learning: Keras sequential model with two hidden (ReLU) and one output (sigmoid) layers.

4. Model Evaluation
Assess all models using accuracy score and classification reports.

Visualize comparison using horizontal bar charts.

5. Model Serialization
Save top-performing models: best_model.pkl, best_dl_model.h5.

6. App Deployment
Streamlit interface for user input, predictions, and visualizations.

Model Development & Evaluation
Best model achieved >90% accuracy after preprocessing and model optimization.

Visualizations include outlier detection (boxplots), feature distributions, and model score comparisons.

Notebook and scripts are provided for full reproducibility.

Deployment
Local: streamlit run app.py

Cloud-ready: compatible with Streamlit Cloud, Heroku, Render, Docker.

Results
Successfully built an accurate, interactive salary classification system.

Demonstrated the impact of preprocessing steps and compared traditional ML vs. deep learning approaches.

Visual interfaces allow end-users to experiment and predict in real time.

Future Scope
Hyperparameter tuning (GridSearchCV, RandomizedSearchCV).

Deploying advanced algorithms (XGBoost, LightGBM).

Implement feature selection (RFE, PCA).

Add explainability modules (SHAP, LIME).

Apply imbalanced class techniques (SMOTE, class weighting).

References
UCI Adult Income Dataset

Scikit-learn Documentation

Keras/TensorFlow Documentation

IQR Outlier Detection (Towards Data Science)

Streamlit Documentation
