# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("📊 Employee Salary Prediction")
st.write("Predict if an employee earns >50K based on various inputs.")

# Load model and preprocessing objects
preproc = joblib.load("preprocessor.pkl")
model_type = preproc['type']
scaler = preproc['scaler']
le_dict = preproc['encoder']

if model_type == 'dl':
    model = tf.keras.models.load_model("best_model.h5")
else:
    model = joblib.load("best_model.pkl")

# Input UI
st.subheader("Enter Employee Details")

def user_input():
    age = st.slider('Age', 18, 70, 30)
    fnlwgt = st.number_input('Final Weight', value=100000)
    edu_num = st.slider('Educational Number', 1, 16, 10)
    cap_gain = st.number_input('Capital Gain', value=0)
    cap_loss = st.number_input('Capital Loss', value=0)
    hours = st.slider('Hours Per Week', 1, 99, 40)

    workclass = st.selectbox('Workclass', le_dict['workclass'].classes_)
    education = st.selectbox('Education', le_dict['education'].classes_)
    marital = st.selectbox('Marital Status', le_dict['marital-status'].classes_)
    occupation = st.selectbox('Occupation', le_dict['occupation'].classes_)
    relationship = st.selectbox('Relationship', le_dict['relationship'].classes_)
    race = st.selectbox('Race', le_dict['race'].classes_)
    gender = st.selectbox('Gender', le_dict['gender'].classes_)
    country = st.selectbox('Native Country', le_dict['native-country'].classes_)

    data = pd.DataFrame([[age, fnlwgt, edu_num, cap_gain, cap_loss, hours,
                          workclass, education, marital, occupation, relationship,
                          race, gender, country]],
                        columns=['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week',
                                 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                 'race', 'gender', 'native-country'])

    for col in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        data[col] = le_dict[col].transform(data[col])

    data[['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']] = scaler.transform(
        data[['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']])

    return data

if st.button("Predict Salary"):
    inp = user_input()
    if model_type == 'dl':
        prediction = model.predict(inp)[0][0]
        pred_label = ">50K" if prediction > 0.5 else "<=50K"
    else:
        pred = model.predict(inp)[0]
        pred_label = ">50K" if pred else "<=50K"
    st.success(f"Prediction: {pred_label}")
