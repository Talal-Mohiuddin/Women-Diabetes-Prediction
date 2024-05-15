import pickle
import numpy as np
import pandas as pd
import streamlit as st

with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(data):
    data = np.asarray(data).reshape(1, -1)
    prediction = model.predict(data)
    return prediction

st.title("Women Diabetes Prediction")

st.header("Please provide the following details:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

if st.button("Predict"):
    prediction = predict(input_data)
    
    if prediction[0] == 1:
        st.write("The model predicts that the person **has diabetes**.")
    else:
        st.write("The model predicts that the person **does not have diabetes**.")
