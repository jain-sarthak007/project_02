import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = [
    "requests",
    "streamlit",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn"
]

# Install packages
for package in packages:
    install(package)

import os
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Attempt to import matplotlib and handle import error
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    import streamlit as st
    st.warning("Matplotlib is not installed. Some visualizations may not be available.")

# Title
st.title("Medical Insurance Cost Prediction")

# Load dataset
@st.cache
def load_data():
    return pd.read_csv('data/medical_insurance.csv')

# Load and cache dataset
insurance_dataset = load_data()

# Display dataset information
if st.checkbox('Show Dataset Info'):
    st.write("First 5 rows of the dataset:")
    st.write(insurance_dataset.head())
    st.write("Number of rows and columns:", insurance_dataset.shape)
    st.write("Dataset info:")
    st.write(insurance_dataset.info())

# Plotting distribution
st.subheader("Data Analysis")

if matplotlib_available:
    if st.checkbox('Show Age Distribution'):
        st.write("Age Distribution:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.histplot(insurance_dataset['age'], kde=True, ax=ax)
        st.pyplot(fig)

    if st.checkbox('Show Gender Distribution'):
        st.write("Gender Distribution:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.countplot(x='sex', data=insurance_dataset, ax=ax)
        st.pyplot(fig)

    if st.checkbox('Show BMI Distribution'):
        st.write("BMI Distribution:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.histplot(insurance_dataset['bmi'], kde=True, ax=ax)
        st.pyplot(fig)

    if st.checkbox('Show Children Distribution'):
        st.write("Children Distribution:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.countplot(x='children', data=insurance_dataset, ax=ax)
        st.pyplot(fig)

    if st.checkbox('Show Smoker Distribution'):
        st.write("Smoker Distribution:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.countplot(x='smoker', data=insurance_dataset, ax=ax)
        st.pyplot(fig)

    if st.checkbox('Show Region Distribution'):
        st.write("Region Distribution:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.countplot(x='region', data=insurance_dataset, ax=ax)
        st.pyplot(fig)

    if st.checkbox('Show Charges Distribution'):
        st.write("Charges Distribution:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.histplot(insurance_dataset['charges'], kde=True, ax=ax)
        st.pyplot(fig)
else:
    st.info("Matplotlib is not available. Some charts may not be displayed.")

# Data Pre-processing
st.subheader("Predictive System")

# Encode categorical features
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Split data
X = insurance_dataset.drop(columns='charges')
Y = insurance_dataset['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# User input
st.sidebar.header("User Input")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 25)
    sex = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.0)
    children = st.sidebar.slider("Number of Children", 0, 10, 0)
    smoker = st.sidebar.selectbox("Smoker", [0, 1], format_func=lambda x: 'Smoker' if x == 0 else 'Non-Smoker')
    region = st.sidebar.selectbox("Region", [0, 1, 2, 3], format_func=lambda x: {0: 'Southeast', 1: 'Southwest', 2: 'Northeast', 3: 'Northwest'}[x])
    return np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)

input_data = user_input_features()
prediction = regressor.predict(input_data)

st.subheader("Insurance Cost Prediction")
st.write(f"The predicted insurance cost is: ${prediction[0]:.2f}")
