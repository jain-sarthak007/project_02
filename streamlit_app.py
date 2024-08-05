import os
import requests

def download_file(url, local_filename):
    """Download a file from a URL and save it locally."""
    response = requests.get(url)
    response.raise_for_status()  # Check for errors
    with open(local_filename, 'wb') as f:
        f.write(response.content)

def create_requirements_file(filename):
    """Create a requirements.txt file with necessary dependencies."""
    requirements = [
        "streamlit",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn"
    ]
    with open(filename, 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")

def main():
    # URLs of the files to download
    dataset_url = "https://example.com/path/to/your/medical_insurance.csv"
    dataset_filename = "medical_insurance.csv"
    requirements_filename = "requirements.txt"
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    
    # Download the dataset
    print(f"Downloading dataset from {dataset_url}...")
    download_file(dataset_url, os.path.join('data', dataset_filename))
    print(f"Dataset saved as {dataset_filename}")
    
    # Create requirements.txt
    print(f"Creating {requirements_filename}...")
    create_requirements_file(requirements_filename)
    print(f"{requirements_filename} created")

if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Title
st.title("Medical Insurance Cost Prediction")

# Load dataset
@st.cache
def load_data():
    return pd.read_csv('medical_insurance.csv')

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
