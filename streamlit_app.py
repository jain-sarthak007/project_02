import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

"""Data Collection & Analysis"""

# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('/content/medical_insurance.csv')

# first 5 rows of the dataframe
insurance_dataset.head()

# number of rows and columns
insurance_dataset.shape

# getting some informations about the dataset
insurance_dataset.info()

"""Categorical Features:
- Sex
- Smoker
- Region
"""

# checking for missing values
insurance_dataset.isnull().sum()

"""Data Analysis"""

# statistical Measures of the dataset
insurance_dataset.describe()

# distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()

insurance_dataset['sex'].value_counts()

# bmi distribution
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()

"""Normal BMI Range --> 18.5 to 24.9"""

# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()

insurance_dataset['children'].value_counts()

# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('smoker')
plt.show()

insurance_dataset['smoker'].value_counts()

# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('region')
plt.show()

insurance_dataset['region'].value_counts()

# distribution of charges value
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()

"""Data Pre-Processing

Encoding the categorical features
"""

# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

3 # encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

"""Splitting the Features and Target"""

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

print(X)

print(Y)

"""Splitting the data into Training data & Testing Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training

Linear Regression
"""

# loading the Linear Regression model
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

"""Model Evaluation"""

# prediction on training data
training_data_prediction =regressor.predict(X_train)

# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)

# prediction on test data
test_data_prediction =regressor.predict(X_test)

# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)

def get_input():
    try:
        x = int(input("Enter age: "))
        if x < 18:
            print("Age should be 18 or above")
            return False
        else:
            print("Age is valid")

        y = int(input("Enter gender (male-0, female-1): "))
        if y not in [0, 1]:
            print("Gender should be either 0 (male) or 1 (female)")
            return False

        z = float(input("Enter BMI: "))
        if z == 0:
            print("BMI cannot be zero")
            return False

        n = int(input("Enter the number of children: "))

        s = int(input("Enter smoker status (smoker-0, non-smoker-1): "))
        if s not in [0, 1]:
            print("Smoker status should be either 0 (smoker) or 1 (non-smoker)")
            return False

        r = int(input("Enter region (southeast:0, southwest:1, northeast:2, northwest:3): "))
        if r not in [0, 1, 2, 3]:
            print("Region should be either 0, 1, 2, or 3")
            return False

        return (x, y, z, n, s, r)

    except ValueError:
        print("Invalid input. Please enter the correct data type.")
        return False

input_data = None
while not input_data:
    input_data = get_input()
    if input_data:
        print("All inputs are valid.")
        print("Input data:", input_data)
    else:
        print("Invalid input. Please try again.")

"""Building a Predictive System"""

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is $', prediction[0])

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

# Display dataset
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
    sex = st.sidebar.selectbox("Gender", [0, 1])
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.0)
    children = st.sidebar.slider("Number of Children", 0, 10, 0)
    smoker = st.sidebar.selectbox("Smoker", [0, 1])
    region = st.sidebar.selectbox("Region", [0, 1, 2, 3])
    return np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)

input_data = user_input_features()
prediction = regressor.predict(input_data)

st.subheader("Insurance Cost Prediction")
st.write(f"The predicted insurance cost is: ${prediction[0]:.2f}")


