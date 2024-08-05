import subprocess

def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to ensure are installed
packages = [
    "streamlit",
    "numpy",
    "pandas",
    "scikit-learn"
]

# Install packages
for package in packages:
    try:
        install(package)
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")

# Try importing libraries and handle errors
try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import streamlit as st
except ImportError as e:
    # Display error in Streamlit and stop execution if import fails
    import streamlit as st
    st.error(f"Error importing libraries: {e}")
    st.stop()

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

def plot_distribution(column, title):
    """Helper function to plot distributions."""
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        if column in ['age', 'bmi', 'charges']:
            sns.histplot(insurance_dataset[column], kde=True, ax=ax)
        else:
            sns.countplot(x=column, data=insurance_dataset, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
    except NameError:
        st.warning(f"Seaborn is not available. {title} plot cannot be displayed.")

if st.checkbox('Show Age Distribution'):
    plot_distribution('age', 'Age Distribution')

if st.checkbox('Show Gender Distribution'):
    plot_distribution('sex', 'Gender Distribution')

if st.checkbox('Show BMI Distribution'):
    plot_distribution('bmi', 'BMI Distribution')

if st.checkbox('Show Children Distribution'):
    plot_distribution('children', 'Children Distribution')

if st.checkbox('Show Smoker Distribution'):
    plot_distribution('smoker', 'Smoker Distribution')

if st.checkbox('Show Region Distribution'):
    plot_distribution('region', 'Region Distribution')

if st.checkbox('Show Charges Distribution'):
    plot_distribution('charges', 'Charges Distribution')

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
