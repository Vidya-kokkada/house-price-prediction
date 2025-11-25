import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
df = pd.read_csv('C:/Internship/Housing.csv')

# Encode categorical variables
le = LabelEncoder()
df['mainroad'] = le.fit_transform(df['mainroad'])
df['guestroom'] = le.fit_transform(df['guestroom'])
df['basement'] = le.fit_transform(df['basement'])
df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])
df['airconditioning'] = le.fit_transform(df['airconditioning'])
df['prefarea'] = le.fit_transform(df['prefarea'])
df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])

# Drop missing values
df = df.dropna()

# Features and target
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.set_page_config(page_title="House Price Prediction", layout="wide")

# Background styling
background_style = """
<style>
    .stApp {
        background: linear-gradient(to right, #e6e6fa, #d8bfd8, #dda0dd); /* Pastel lavender gradient */
        background-size: cover;
    }
    .stButton button {
        background-color: #9370db; /* Medium purple button */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #8a2be2; /* Blue violet on hover */
    }
</style>
"""
st.markdown(background_style, unsafe_allow_html=True)

# Title
st.title("🏠 House Price Prediction")

# Initialize session state for tab navigation and variables
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "tab1"

if "form_data" not in st.session_state:
    st.session_state.form_data = {}

# Tab Navigation Logic
if st.session_state.active_tab == "tab1":
    # Tab 1: Basic Info
    st.header("Basic Info")
    st.session_state.form_data["area"] = st.number_input("Area (sq ft)", step=100, key="area")
    st.session_state.form_data["bedrooms"] = st.number_input("Number of Bedrooms", step=1, key="bedrooms")
    st.session_state.form_data["bathrooms"] = st.number_input("Number of Bathrooms", step=1, key="bathrooms")
    st.session_state.form_data["stories"] = st.number_input("Number of Stories", step=1, key="stories")

    if st.button("Additional Features"):
        st.session_state.active_tab = "tab2"

elif st.session_state.active_tab == "tab2":
    # Tab 2: Additional Features
    st.header("Additional Features")
    mainroad = st.selectbox("Main Road Access", ['Select', 'Yes', 'No'], key="mainroad")
    guestroom = st.selectbox("Guest Room Available", ['Select', 'Yes', 'No'], key="guestroom")
    basement = st.selectbox("Basement Available", ['Select', 'Yes', 'No'], key="basement")
    hotwaterheating = st.selectbox("Hot Water Heating", ['Select', 'Yes', 'No'], key="hotwaterheating")
    airconditioning = st.selectbox("Air Conditioning", ['Select', 'Yes', 'No'], key="airconditioning")
    st.session_state.form_data["parking"] = st.number_input("Parking Spaces", value=1, step=1, key="parking")
    prefarea = st.selectbox("Preferred Area", ['Select', 'Yes', 'No'], key="prefarea")
    furnishingstatus = st.selectbox("Furnishing Status", ['Select', 'Furnished', 'Semi-Furnished', 'Unfurnished'], key="furnishingstatus")

    if 'Select' in [mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus]:
        st.warning("Please make sure to select an option for all fields in Additional Features.")
    else:
        # Encode inputs
        st.session_state.form_data["mainroad"] = 1 if mainroad == 'Yes' else 0
        st.session_state.form_data["guestroom"] = 1 if guestroom == 'Yes' else 0
        st.session_state.form_data["basement"] = 1 if basement == 'Yes' else 0
        st.session_state.form_data["hotwaterheating"] = 1 if hotwaterheating == 'Yes' else 0
        st.session_state.form_data["airconditioning"] = 1 if airconditioning == 'Yes' else 0
        st.session_state.form_data["prefarea"] = 1 if prefarea == 'Yes' else 0
        st.session_state.form_data["furnishingstatus"] = {'Furnished': 0, 'Semi-Furnished': 1, 'Unfurnished': 2}[furnishingstatus]

        # Prediction
        input_data = np.array([[st.session_state.form_data[key] for key in ["area", "bedrooms", "bathrooms", "stories", 
                                                                            "mainroad", "guestroom", "basement", 
                                                                            "hotwaterheating", "airconditioning", 
                                                                            "parking", "prefarea", "furnishingstatus"]]])
        prediction = model.predict(input_data)

        if st.button("Predict"):
            st.write(f"### Predicted House Price: ₹{prediction[0]:,.2f}")

