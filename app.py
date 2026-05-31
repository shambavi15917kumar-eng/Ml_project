import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Bollworm Infestation Prediction",
    page_icon="🐛",
    layout="wide",
)

# --------- UI ---------
st.title("🐛 Bollworm Infestation Prediction Dashboard")
st.markdown(
    "Predict weekly bollworm infestation probability using **Logistic Regression** + **Platt Scaling**."
)

with st.sidebar:
    st.header("Inputs")
    temp = st.number_input("Temperature (CDD / DegreeDays proxy)", value=120.0)
    humidity = st.number_input("Humidity (%)", value=65.0)
    crop_age = st.number_input("Crop Age (days)", value=20.0)

    st.divider()
    st.caption("Upload dataset to train & visualize.")
    uploaded_file = st.file_uploader("bollworm.csv", type=["csv"])

# --------- Helpers ---------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Support both naming styles found in this project
    # Expected dataset (from local bollworm.csv):
    # Temperature, DegreeDays, Humidity, CropAge, Infestation
    colmap = {
        "temperature_cdd": "Temperature",
        "degree_days": "DegreeDays",
        "humidity": "Humidity",
        "crop_age": "CropAge",
        "bollworm": "Infestation",
    }
    # If already in normalized style, keep.
    if all(c in df.columns for c in ["temperature_cdd", "humidity", "crop_age", "bollworm"]):
        return df

    # Otherwise, try to map from dataset style to normalized style.
    if all(c in df.columns for c in ["Temperature", "Humidity", "CropAge", "Infestation"]):
        df = df.copy()
        df["temperature_cdd"] = df["DegreeDays"] if "DegreeDays" in df.columns else df["Temperature"]
        df["humidity"] = df["Humidity"]
        df["crop_age"] = df["CropAge"]
        df["bollworm"] = df["Infestation"]
        return df

    return df

# --------- Main content ---------

if uploaded_file is None:
    st.info("Upload a CSV (e.g., **bollworm.csv**) to view graphs and train the model.")
    st.stop()

# Load + normalize
raw_df = pd.read_csv(uploaded_file)
df = normalize_columns(raw_df)

required_columns = ["temperature_cdd", "humidity", "crop_age", "bollworm"]

if not all(c in df.columns for c in required_columns):
    st.error(
        "Dataset must contain either:\n"
        "Option A: temperature_cdd, humidity, crop_age, bollworm\n"
        "Option B: Temperature, DegreeDays, Humidity, CropAge, Infestation"
    )
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])

# --------- Charts ---------

st.subheader("Exploratory Graphs")

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="bollworm", data=df, ax=ax)
    ax.set_title("Class Distribution (bollworm)")
    ax.set_xlabel("bollworm (0/1)")
    st.pyplot(fig, clear_figure=True)

with c2:
    # Pie chart of class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts = df["bollworm"].value_counts().sort_index()
    labels = [f"0 (Low)" if i == 0 else f"1 (High)" for i in class_counts.index]
    ax.pie(
        class_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Pie Chart: Bollworm vs Non-bollworm")
    st.pyplot(fig, clear_figure=True)

c3, c4 = st.columns(2)
with c3:
    # Histogram: humidity
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["humidity"], bins=15, kde=True, ax=ax)
    ax.set_title("Humidity Distribution")
    st.pyplot(fig, clear_figure=True)

with c4:
    # Boxplot: crop age by class
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="bollworm", y="crop_age", data=df, ax=ax)
    ax.set_title("Crop Age by Bollworm Class")
    ax.set_xlabel("bollworm (0/1)")
    st.pyplot(fig, clear_figure=True)

st.divider()

# --------- Model training ---------

X = df[["temperature_cdd", "humidity", "crop_age"]]
y = df["bollworm"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

base_model = LogisticRegression(penalty="l2", max_iter=1000)
calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)

calibrated_model.fit(X_train, y_train)

pred = calibrated_model.predict(X_test)
acc = accuracy_score(y_test, pred)

st.subheader("Model Performance")
st.success(f"Model Accuracy: {acc:.2f}")

# --------- Prediction ---------

st.subheader("Predict New Sample")

colA, colB, colC = st.columns(3)
with colA:
    temp_in = st.number_input("Temperature CDD", value=float(temp))
with colB:
    humidity_in = st.number_input("Humidity (%)", value=float(humidity))
with colC:
    crop_age_in = st.number_input("Crop Age (days)", value=float(crop_age))

if st.button("Predict Bollworm Probability"):
    sample = np.array([[temp_in, humidity_in, crop_age_in]], dtype=float)
    sample_scaled = scaler.transform(sample)

    probability = calibrated_model.predict_proba(sample_scaled)[0][1]

    st.metric("Infestation Probability", f"{probability*100:.2f}%")

    if probability >= 0.5:
        st.error("High Bollworm Risk Detected")
    else:
        st.success("Low Bollworm Risk")

