```python
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Bollworm Infestation Prediction",
    page_icon="🐛",
    layout="wide"
)

st.title("🐛 Bollworm Infestation Prediction Dashboard")
st.markdown("Predict weekly bollworm infestation probability using Logistic Regression with Platt Scaling.")

uploaded_file = st.file_uploader("bollworm.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.write("Columns Found:")
    st.write(list(df.columns))

    required_columns = [
        "temperature_cdd",
        "humidity",
        "crop_age",
        "bollworm"
    ]

    if all(col in df.columns for col in required_columns):

        X = df[["temperature_cdd", "humidity", "crop_age"]]
        y = df["bollworm"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42
        )

        model = LogisticRegression(
            penalty="l2",
            max_iter=1000
        )

        calibrated_model = CalibratedClassifierCV(
            model,
            method="sigmoid",
            cv=5
        )

        calibrated_model.fit(X_train, y_train)

        predictions = calibrated_model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.success(f"Model Accuracy: {accuracy:.2f}")

        st.subheader("Predict New Sample")

        temp = st.number_input(
            "Temperature CDD",
            min_value=0.0,
            value=100.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        crop_age = st.number_input(
            "Crop Age (Days)",
            min_value=0.0,
            value=45.0
        )

        if st.button("Predict Bollworm Probability"):

            sample = np.array([
                [temp, humidity, crop_age]
            ])

            sample_scaled = scaler.transform(sample)

            probability = calibrated_model.predict_proba(
                sample_scaled
            )[0][1]

            st.metric(
                "Infestation Probability",
                f"{probability*100:.2f}%"
            )

            if probability >= 0.5:
                st.error(
                    "High Bollworm Risk Detected"
                )
            else:
                st.success(
                    "Low Bollworm Risk"
                )

    else:

        st.error(
            "Dataset must contain columns:\n"
            "temperature_cdd, humidity, crop_age, bollworm"
        )

else:
    st.info("Upload a Bollworm CSV dataset to begin.")
```
