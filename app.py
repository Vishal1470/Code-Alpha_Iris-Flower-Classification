import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image

st.set_page_config(page_title="🌸 Iris Flower & Freshness Detector", page_icon="🌼")

st.title("🌼 Iris Flower Classification & Freshness Prediction")
st.write("This app predicts Iris flower species and freshness status from image or measurements.")

# Load models
iris_model = joblib.load("models/iris_model.joblib")
iris_scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

fresh_model = joblib.load("models/freshness_model.joblib")
fresh_scaler = joblib.load("models/freshness_scaler.joblib")

# Sidebar input
st.sidebar.header("🌸 Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# --- Species Prediction ---
if st.button("🔍 Predict Species"):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_scaled = iris_scaler.transform(X)
    y_pred = iris_model.predict(X_scaled)
    predicted_class = label_encoder.inverse_transform(y_pred)[0]
    st.success(f"🌸 Predicted Species: **{predicted_class}**")

# --- Freshness Prediction ---
st.subheader("🌼 Upload a Flower Image to Check Freshness")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Flower", use_container_width=True)

    # Convert to OpenCV format
    img_cv = np.array(img)
    img_cv = cv2.resize(img_cv, (64, 64))
    img_cv = img_cv / 255.0
    features = img_cv.flatten().reshape(1, -1)
    features_scaled = fresh_scaler.transform(features)

    pred = fresh_model.predict(features_scaled)[0]
    freshness_label = "Fresh 🌼" if pred == 0 else "Not Fresh 🥀"

    st.info(f"Freshness Prediction: **{freshness_label}**")

    # Purchase suggestion
    if freshness_label == "Fresh 🌼":
        st.success("✅ Suitable for purchase!")
    else:
        st.error("❌ Not recommended for purchase.")
