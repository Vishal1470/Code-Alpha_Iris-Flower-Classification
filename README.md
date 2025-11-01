🌸 Iris Flower Classification & Freshness Detection System
📘 Overview

This project is a Machine Learning + Computer Vision based application that classifies Iris flower species from numeric features and determines whether a flower is Fresh 🌼 or Not Fresh 🥀 using image processing.
It also provides a purchase recommendation based on freshness.

The app is built with Python, Scikit-learn, OpenCV, and Streamlit.

🧩 Features

✅ Iris Species Classification
Predicts the species (Setosa, Versicolor, Virginica) using Sepal and Petal measurements.

✅ Freshness Detection
Classifies flower images as Fresh or Not Fresh using a Random Forest model trained on image data.

✅ Purchase Recommendation
Gives an automatic suggestion — “Suitable for purchase” or “Not recommended”.

✅ Interactive Web App
User-friendly interface made with Streamlit sliders and image upload functionality.

🏗️ Project Structure
Iris Flower Classification/
│

├── Iris.csv                     # Dataset for Iris classification

├── train_iris_model.py          # Train model for Iris species

├── train_freshness_model.py     # Train model for image-based freshness

├── app.py                       # Streamlit app combining both models

│

├── models/

│   ├── iris_model.joblib

│   ├── scaler.joblib

│   ├── label_encoder.joblib

│   ├── freshness_model.joblib

│   └── freshness_scaler.joblib

│

└── dataset/
    ├── fresh/                   # Images of fresh flowers
    
    └── not_fresh/               # Images of wilted flowers

⚙️ Installation and Setup
1️⃣ Clone or Download
git clone https://github.com/Vishal1470/Iris-Flower-Classification.git
cd "Iris Flower Classification"

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Train the Models
(a) Train Iris Model
python train_iris_model.py

(b) Train Freshness Model

Make sure dataset/fresh and dataset/not_fresh contain flower images.

python train_freshness_model.py

4️⃣ Run the Streamlit App
streamlit run app.py

🧠 Technologies Used
Component	Technology
Programming Language	Python 3.10+
Libraries	pandas, numpy, scikit-learn, joblib, OpenCV, matplotlib, streamlit
ML Models	Random Forest Classifier
Interface	Streamlit
Image Input	cv2 + PIL
🌼 Output Examples

Prediction Result:

🌸 Predicted Species: Iris-versicolor  
🌼 Freshness: Fresh  
✅ Suitable for purchase

🚀 Future Scope

Upgrade image model using CNN (Convolutional Neural Network) for higher accuracy.

Add dataset upload & retraining via UI.

Deploy the app on Streamlit Cloud / Hugging Face Spaces for public access.

Integrate a dashboard for visual analytics and flower statistics.

📜 License

This project is open-source and available under the MIT License.
