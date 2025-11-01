# train_freshness_model.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

data_dir = "dataset"
categories = ["fresh", "not_fresh"]

X = []
y = []

# Load images
for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))  # resize all images
        img = img / 255.0  # normalize
        X.append(img.flatten())  # convert image to 1D array
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"✅ Freshness Model trained successfully with accuracy: {accuracy*100:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/freshness_model.joblib")
joblib.dump(scaler, "models/freshness_scaler.joblib")

print("💾 Freshness model saved successfully!")
