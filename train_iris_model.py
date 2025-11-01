# train_iris_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
data = pd.read_csv("Iris.csv")

# Drop unnecessary column if exists
if "Id" in data.columns:
    data = data.drop(columns=["Id"])

# Separate features and target
X = data.drop(columns=["Species"])
y = data["Species"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Accuracy check
accuracy = model.score(X_test_scaled, y_test)
print(f"✅ Model trained successfully with accuracy: {accuracy * 100:.2f}%")

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save model, scaler and encoder
joblib.dump(model, "models/iris_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(encoder, "models/label_encoder.joblib")

print("💾 Model, scaler and encoder saved in 'models' folder!")
