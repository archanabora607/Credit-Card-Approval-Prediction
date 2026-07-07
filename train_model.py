# Import libraries
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("clean_dataset.csv")

# -----------------------------
# Encode categorical columns
# -----------------------------

from sklearn.preprocessing import LabelEncoder

industry_encoder = LabelEncoder()
ethnicity_encoder = LabelEncoder()
citizen_encoder = LabelEncoder()

data["Industry"] = industry_encoder.fit_transform(data["Industry"])
data["Ethnicity"] = ethnicity_encoder.fit_transform(data["Ethnicity"])
data["Citizen"] = citizen_encoder.fit_transform(data["Citizen"])
# -----------------------------
# Separate Features and Target
# -----------------------------
X = data.drop("Approved", axis=1)
y = data["Approved"]

# -----------------------------
# Scale the data
# -----------------------------
scaler = StandardScaler()

X = scaler.fit_transform(X)

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(industry_encoder, "industry_encoder.pkl")
joblib.dump(ethnicity_encoder, "ethnicity_encoder.pkl")
joblib.dump(citizen_encoder, "citizen_encoder.pkl")

print("\nModel saved successfully!")
print("Scaler saved successfully!")
print("\nIndustry Values:")
print(industry_encoder.classes_)

print("\nEthnicity Values:")
print(ethnicity_encoder.classes_)

print("\nCitizen Values:")
print(citizen_encoder.classes_)