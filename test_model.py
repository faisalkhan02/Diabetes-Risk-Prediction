
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("model/diabetes_risk_model.pkl")

# Load test data
df = pd.read_csv("data/diabetes.csv")
df = df.drop(columns=["patient_id"])

X = df.drop(columns=["diabetes_risk"])
y = df["diabetes_risk"]

# Use the same test split as training
from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Predict using the saved model
predictions = model.predict(X_test)

# Compare actual and predicted labels
results = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": predictions
})

print("First 10 predictions:")
print(results.head(10))

print("\nTest accuracy:", round(
    (results["Actual"] == results["Predicted"]).mean(), 4
))

print("\nPrediction counts:")
print(results["Predicted"].value_counts())
