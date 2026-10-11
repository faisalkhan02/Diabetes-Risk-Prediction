
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("data/diabetes.csv")

# 2. Remove patient ID
df = df.drop(columns=["patient_id"])

# 3. Separate features and target
X = df.drop(columns=["diabetes_risk"])
y = df["diabetes_risk"]

# 4. Identify feature types
numerical_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["str", "object", "category"]
).columns.tolist()

# 5. Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 8. Fit and transform the features
X_processed = preprocessor.fit_transform(X)

print("Original feature shape:", X.shape)
print("Processed feature shape:", X_processed.shape)
print("Missing values after processing:", 
      pd.DataFrame(X_processed).isnull().sum().sum()) # type: ignore
print("Target categories:", sorted(y.unique().tolist()))
print("\nPreprocessing completed successfully!")
