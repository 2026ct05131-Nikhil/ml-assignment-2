import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 1. Load Dataset (uses local file if present, otherwise downloads from UCI)
local_file = "garments_worker_productivity.csv"
uci_url = "https://archive.ics.uci.edu/static/public/597/data.csv"

if os.path.exists(local_file):
    df = pd.read_csv(local_file)
else:
    df = pd.read_csv(uci_url)
    df.to_csv(local_file, index=False)

print("=== 1. RAW DATASET OVERVIEW ===")
print(f"Total Instances: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")
print("\nMissing Values per Column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 2. Handle Duplicates & Clean Formatting
dup_count = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nDuplicate Rows Removed: {dup_count}")

# Clean trailing whitespace in 'department' ('finishing ' -> 'finishing')
df["department"] = df["department"].astype(str).str.strip()

# Extract month and day_of_month from 'date' to retain informative features
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df = df.drop(columns=["date"])

# 3. Target Variable Formulation & Class Imbalance Check
# Binary Target: 1 = Team met/exceeded targeted productivity, 0 = Missed target
df["target_met"] = (df["actual_productivity"] >= df["targeted_productivity"]).astype(int)
df = df.drop(columns=["actual_productivity"])

print("\n=== 2. TARGET CLASS DISTRIBUTION ===")
counts = df["target_met"].value_counts()
pcts = (df["target_met"].value_counts(normalize=True) * 100).round(2)
for cls in counts.index:
    label = "Met Target (1)" if cls == 1 else "Missed Target (0)"
    print(f"{label}: {counts[cls]} instances ({pcts[cls]}%)")

X = df.drop(columns=["target_met"])
y = df["target_met"]
print(f"\nFinal Input Feature Count: {X.shape[1]} (Minimum required: 12)")

# 4. Stratified 80:20 Train-Test Split (Before fitting any transformers)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
print(f"Training Set Size: {X_train.shape[0]} | Test Set Size: {X_test.shape[0]}")

# Export exact 20% test split required by repository checklist
test_df = X_test.copy()
test_df["target_met"] = y_test
test_df.to_csv("test_data.csv", index=False)
print("\nSaved test_data.csv to project root.")

# 5. Build & Fit Preprocessing Pipeline ONLY on Training Data
categorical_features = ["quarter", "department", "day", "team"]
numerical_features = [col for col in X.columns if col not in categorical_features]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

os.makedirs("model/artifacts", exist_ok=True)
joblib.dump(
    {
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    },
    "model/artifacts/preprocessed_split.pkl"
)
print(f"Transformed feature matrix shape: {X_train_processed.shape}")
print("Saved fitted preprocessor to model/artifacts/preprocessed_split.pkl")
