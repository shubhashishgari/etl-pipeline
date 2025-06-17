import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib

# === USER INPUT ===
input_path = input("Enter the full path to your CSV dataset: ").strip().strip('"')
if not os.path.exists(input_path):
    raise FileNotFoundError(f"The file was not found at the path: {input_path}")

df = pd.read_csv(input_path)
print(f"Loaded data from {input_path}")
print("Available columns:", df.columns.tolist())

# Ask user to enter the target column
TARGET_COLUMN = input("Enter the name of the target column: ").strip().strip('"')
while 'id' in TARGET_COLUMN.lower():
    print(f" The column '{TARGET_COLUMN}' seems to be an ID and is not suitable as a target.")
    print("Please choose another column.")
    print("Available columns:", [col for col in df.columns if 'id' not in col.lower()])
    TARGET_COLUMN = input("Enter the name of the target column: ").strip().strip('"')

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")

OUTPUT_DIR = "processed_data"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# === CLEAN DATA ===
df.dropna(subset=[TARGET_COLUMN], inplace=True)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Optionally drop common ID columns
id_cols = [col for col in X.columns if 'id' in col.lower()]
X.drop(columns=id_cols, inplace=True, errors='ignore')

# === STRATEGY SELECTION ===
is_classification = (y.dtype == 'object') or (y.nunique() <= 20 and str(y.dtype).startswith('int'))

# Warn for high cardinality if classification
if is_classification and y.nunique() > 10:
    print(f" High number of target classes: {y.nunique()}. Consider grouping rare classes.")
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 0.01 * len(y)].index
    y = y.apply(lambda val: 'Other' if val in rare_classes else val)

# === SPLIT ===
stratify_param = y if is_classification else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_param)

# === FEATURE TYPES ===
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# === PIPELINES ===
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# === TRANSFORM ===
print("Preprocessing training data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# === FEATURE NAMES ===
try:
    cat_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features)
    processed_columns = numerical_features + cat_names.tolist()
except:
    processed_columns = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

# === SAVE PROCESSED DATA ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_df = pd.DataFrame(X_train_processed, columns=processed_columns)
train_df[TARGET_COLUMN] = y_train.reset_index(drop=True)
test_df = pd.DataFrame(X_test_processed, columns=processed_columns)
test_df[TARGET_COLUMN] = y_test.reset_index(drop=True)
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_processed.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_processed.csv"), index=False)
print("Processed data saved in 'processed_data/'")

# === TRAIN MODEL ===
if is_classification:
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
else:
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    print("\nRegression Report:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")

# === FEATURE IMPORTANCE ===
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh([processed_columns[i] for i in sorted_idx], importances[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Important Features")
    plt.tight_layout()
    plt.show()

# === SAVE MODEL ===
joblib.dump(model, os.path.join(OUTPUT_DIR, "trained_model.pkl"))
print(f"Model saved to {os.path.join(OUTPUT_DIR, 'trained_model.pkl')}")
