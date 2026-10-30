import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("Student_Performance_Dataset.xlsx")

print("=" * 40)
print("STUDENT PERFORMANCE DATA MINING")
print("=" * 40)

# 1. Dataset information
print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Data Information:")
df.info()

# 2. Missing values
print("\n3. Missing Values:")
print(df.isnull().sum())

# 3. Duplicate records
print("\n4. Duplicate Rows:")
print(df.duplicated().sum())

# 4. Statistical summary
print("\n5. Statistical Summary:")
print(df.describe().round(2))

# 5. Data cleaning
df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical_cols:
    modes = df[col].mode()
    if not modes.empty:
        df[col] = df[col].fillna(modes.iloc[0])

# 6. Save cleaned dataset
df.to_csv("Student_Performance_Cleaned.csv", index=False)

print("\n6. Cleaned Dataset Shape:")
print(df.shape)

print("\n7. Remaining Missing Values:")
print(df.isnull().sum())

print("\nCleaned dataset saved successfully!")
# =====================================
# 8. MACHINE LEARNING - DECISION TREE
# =====================================

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n8. MACHINE LEARNING")

# Select input features
features = [
    "study_hours",
    "attendance",
    "previous_score",
    "sleep_hours"
]

X = df[features]
y = df["status"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create and train the model
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel: Decision Tree")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
import joblib

joblib.dump(model, "student_performance_model.pkl")

print("\nModel saved successfully!")