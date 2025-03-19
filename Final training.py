
# trainning random forest

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# File Paths
file_path_train = "/Users/alicezanni/Desktop/Dissertation /Data/Python/Dissertation/second_training.csv"
file_path_unseen = "/Users/alicezanni/Desktop/Dissertation /Data/Python/Dissertation/unseen_data_subset.csv"
output_file_path = "/Users/alicezanni/Desktop/Dissertation /Data/Python/Dissertation/predicted_values.csv"

# -------------------- Load and preprocess training data --------------------

# Load CSV file
df_train = pd.read_csv(file_path_train)

# Convert 'vector' column from string to list
df_train['vector'] = df_train['vector'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Function to pad vectors (keeping your version)
def pad_vectors(vectors, max_length=55):
    return np.array([vec[:max_length] + [0] * max(0, max_length - len(vec)) for vec in vectors], dtype=np.float32)

# Pad the vectors
X = pad_vectors(df_train['vector'].tolist(), max_length=55)
y = df_train['label']

# Normalise Features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Balance Classes with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -------------------- Train Random Forest Model --------------------

# Hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_random = RandomizedSearchCV(rf, param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1)
rf_random.fit(X_train, y_train)

# Train with best parameters from RandomizedSearchCV
best_rf = rf_random.best_estimator_
best_rf.fit(X_train, y_train)

# -------------------- Cross-Validation Scores --------------------
# Cross-validation to get the model performance scores
cross_val_scores = cross_val_score(best_rf, X_train, y_train, cv=5)

# Output cross-validation scores
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean Cross-validation score: {np.mean(cross_val_scores)}")
print(f"Standard Deviation of Cross-validation scores: {np.std(cross_val_scores)}")

# -------------------- Validation and test set --------------------

# Evaluate on validation set
y_val_pred = best_rf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Report:\n", classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = best_rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Report:\n", classification_report(y_test, y_test_pred))

# -------------------- Load and Preprocess Unseen Data --------------------

# Load JSON file
df_unseen = pd.read_csv(file_path_unseen)

# Convert vector column to list format
df_unseen['vector'] = df_unseen['vector'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)

# Pad unseen data to match training data dimensions
X_unseen = pad_vectors(df_unseen['vector'].tolist(), max_length=55)
X_unseen = scaler.transform(X_unseen)  # Normalize

# Predict labels for unseen data
df_unseen['predicted_label'] = best_rf.predict(X_unseen)

# Save predictions to CSV
df_unseen.to_csv(output_file_path, index=False)
print("Predictions saved to:", output_file_path)
