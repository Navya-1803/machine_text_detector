# evaluate_saved_models.py

import pickle
import os
from sklearn.metrics import accuracy_score, classification_report

# Load test data
with open("data_raw/X_test_tfidf.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("data_raw/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# List of saved model files
model_dir = "models"
model_files = [
    "Logistic_Regression_model.pkl",
    "KNN_model.pkl",
    "Decision_Tree_model.pkl",
    "Naive_Bayes_model.pkl"
]

print("📊 Evaluating Saved Models:\n")

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n🔹 Evaluating {model_file.replace('_model.pkl', '').replace('_', ' ')}...")
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
