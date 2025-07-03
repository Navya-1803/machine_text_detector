# view_models_info.py

import pickle
import os

model_dir = "models"
model_files = [
    "Logistic_Regression_model.pkl",
    "KNN_model.pkl",
    "Decision_Tree_model.pkl",
    "Naive_Bayes_model.pkl"
]

print(" Viewing Model Internals:\n")

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    model_name = model_file.replace("_model.pkl", "").replace("_", " ")

    print(f" {model_name}")
    print("-" * 50)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(" Parameters:")
    print(model.get_params())

    # Additional insights
    if hasattr(model, "coef_"):
        print("\n Coefficients (first 5):", model.coef_[0][:5])
    if hasattr(model, "feature_importances_"):
        print("\n Feature Importances (first 5):", model.feature_importances_[:5])
    if hasattr(model, "classes_"):
        print("\n Classes:", model.classes_)

    print("\n")
