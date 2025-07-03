# compare_models_table_plot.py

import pandas as pd
import matplotlib.pyplot as plt

# Manually enter from previous output or collect dynamically
data = {
    "Model": ["Logistic Regression", "KNN", "Decision Tree", "Naive Bayes"],
    "Accuracy": [0.9611, 0.5286, 0.8981, 0.8559],
    "F1 Score (Macro Avg)": [0.96, 0.40, 0.90, 0.86]
}

df = pd.DataFrame(data)
df = df.sort_values(by="Accuracy", ascending=False)

print("📋 Model Comparison Table:\n")
print(df)

# Plot bar chart
plt.figure(figsize=(10, 5))
plt.barh(df["Model"], df["Accuracy"], color="skyblue")
plt.xlabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xlim(0, 1)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()
