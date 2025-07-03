import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pickle
import os

# Load vectorizer
with open("data_raw/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load models
model_paths = {
    "Logistic Regression": "models/Logistic_Regression_model.pkl",
    "KNN": "models/KNN_model.pkl",
    "Decision Tree": "models/Decision_Tree_model.pkl",
    "Naive Bayes": "models/Naive_Bayes_model.pkl"
}

models = {}
accuracies = {
    "Logistic Regression": 0.9121,
    "KNN": 0.8645,
    "Decision Tree": 0.8710,
    "Naive Bayes": 0.8992
}

for name, path in model_paths.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[name] = pickle.load(f)

# Predict function
def predict():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showerror("Error", "Please enter some text.")
        return

    selected = [m for m, v in selected_models.items() if v.get()]
    if not selected:
        messagebox.showwarning("Warning", "Please select at least one model.")
        return

    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    vec = tfidf.transform([text])
    for model_name in selected:
        model = models.get(model_name)
        if model:
            pred = model.predict(vec)[0]
            label = "AI-generated" if pred == 1 else "Human-written"
            tag = "red" if pred == 1 else "green"
            result_text.insert(tk.END, f"🔹 {model_name}: {label}\n", tag)
    result_text.config(state=tk.DISABLED)

# Accuracy popup
def show_accuracy():
    acc_info = "\n".join([f"{m}: {acc*100:.2f}%" for m, acc in accuracies.items()])
    messagebox.showinfo("Model Accuracies", acc_info)

# --- UI Setup ---
root = tk.Tk()
root.title("✨ AI vs Human Text Classifier")
root.geometry("820x740")
root.minsize(700, 640)
root.configure(bg="#f9fafb")

# Style config
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#f9fafb", font=("Segoe UI", 12))
style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)
style.configure("TCheckbutton", background="#f9fafb", font=("Segoe UI", 11))

# Header
header = tk.Label(root, text="🤖 AI vs Human Text Detector", bg="#1f2937", fg="#ffffff", font=("Segoe UI", 20, "bold"), pady=15)
header.pack(fill=tk.X)

# Input Frame
frame = tk.Frame(root, bg="#ffffff", bd=2, relief=tk.GROOVE)
frame.place(relx=0.5, rely=0.05, anchor="n", relwidth=0.9)

tk.Label(frame, text="✍️ Enter your text below:", font=("Segoe UI", 13, "bold"), bg="#ffffff").pack(anchor="w", padx=15, pady=10)
input_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=8, font=("Segoe UI", 11))
input_text.pack(padx=15, fill=tk.BOTH, expand=True)

# Model Selection Frame
model_frame = tk.LabelFrame(root, text="Select Models", font=("Segoe UI", 12, "bold"), bg="#f9fafb", padx=10, pady=5)
model_frame.place(relx=0.5, rely=0.45, anchor="n", relwidth=0.9)

selected_models = {}
for model in model_paths:
    var = tk.BooleanVar()
    chk = ttk.Checkbutton(model_frame, text=model, variable=var)
    chk.pack(anchor="w", padx=10)
    selected_models[model] = var

# Buttons
btn_frame = tk.Frame(root, bg="#f9fafb")
btn_frame.place(relx=0.5, rely=0.7, anchor="n")

predict_btn = ttk.Button(btn_frame, text="🔮 Predict", command=predict)
predict_btn.grid(row=0, column=0, padx=20, pady=10)

accuracy_btn = ttk.Button(btn_frame, text="📊 Show Accuracies", command=show_accuracy)
accuracy_btn.grid(row=0, column=1, padx=20, pady=10)

# Result Display
result_label = tk.Label(root, text="📢 Results:", font=("Segoe UI", 13, "bold"), bg="#f9fafb")
result_label.place(relx=0.07, rely=0.78)

result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=6, font=("Segoe UI", 11))
result_text.place(relx=0.5, rely=0.81, anchor="n", relwidth=0.9)
result_text.tag_config("red", foreground="red")
result_text.tag_config("green", foreground="green")
result_text.config(state=tk.DISABLED)

# Footer
tk.Label(root, text="© 2025 | NIT Patna | Machine Text Detector", bg="#f9fafb", font=("Segoe UI", 9)).pack(side=tk.BOTTOM, pady=8)

root.mainloop()
