import streamlit as st
import pickle
import os
import PyPDF2
from nltk.tokenize import sent_tokenize
from streamlit_option_menu import option_menu

# --- Load TF-IDF vectorizer ---
with open("data_raw/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# --- Load models ---
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
f1_scores = {
    "Logistic Regression": 0.9132,
    "KNN": 0.8598,
    "Decision Tree": 0.8729,
    "Naive Bayes": 0.8947
}

for name, path in model_paths.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[name] = pickle.load(f)

# --- Streamlit Configuration ---
st.set_page_config(page_title="AI vs Human Detector", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    mark {
        background-color: #fff89a;
        padding: 0.2rem;
        border-radius: 4px;
    }
    .result-box {
        padding: 0.8rem;
        border-radius: 0.5rem;
        background-color: #f0f0f5;
        margin-top: 1rem;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🧠 AI vs Human Text Detector")
st.caption("Upload text or PDF and detect AI-generated content")

# --- Sidebar Navigation ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Text Input", "PDF Upload", "Model Info"],
        icons=["keyboard", "file-earmark-pdf", "graph-up"],
        menu_icon="cast",
        default_index=0
    )

# --- Highlighting Function ---
def highlight_ai_sentences(text, selected_models):
    sentences = sent_tokenize(text)
    vectors = tfidf.transform(sentences)
    results = {}

    for model_name in selected_models:
        model = models[model_name]
        preds = model.predict(vectors)
        highlighted_text = ""
        for sent, pred in zip(sentences, preds):
            if pred == 1:
                highlighted_text += f"<mark>{sent}</mark> "
            else:
                highlighted_text += f"{sent} "
        results[model_name] = highlighted_text.strip()

    return results

# --- Tab 1: Text Input ---
if selected == "Text Input":
    st.subheader("📄 Enter your text:")
    text = st.text_area("Paste your text below:", height=250)
    selected_models = st.multiselect("✅ Choose models:", list(models.keys()), default=list(models.keys())[:1])

    if st.button("🔍 Predict"):
        if not text.strip():
            st.warning("Enter some text first.")
        elif not selected_models:
            st.warning("Choose at least one model.")
        else:
            results = highlight_ai_sentences(text, selected_models)
            for model_name, html in results.items():
                st.markdown(f"### {model_name}")
                st.markdown(html, unsafe_allow_html=True)

# --- Tab 2: PDF Upload ---
elif selected == "PDF Upload":
    st.subheader("📄 Upload PDF file")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    selected_models = st.multiselect("✅ Choose models:", list(models.keys()), default=list(models.keys())[:1])

    if uploaded_file and selected_models:
        reader = PyPDF2.PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                full_text += content + "\n"

        if not full_text.strip():
            st.error("❌ Could not extract text from the PDF.")
        else:
            results = highlight_ai_sentences(full_text, selected_models)
            for model_name, html in results.items():
                st.markdown(f"### {model_name}")
                st.markdown(html, unsafe_allow_html=True)

# --- Tab 3: Model Info ---
elif selected == "Model Info":
    st.subheader("📊 Model Accuracy and F1 Scores")
    st.markdown("**ℹ️ F1 Score** is the harmonic mean of precision and recall. It's a balanced metric when classes are imbalanced.")
    st.table({
        "Model": list(accuracies.keys()),
        "Accuracy": [f"{v*100:.2f}%" for v in accuracies.values()],
        "F1 Score": [f"{f1_scores[k]:.4f}" for k in f1_scores]
    })
    st.success("✅ Choose the best-performing model or compare results using multiple models.")

# --- Footer ---
st.markdown("---")
st.markdown("<center><small>Made with ❤️ | NIT Patna</small></center>", unsafe_allow_html=True)
