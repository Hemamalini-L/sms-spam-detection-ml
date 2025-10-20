import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords
import string

st.title("📱 SMS Spam Detection Using Machine Learning")
st.write("This app classifies SMS messages as **Spam** or **Ham (Not Spam)** using a trained ML model.")

# ------------------------------------------------------
# 🧩 Function to load dataset (local or fallback)
# ------------------------------------------------------
@st.cache_data
def load_dataset():
    try:
        # First try local dataset
        if os.path.exists("spam.csv"):
            df = pd.read_csv("spam.csv", encoding='latin-1')
        else:
            # fallback to public dataset
            df = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/spam.csv", encoding='latin-1')
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        df = df[['label', 'message']]
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame(columns=['label', 'message'])

# ------------------------------------------------------
# 🧠 Preprocessing
# ------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# ------------------------------------------------------
# 💾 Train or load model
# ------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "spam_model.pkl"
    vectorizer_path = "vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    else:
        st.warning("⚙️ No pre-trained model found. Training a basic model now...")
        df = load_dataset()
        if df.empty:
            st.error("No dataset available for training.")
            return None, None

        df['clean_message'] = df['message'].apply(clean_text)
        vectorizer = TfidfVectorizer(max_features=3000)
        X = vectorizer.fit_transform(df['clean_message'])
        y = df['label'].map({'ham': 0, 'spam': 1})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

        # Save
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        return model, vectorizer

# ------------------------------------------------------
# 🚀 Prediction Function
# ------------------------------------------------------
def predict_message(message, model, vectorizer):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Spam" if prediction == 1 else "Ham (Not Spam)"

# ------------------------------------------------------
# 🧰 Main App Interface
# ------------------------------------------------------
model, vectorizer = load_model()
if model is not None:
    st.subheader("🔎 Enter an SMS message to classify:")
    user_input = st.text_area("Type your message here...")

    if st.button("Classify"):
        if user_input.strip():
            result = predict_message(user_input, model, vectorizer)
            if result == "Spam":
                st.error("🚨 This message is classified as **SPAM!**")
            else:
                st.success("✅ This message is **HAM (Not Spam)**.")
        else:
            st.warning("Please enter a message before classifying.")

st.markdown("---")
st.caption("Developed by Hemamalini L | Department of AI & DS | Vivekananda College of Technology for Women")
