# app.py
import streamlit as st
import pandas as pd
import joblib
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ----------------------------
# 1. Page Configuration
# ----------------------------
st.set_page_config(
    page_title="SMS Spam Detection App",
    page_icon="📩",
    layout="centered"
)

st.title("📱 SMS Spam Detection Using Machine Learning")
st.write("This app classifies SMS messages as **Spam** or **Ham (Not Spam)** using a trained ML model.")

# ----------------------------
# 2. Load Data and Model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("spam_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    except:
        # If model not found, train a simple one
        st.warning("No pre-trained model found. Training a basic model now...")
        df = pd.read_csv("https://raw.githubusercontent.com/hemamalini/data/main/spam.csv", encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X = vectorizer.fit_transform(df['message'])
        y = df['label']
        model = MultinomialNB()
        model.fit(X, y)
        joblib.dump(model, "spam_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# 3. Text Preprocessing
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# 4. User Input Section
# ----------------------------
st.subheader("🔤 Enter an SMS Message")
user_sms = st.text_area("Type or paste the message here...", height=150)

if st.button("🔍 Classify Message"):
    if user_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        cleaned_sms = clean_text(user_sms)
        features = vectorizer.transform([cleaned_sms])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][prediction]
        
        if prediction == 1:
            st.error(f"🚨 Spam Message Detected! (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Ham Message (Not Spam). (Confidence: {proba:.2%})")

# ----------------------------
# 5. Example Messages
# ----------------------------
st.write("---")
st.subheader("📋 Example Messages")
example_spam = "Congratulations! You've won a $500 Amazon gift card. Click here to claim your prize."
example_ham = "Hey, are we still meeting at 6 PM today?"

col1, col2 = st.columns(2)
with col1:
    st.code(example_spam, language='text')
    if st.button("Predict Example Spam"):
        pred = model.predict(vectorizer.transform([clean_text(example_spam)]))[0]
        if pred == 1:
            st.error("Predicted: Spam")
        else:
            st.success("Predicted: Ham")

with col2:
    st.code(example_ham, language='text')
    if st.button("Predict Example Ham"):
        pred = model.predict(vectorizer.transform([clean_text(example_ham)]))[0]
        if pred == 1:
            st.error("Predicted: Spam")
        else:
            st.success("Predicted: Ham")

# ----------------------------
# 6. Footer
# ----------------------------
st.write("---")
st.markdown(
    """
    **Developed by:** Hemamalini L  
    Department of AI & DS, Vivekanandha College of Technology for Women  
    🧠 Model: Multinomial Naive Bayes | Feature: TF-IDF  
    """
)
