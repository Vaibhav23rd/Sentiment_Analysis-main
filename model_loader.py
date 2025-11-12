# model_loader.py

import nltk
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# --- Load model and vectorizer ---
try:
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.error("Model or vectorizer not found. Please run `train_model.py` first.")
    st.stop()

# --- Setup NLTK tools ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Text preprocessing ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(cleaned_tokens)

# --- Predict list of comments ---
def analyze_sentiment_svm(comments_list):
    processed = [preprocess_text(c) for c in comments_list]
    vect = vectorizer.transform(processed)
    preds = model.predict(vect)
    avg = sum(preds) / len(preds) if preds.size > 0 else 0
    return preds, avg

# --- Predict single comment ---
def predict_single_comment_svm(comment):
    processed = preprocess_text(comment)
    vect = vectorizer.transform([processed])
    return model.predict(vect)[0]
