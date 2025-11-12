
# 1. Import Libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC  # Import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk

# 2. Load and Prepare the Data
print("Loading IMDb dataset...")
df = pd.read_csv('IMDB Dataset.csv')

# Map labels to numbers: 1 for 'positive', 0 for 'negative'
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
print("Dataset loaded and labels encoded.")

# 3. Define the NLP Preprocessing Pipeline
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Remove special characters
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(cleaned_tokens)

print("\nPreprocessing text data... (This can take a few minutes)")
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Preprocessing complete.")

# 4. Split Data and Vectorize
X = df['cleaned_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nVectorizing text using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("Vectorization complete.")

# 5. Train the SVM Model
print("\nTraining Support Vector Machine (SVM) model...")
# Using LinearSVC which is optimized for large datasets
model = LinearSVC(random_state=42, tol=1e-5)
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# 6. Evaluate the Model
print("\nEvaluating model performance...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# 7. Save the Model and Vectorizer
print("\nSaving model and vectorizer to disk...")
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
print("Files saved successfully: sentiment_model.joblib, tfidf_vectorizer.joblib")