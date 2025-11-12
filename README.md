
# 🎭 Multi-Page Sentiment Analysis Web App

<div align="center">

**A comprehensive, multi-page web application built with Streamlit for real-time sentiment analysis of YouTube comments and individual text inputs.**

</div>

## 📖 Overview

This project provides a powerful and user-friendly web interface for performing sentiment analysis. It features a multi-page design that allows users to either analyze the collective sentiment of comments from a YouTube video or get an instant classification for a single piece of text.

The backend is powered by a custom-trained **Support Vector Machine (SVM)** model, which has been trained on the extensive IMDB movie review dataset to classify text as either **Positive** or **Negative**. The application is designed for a wide audience, including data analysts, content creators, marketers, and anyone interested in applied Natural Language Processing (NLP).

## ✨ Features

* **Multi-Page Interface:** A clean, navigable sidebar separates the application into three distinct sections:
    * **YouTube Analysis:** Fetches and analyzes comments from any public YouTube video.
    * **Single Comment Analysis:** Provides instant sentiment classification for any user-entered text.
    * **About Page:** Describes the project and its developer.
* **YouTube Comment Aggregator:**
    * Accepts a standard YouTube video URL.
    * Uses the YouTube Data API v3 to fetch up to 100 of the latest English comments.
    * Calculates and displays an "Overall Positive Comment Ratio."
    * Provides a summary verdict (Positive, Negative, or Mixed/Neutral).
* **Real-Time Text Analysis:**
    * A dedicated page for analyzing individual sentences or paragraphs.
    * Delivers immediate Positive/Negative feedback.
* **Custom Machine Learning Model:**
    * Utilizes a LinearSVC (Support Vector Machine) model trained specifically for this task.
    * The model and the TF-IDF vectorizer are pre-trained and loaded for fast predictions.

## 🛠️ Technical Stack

* **Frontend / Web Application:**
    * **Streamlit:** For creating the interactive, multi-page web UI.
* **Backend / Machine Learning:**
    * **Python:** Core programming language.
    * **Scikit-learn:** For building and training the SVM model and using the TF-IDF vectorizer.
    * **Pandas:** For loading and manipulating the training data.
    * **NLTK:** For the core NLP preprocessing pipeline (tokenization, stop-word removal, lemmatization).
    * **Joblib:** For saving and loading the trained model and vectorizer.
* **API Integration:**
    * **Google API Python Client:** To interact with the YouTube Data API v3.
* **Dataset:**
    * **IMDB Movie Review Dataset:** A large, standard dataset used for training the sentiment model.

## 🚀 Setup and Installation

### 1. Prerequisites

* Python 3.8+
* A YouTube Data API Key. You can obtain one from the [Google Cloud Console](https://console.cloud.google.com/).

### 2. Clone the Repository

```bash
git clone [https://github.com/Vaibhav23rd/Sentiment_Analysis-main.git]
cd Sentiment_Analysis
```

### 3. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

Install all required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

The application uses Streamlit's secrets management for the API key.

1.  Create a directory named `.streamlit` in your project's root folder.
2.  Inside `.streamlit`, create a file named `secrets.toml`.
3.  Add your API key to this file in the following format:

    ```toml
    # .streamlit/secrets.toml

    [api_keys]
    youtube = "YOUR_API_KEY_HERE"
    ```

## How to Run

1.  Ensure your virtual environment is activated and you have completed the setup steps.
2.  Navigate to the project's root directory in your terminal.
3.  Run the main Streamlit application file:

    ```bash
    streamlit run YoutubeComments.py
    ```
4.  The application will open in your browser, with navigation to all pages available in the sidebar.

## 📁 Project Structure

```
Sentiment_Analysis/
├── .streamlit/
│   └── secrets.toml         # For storing API keys
├── pages/
│   ├── 01_Comments.py       # Page for single comment analysis
│   └── 02_About.py          # The "About" page
├── IMDB Dataset.csv         # Dataset for training
├── Train_model.py           # Script to train the SVM model
├── YoutubeComments.py       # Main application file (YouTube analysis page)
├── model_loader.py          # Helper script to load model and preprocess text
├── requirements.txt         # Project dependencies
├── sentiment_model.joblib   # (Generated) The trained SVM model
└── tfidf_vectorizer.joblib  # (Generated) The trained TF-IDF vectorizer
```

## 🧠 Model Training (Optional)

To retrain the model with new data or different parameters:

1.  Ensure the `IMDB Dataset.csv` file is in the root directory.
2.  Run the training script. This may take several minutes.
    ```bash
    python Train_model.py
    ```
3.  This will overwrite `sentiment_model.joblib` and `tfidf_vectorizer.joblib` with the newly trained versions.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">


</div>
