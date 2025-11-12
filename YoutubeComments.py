# app.py
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect
import re
# Import our new SVM model functions
from model_loader import analyze_sentiment_svm

# --- WARNING: Hardcoding API keys is a security risk! ---
# Replace 'YOUR_API_KEY_HERE' with your actual YouTube Data API key.
# For deployed apps, use st.secrets["api_key"] instead.

api_key = st.secrets["api_keys"]["youtube"]

# ---------------------------------------------------------


def extract_youtube_video_comments(video_id):
    """Fetches up to 100 English comments from a YouTube video."""
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        st.error("API Key not configured. Please add your key at the top of the app.py file.")
        return None

    try:
        # Build the YouTube API service object
        youtube = build('youtube', 'v3', developerKey=api_key)

        # Initial request to fetch comment threads
        video_response = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id,
            maxResults=100, # Request up to 100 items per page
            textFormat='plainText' # Get plain text to avoid HTML tags
        ).execute()

    except HttpError as e:
        st.error(f"An API error occurred: {e}")
        st.info("This could be due to an invalid API key, disabled comments on the video, or exceeding API quota.")
        return None

    comments = []
    # Loop until we have 100 comments or there are no more pages
    while len(comments) < 100:
        for item in video_response['items']:
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            try:
                # Add only English comments to the list
                if detect(comment_text) == 'en':
                    comments.append(comment_text)
            except Exception:
                # Ignore comments where language detection fails
                pass
        
        # Stop if we've reached our target of 100 comments
        if len(comments) >= 100:
            break

        # Check for more pages of comments
        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=video_response['nextPageToken'],
                maxResults=100,
                textFormat='plainText'
            ).execute()
        else:
            # No more pages, exit the loop
            break
            
    return comments[:100] # Ensure we return a maximum of 100 comments


def extract_video_id(youtube_link):
    """Extracts the YouTube video ID from a URL using regular expressions."""
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, youtube_link)
    return match.group(1) if match else None


# --- Streamlit UI (Updated for SVM Model) ---
st.set_page_config(page_title="YouTube Sentiment Analysis", layout="wide")
st.title("YouTube Video Sentiment Analyzer (SVM Model)")
st.markdown("Analyzes comment sentiment (Positive/Negative) using a custom-trained Support Vector Machine.")

# Since the API key is hardcoded, we only need one input field.
youtube_url = st.text_input(
    "Enter YouTube Video URL:", 
    placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ"
)

if st.button("Analyze Comments", type="primary"):
    # Now we only need to check if the URL was provided
    if not youtube_url:
        st.warning("Please enter a YouTube URL to analyze.")
    else:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please make sure it's a valid link to a YouTube video.")
        else:
            with st.spinner(f"Fetching comments for video ID: {video_id}..."):
                comments = extract_youtube_video_comments(video_id)

            # The 'comments' variable can be None if there was an API error
            if comments:
                st.info(f"Successfully fetched {len(comments)} English comments.")
                with st.spinner("Analyzing sentiment with SVM model..."):
                    predictions, avg_sentiment = analyze_sentiment_svm(comments)
                
                st.subheader("Overall Sentiment Analysis")

                # Display a metric showing the percentage of positive comments
                st.metric(label="Positive Comment Ratio", value=f"{avg_sentiment:.2%}")

                # Determine overall review based on the average sentiment
                if avg_sentiment > 0.55:
                    st.success(f"**Overall Review: Positive**")
                    st.balloons()
                elif avg_sentiment < 0.45:
                    st.error(f"**Overall Review: Negative**")
                else:
                    st.warning(f"**Overall Review: Mixed / Neutral**")

                # Display sample predictions
                st.write("---")
                st.subheader("Sample Comment Predictions")
                sentiment_map_rev = {0: "Negative", 1: "Positive"}
                for i, comment in enumerate(comments[:5]):
                    sentiment_label = sentiment_map_rev[predictions[i]]
                    st.markdown(f"> {comment}\n\n**Predicted Sentiment:** `{sentiment_label}`")
                        
            elif comments is not None: # This case handles when 0 comments were found
                 st.warning("Could not find any English comments to analyze for this video.")