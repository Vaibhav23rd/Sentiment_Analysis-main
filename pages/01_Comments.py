import streamlit as st
# Import the prediction function from your custom model loader
from model_loader import predict_single_comment_svm

# --- Streamlit UI and Logic ---

st.set_page_config(page_title="Single Comment Analysis", layout="centered")

st.title("Single Text Sentiment Analysis (SVM)")
st.markdown(
    "Enter any text—like a course review or a single comment—to classify its "
    "sentiment as **Positive** or **Negative** using the custom-trained SVM model."
)

# Use st.text_area for a larger, more user-friendly input box
comment_text = st.text_area(
    "Enter the text to analyze:",
    height=150,
    placeholder="e.g., 'The instructor explained the concepts clearly and the course was very engaging.'"
)

# Use a primary button for the main action
if st.button("Analyze Sentiment", type="primary"):
    # First, validate that the user has entered some text
    if comment_text:
        # Show a spinner to indicate that processing is happening
        with st.spinner("Classifying..."):
            # Call your SVM model for prediction (returns 0 for negative, 1 for positive)
            prediction = predict_single_comment_svm(comment_text)

        st.subheader("Analysis Result")

        # Display the result in a visually distinct and clear way
        if prediction == 1:
            st.success("Sentiment: Positive")
            st.markdown("✅ The model predicts this text has a **positive** sentiment.")
        else: # This means the prediction was 0
            st.error("Sentiment: Negative")
            st.markdown("❌ The model predicts this text has a **negative** sentiment.")

    else:
        # Show a warning if the text area is empty
        st.warning("Please enter some text to analyze.")