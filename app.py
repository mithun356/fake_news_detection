import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Streamlit Page Configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)

# Set the path to your model
MODEL_PATH = r"fake_news_cnn_model_V2.h5"

# Load the trained model with error handling
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.stop()  # Stop execution if the model fails to load

# Tokenizer configuration (adjust as per your training configuration)
MAX_SEQUENCE_LENGTH = 500  # Match the length used in training
VOCAB_SIZE = 10000  # Match the tokenizer vocab size used in training

# Create or load tokenizer
# Replace this placeholder tokenizer with the one used during training
tokenizer = Tokenizer(num_words=VOCAB_SIZE)

# App Title and Description
st.title("📰 Fake News Detection App")
st.write("Determine whether a news article is **real** or **fake** using a trained CNN model.")

# User Input
st.write("### Enter the news text below:")
user_input = st.text_area("News Text", placeholder="Paste the news article here...", height=200)

if st.button("Predict"):
    if user_input.strip():
        try:
            # Preprocess the input text
            sequences = tokenizer.texts_to_sequences([user_input])  # Tokenize input text
            padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # Pad to match training data

            # Make prediction
            prediction = model.predict(padded_sequences)[0][0]

            # Display the result
            if prediction > 0.5:
                st.success("🛑 This news article is predicted to be **Fake News**.")
            else:
                st.success("✅ This news article is predicted to be **Real News**.")
            
            # Display prediction confidence
            st.write(f"### Prediction Confidence: {prediction * 100:.2f}% Fake")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Please enter some text for prediction.")

# About Section
st.sidebar.title("About")
st.sidebar.info(
    """
    **Fake News Detection App**  
    This tool uses a CNN-based model trained on labeled datasets of real and fake news articles.  
    Created with ❤️ using [Streamlit](https://streamlit.io) and [TensorFlow](https://www.tensorflow.org).  
    """
)


