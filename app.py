import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model which we have saved previously
model = tf.keras.models.load_model('final_text_classification_model.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define sentiment labels as we are having 4 labels Negative Positive Neutral Irrelevant.
sentiment_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral', 3: 'Irrelevant'}

# Lets create a streamlit application 
st.title("📝 Sentiment Analysis(Text Classifiaction) App BY RVRJC-CSM-2025")
st.write("Enter a sentence to analyze its sentiment.")

# Text input which can be positive neutral negative also it can be irrelevant to our review dataset
user_input = st.text_area("Enter your text:")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess input text
        sequences = tokenizer.texts_to_sequences([user_input])
        padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
        
        # Predict sentiment
        prediction = model.predict(padded_sequences)
        sentiment = np.argmax(prediction)
        
        # Display result
        st.success(f"Predicted Sentiment: {sentiment_labels[sentiment]}")
    else:
        st.warning("Please enter some text to analyze.")
