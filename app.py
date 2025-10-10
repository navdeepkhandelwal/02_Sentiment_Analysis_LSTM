import streamlit as st
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("sentiment_model.keras")
word_index = imdb.get_word_index()

def text_to_sequence(text, num_words=2000, maxlen=200):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    sequence = []
    for word in words:
        index = word_index.get(word)
        if index is not None and index < num_words:
            sequence.append(index)
        else:
            sequence.append(2)
    return pad_sequences([sequence], maxlen=maxlen, padding='post', truncating='post')

st.title("🎭 Sentiment Analysis (IMDB + LSTM)")
user_input = st.text_area("Enter movie review text:")
if st.button("Predict"):
    x = text_to_sequence(user_input)
    pred = model.predict(x)
    sentiment = "Positive 😊" if pred[0][0] > 0.5 else "Negative 😠"
    st.subheader(f"Prediction: {sentiment}")
    st.write(f"Confidence: {float(pred[0][0]):.2f}")
