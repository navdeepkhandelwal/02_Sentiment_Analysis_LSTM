import streamlit as st
import numpy as np
import re
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Load model (.h5 format)
model = load_model("sentiment_model.h5")

# ✅ Load saved tokenizer (word index)
word_index = joblib.load("word_index.joblib")

# ✅ Convert input text to padded sequence
def text_to_sequence(text, word_index, num_words=2000, maxlen=200):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    seq = []
    for word in text.split():
        idx = word_index.get(word)
        if idx is not None and idx < num_words:
            seq.append(idx)        # 👈 No +3 offset (since you trained without it)
        else:
            seq.append(2)          # Unknown token
    return pad_sequences([seq], maxlen=maxlen, padding='post', truncating='post')

# ✅ Streamlit UI
st.title("🎭 Sentiment Analysis (IMDB + LSTM)")
user_input = st.text_area("Enter movie review text:")

if st.button("Predict"):
    if user_input.strip():
        seq = text_to_sequence(user_input, word_index)
        pred = model.predict(seq)
        sentiment = "Positive 😊" if pred[0][0] > 0.5 else "Negative 😠"
        st.subheader(f"Prediction: {sentiment}")
        st.write(f"Confidence: {float(pred[0][0]):.2f}")
    else:
        st.warning("Please enter some text before predicting.")
