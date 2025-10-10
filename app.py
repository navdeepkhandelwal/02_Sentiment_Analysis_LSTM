import streamlit as st
import numpy as np
import re
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Load model and saved tokenizer
model = load_model("sentiment_model.keras")
word_index = joblib.load("word_index.joblib")  # <-- uses saved word index instead of imdb.get_word_index()

# ✅ Convert text to sequence using loaded tokenizer
def text_to_sequence(text, word_index, maxlen=200):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    seq = []
    for word in text.split():
        idx = word_index.get(word)
        if idx is not None and idx < 10000:
            seq.append(idx + 3)
        else:
            seq.append(2)
    return pad_sequences([seq], maxlen=maxlen)

# ✅ Streamlit UI
st.title("🎭 Sentiment Analysis (IMDB + LSTM)")
user_input = st.text_area("Enter movie review text:")

if st.button("Predict"):
    if user_input.strip():
        seq = text_to_sequence(user_input, word_index)
        prediction = model.predict(seq)
        sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😠"
        st.subheader(f"Prediction: {sentiment}")
        st.write(f"Confidence: {float(prediction[0][0]):.2f}")
    else:
        st.warning("Please enter some text before predicting.")
