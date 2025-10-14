# 🎬 IMDB Sentiment Analysis using LSTM

A deep learning-based sentiment analysis project to classify IMDb movie reviews as **Positive** or **Negative** using an **LSTM (Long Short-Term Memory)** network.  
The model trained here powers the **live Streamlit web apps** linked below 👇

---

## 🌐 Live Apps (Deployed on Streamlit Cloud)

| Version | Description | Link |
|----------|--------------|------|
| 🎨 **Advanced UI** | Adjustable threshold, batch `.txt` upload, and confidence display | [Open Advanced App](https://sentimentapp-utifncjl4aubkfgbv2d3sg.streamlit.app) |
| ⚡ **Simple UI** | Quick single-review sentiment prediction | [Open Simple App](https://sentimentapp-ixdtimc7zkczbhnq434dc.streamlit.app) |

> 🧩 **Deployed using:** [navdeepkhandelwal/sentiment_app](https://github.com/navdeepkhandelwal/sentiment_app)

---

## 🎯 Objective
Build a deep learning model to classify IMDb movie reviews as **Positive** or **Negative** based on their textual content using a Long Short-Term Memory (LSTM) network.

---

## 🧠 Dataset
- **Dataset:** IMDb Large Movie Review Dataset (50,000 labeled reviews)  
- **Source:** [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🧩 Methodology

### 1️⃣ Text Preprocessing
- Tokenization, padding, and text cleaning  
- Removal of stopwords and lowercasing  
- Converted text into numerical sequences using **Keras Tokenizer**

### 2️⃣ Model Architecture
- Embedding → LSTM → Dense (Sigmoid)  
- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Activation:** Sigmoid  
- Output: Binary (Positive = 1, Negative = 0)

### 3️⃣ Training Details
| Parameter | Value |
|:-----------|:-------:|
| Epochs | 10 |
| Batch Size | 64 |
| Validation Split | 20% |

---

## 📊 Results
| Metric | Value |
|:--|:--:|
| **Training Accuracy** | 96.07% |
| **Validation Accuracy** | 85.34% |
| **Test Accuracy** | 85.50% |
| **Test Loss** | 0.4515 |

---

### 🧾 Example Prediction
> **Input:** “The movie was absolutely fantastic!”  
> **Predicted Output:** Positive ✅ (Confidence: 0.95)

---

## 📈 Visualization

### Training vs Validation Accuracy
A line plot shows accuracy trends for training and validation sets, helping detect overfitting or underfitting.

![Training vs Validation Accuracy](results/accuracy_plot.png)

---

## 🧪 Example Predictions (Connected to Deployed Model)
| Review | Predicted Sentiment | Confidence |
|--------|---------------------|-------------|
| “The movie was absolutely wonderful with stunning visuals and heartfelt performances.” | Positive ✅ | 0.98 |
| “I couldn’t even finish the movie — it was painfully slow and poorly written.” | Negative 😠 | 0.04 |
| “It wasn’t great, but it wasn’t terrible either — just an average movie.” | Neutral ⚙️ | 0.61 |

---

## 🧰 Tech Stack
- **TensorFlow / Keras** — Model Training  
- **NumPy / Pandas** — Data Preprocessing  
- **Matplotlib** — Visualization  
- **Joblib** — Tokenizer Saving  
- **Streamlit** — Deployment  

---

## 🚀 Deployment
The trained model and tokenizer are exported and used in a separate deployment repository:  
👉 [navdeepkhandelwal/sentiment_app](https://github.com/navdeepkhandelwal/sentiment_app)

- Uses Streamlit Cloud for hosting  
- Two UI versions: *Simple* and *Advanced*  
- Loads pre-trained model `.keras` and tokenizer `.joblib`  
- Displays confidence score and threshold control  

---

## 📚 Future Enhancements
- Add **BERT-based transformer model** for improved accuracy  
- Multi-class sentiment (Positive / Negative / Neutral)  
- Add Flask API for REST integration  

---

## 👨‍💻 Author
**Navdeep Khandelwal**  
📜 Certified in Artificial Intelligence, Machine Learning & Deep Learning — **IIT Delhi (2025)**  
📍 Rajasthan, India  
🔗 [GitHub Profile](https://github.com/navdeepkhandelwal)

---

### 🏁 Summary
> - ✅ Model trained & validated on IMDb dataset  
> - ✅ Saved LSTM model and tokenizer used in live Streamlit deployment  
> - ✅ Advanced + Simple UI deployed  
> - ✅ End-to-end workflow: **Training → Model Saving → Deployment**



