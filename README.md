# 😃 Emotion Detection from Tweets

A **Streamlit-based web application** that analyzes emotions in tweets using a **Support Vector Machine (SVM)** model and **TF-IDF vectorization**.  
The app also includes an alert detection feature using text clustering to identify potentially critical tweets.

---

## 🚀 Overview

This project detects the **emotional tone** of text data (such as tweets) and categorizes them into predefined emotion labels (e.g., *joy, anger, sadness, fear,* etc.).  
Users can enter one or more tweets, and the model will predict the dominant emotion for each.

---

## 🧠 Features

- 🧹 **Text Preprocessing** – cleans and normalizes tweet text (handled in `utils.py`).  
- 🤖 **SVM Classifier** – trained on TF-IDF features to predict emotions.  
- 🧠 **TF-IDF Vectorization** – converts input text into numerical vectors for machine learning.  
- 🚨 **Alert Detection** – detects critical emotional patterns using clustering (`check_alert()` function).  
- ⚡ **Streamlit UI** – simple, interactive interface for emotion analysis.  

---

## 🗂️ Project Structure

emotion-detection/
│
├── app.py # Main Streamlit application
├── utils.py # Helper functions (preprocessing & alert detection)
├── svm_model_linear.pkl # Trained SVM model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Project dependencies
└── README.md # Project documentation

📈 Future Enhancements

Integrate deep learning models like BERT or RoBERTa for better accuracy.

Expand emotion categories beyond basic classes.

Add real-time Twitter integration (using Twitter API).

Visualize emotion distribution with charts.
