import streamlit as st
import joblib
from utils import preprocess_text, check_alert
import pandas as pd

# Load model and vectorizer
model = joblib.load('svm_model_linear.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Emotion Detection from Tweets")

user_input = st.text_area("Enter a tweet or multiple tweets (separated by new lines):")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        texts = user_input.strip().split('\n')
        cleaned = [preprocess_text(t) for t in texts]
        X_new = vectorizer.transform(cleaned)
        preds = model.predict(X_new)

        df = pd.DataFrame({'Tweet': texts, 'Predicted Emotion': preds})
        st.write(df)

        check_alert(cleaned)  # Uses clustering to detect alerts
