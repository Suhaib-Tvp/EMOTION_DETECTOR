import re
from nltk.corpus import stopwords
import joblib

stop_words = set(stopwords.words('english'))
kmeans = joblib.load('kmeans_model.pkl')  # Load your clustering model
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Same vectorizer used in clustering

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def check_alert(cleaned_texts):
    X_new = vectorizer.transform(cleaned_texts)
    clusters_new = kmeans.predict(X_new)
    if sum(clusters_new == 2) > 3:
        st.error("⚠️ Alert: High volume of negative emotion tweets detected!")
