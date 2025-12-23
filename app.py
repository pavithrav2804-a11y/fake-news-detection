import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)
data["content"] = data["title"] + " " + data["text"]
data["clean_text"] = data["content"].apply(clean_text)

X = data["clean_text"]
y = data["label"]

tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
X_tfidf = tfidf.fit_transform(X)

model = LogisticRegression(max_iter=2000)
model.fit(X_tfidf, y)

st.title("📰 Fake News Detection App")

news_input = st.text_area("Enter News Text")

if st.button("Check News"):
    cleaned = clean_text(news_input)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        st.success("✅ Real News")
    else:
        st.error("❌ Fake News")
