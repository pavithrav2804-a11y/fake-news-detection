import streamlit as st
import pandas as pd
import re
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

@st.cache_data
def load_data():
    with zipfile.ZipFile("data/Fake.zip", "r") as z:
        with z.open("Fake.csv") as f:
            fake = pd.read_csv(f)

    with zipfile.ZipFile("data/True.zip", "r") as z:
        with z.open("True.csv") as f:
            true = pd.read_csv(f)

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true], ignore_index=True)

    data["title"] = data["title"].fillna("")
    data["text"] = data["text"].fillna("")

    data["content"] = data["title"] + " " + data["text"]
    data["clean_text"] = data["content"].apply(clean_text)

    return data

data = load_data()

X = data["clean_text"]
y = data["label"]


tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
X_tfidf = tfidf.fit_transform(X)

model = LogisticRegression(max_iter=2000)
model.fit(X_tfidf, y)

st.title("📰 Fake News Detection App")

news_input = st.text_area("Enter News Text")

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text")
    else:
        cleaned = clean_text(news_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("Real News")
        else:
            st.error("Fake News")
