# 📰 Fake News Detection using Machine Learning

This project detects whether a given news article is **Real** or **Fake**
using Natural Language Processing and Machine Learning techniques.

---

## 🚀 Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 📂 Dataset
Due to GitHub file size limits, the dataset is not uploaded to this repository.

Dataset used:
**Fake and Real News Dataset (Kaggle)**

🔗 Link:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Files:
- Fake.csv → Fake news articles
- True.csv → Real news articles

---

## ⚙️ Project Workflow
1. Load and combine Fake and Real news datasets
2. Clean the text data
3. Convert text into numerical features using TF-IDF
4. Train Logistic Regression model
5. Evaluate model accuracy
6. Predict whether news is Real or Fake

---

## 🎯 Accuracy
- Achieved approximately **99% accuracy** on test data

---

## ▶️ How to Run the Project
1. Download `Fake.csv` and `True.csv` from Kaggle
2. Place them in the same folder as the notebook
3. Open `fake_news_detection.ipynb`
4. Run all cells
5. Test the model using the `predict_news()` function

---

## ✨ Example
```python
predict_news("Government announces new education policy")
