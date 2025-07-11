import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Load and Train Model
df = pd.read_csv("product_reviews.csv")
df['sentiment'] = df['sentiment'].str.lower().str.strip()
df['clean_review'] = df['review'].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']
model = MultinomialNB()
model.fit(X, y)

# Streamlit App
st.set_page_config(page_title="Sentiment Analyzer", layout="centered", page_icon="💬")

st.title("💬 Product Review Sentiment Analyzer")
st.markdown("This app predicts the sentiment (Positive/Negative/Neutral) of a product review using **Naive Bayes**.")

st.markdown("---")

# Input Box
review = st.text_area("Enter your product review here:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter a product review.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"**Predicted Sentiment:** {prediction.capitalize()}")

st.markdown("---")
st.markdown("<center>Made with ❤️ using Naive Bayes</center>", unsafe_allow_html=True)
