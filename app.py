import streamlit as st
import pickle
import numpy as np
import re
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# -----------------------------------
# 1. Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Spam Message Classifier")
st.write("Enter a message to check whether it is Spam or Ham.")

# -----------------------------------
# 2. Load model and vectorizer
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

# -----------------------------------
# 3. Load NLP resources
# -----------------------------------
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

model, vectorizer = load_model()
stop_words = load_stopwords()
ps = PorterStemmer()

# -----------------------------------
# 4. Text preprocessing
# -----------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------------
# 5. Feature extraction
# -----------------------------------
def extract_features(text):
    clean = preprocess(text)

    tfidf_features = vectorizer.transform(
        [clean]
    ).toarray()

    char_count = len(text)
    word_count = len(text.split())

    digit_ratio = (
        sum(char.isdigit() for char in text) / len(text)
        if len(text) > 0
        else 0
    )

    extra_features = np.array([[
        char_count,
        word_count,
        digit_ratio
    ]])

    return np.hstack((
        tfidf_features,
        extra_features
    ))

# -----------------------------------
# 6. User input
# -----------------------------------
message = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! You have won a prize...",
    height=150
)

# -----------------------------------
# 7. Predict Spam or Ham
# -----------------------------------
if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        try:
            features = extract_features(message)
            prediction = model.predict(features)[0]

            if prediction == 1:
                st.error("🚨 Prediction: Spam")
                st.write(
                    "This message has been classified as spam."
                )
            else:
                st.success("✅ Prediction: Ham")
                st.write(
                    "This message has been classified as legitimate."
                )

            with st.expander("View submitted message"):
                st.write(message)

        except Exception as e:
            st.error("Prediction failed. Please check the model files.")
            st.exception(e)
