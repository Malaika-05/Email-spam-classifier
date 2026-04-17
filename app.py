from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize app
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# NLP setup
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Preprocessing function (SAME as training)
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]

    return " ".join(words)

# Feature extraction (important: match training)
def extract_features(text):
    clean = preprocess(text)

    tfidf_features = vectorizer.transform([clean]).toarray()

    char_count = len(text)
    word_count = len(text.split())
    digit_ratio = sum(c.isdigit() for c in text) / len(text) if len(text) > 0 else 0

    extra_features = np.array([[char_count, word_count, digit_ratio]])

    return np.hstack((tfidf_features, extra_features))

# Home route (optional UI)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'message' not in data:
        return jsonify({'error': 'No message provided'})

    message = data['message']

    features = extract_features(message)
    prediction = model.predict(features)[0]

    result = "Spam" if prediction == 1 else "Ham"

    return jsonify({
        'message': message,
        'prediction': result
    })

# Run app
if __name__ == '__main__':
    app.run(debug=True)