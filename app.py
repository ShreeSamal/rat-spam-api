from flask import Flask, request, jsonify
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load the vectorizer and model
vectorizer = pickle.load(open("/assets/dumps/vectorizer.pkl", "rb"))
model = pickle.load(open("/assets/dumps/model.pkl", "rb"))

# Download NLTK resources
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

def transform_msg(msg):
    msg = msg.lower()
    msg = nltk.word_tokenize(msg)
    new = []
    for i in msg:
        if i.isalnum() and (i not in stopwords.words('english')):
            new.append(ps.stem(i))
    return " ".join(new)

@app.route('/')
def index():
    return "Hello, World! "

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
    transformed_message = transform_msg(message)
    X = vectorizer.transform([transformed_message]).toarray()
    prediction = model.predict(X)[0]
    return jsonify({'prediction': int(prediction)})

