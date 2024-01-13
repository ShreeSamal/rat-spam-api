from flask import Flask, request, jsonify
import pickle
import nltk
import requests
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load the vectorizer and model
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

def notifyParent(msg,to):
    fcm_url = "https://fcm.googleapis.com/fcm/send"
    fcm_server_key = "AAAAtD1yPG0:APA91bH5XpUJdmXBW71cQ2AP9kZt6AuVf7RL4DX-OP4enQOih6_Zs8eAXgPEVaBg6isO-XMAqZfBTYqZftzeBJPrxzJkL-zJQpKkeF1LfxLfeRC3tzdpCCQ4F12KViTBLm5IhWpm1vYa"  # Replace with your FCM server key

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'key=' + fcm_server_key,
    }

    payload = {
        "to": to,
        "notification": {
            "body": str(msg),
            "title": "spam alert",
            "subtitle": "sms"
        }
    }

    response = requests.post(fcm_url, headers=headers, json=payload)
    print(response)

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
    message = data['messages']
    to = str(data['to'])
    for msg in message:
        transformed_message = transform_msg(msg)
        X = vectorizer.transform([transformed_message]).toarray()
        prediction = model.predict(X)[0]
        if prediction == 1:
            notifyParent(msg,to)
    return jsonify({'response': "done"})

