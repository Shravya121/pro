from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

# Load the saved model and label encoder
pipeline = joblib.load("xgb_text_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Clean user input
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_text = request.form["text"]
        cleaned = clean_text(input_text)
        pred_encoded = pipeline.predict([cleaned])[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
