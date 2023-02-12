import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
classifier = pickle.load(open("model.pkl", "rb"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    pred = classifier.predict(features)
    return render_template("index.html", prediction_text = "{}".format(pred[0]))

if __name__ == "__main__":
    app.run(debug=True)