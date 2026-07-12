from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["N"]),
        float(request.form["P"]),
        float(request.form["K"]),
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["ph"]),
        float(request.form["rainfall"])
    ]

    prediction = model.predict([features])[0]

    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)