from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained ML model (adjust path if needed)
model = joblib.load("ml_model/water_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")  # Your homepage

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Example: {"feature1": 10, "feature2": 5}
        data = request.json
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

