from flask import Flask, render_template, request
import joblib
from pathlib import Path
import numpy as np

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load clean model
model = joblib.load(MODEL_DIR / "fake_news_svm_clean.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer_clean.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form.get("news", "").strip()

        if news_text:
            transformed_text = vectorizer.transform([news_text])
            predicted_label = model.predict(transformed_text)[0]

            decision_score = model.decision_function(transformed_text)[0]

            # Confidence-like score, not a probability
            confidence = round(
                1 / (1 + np.exp(-abs(decision_score))) * 100, 2
            )

            if predicted_label == 0:
                prediction = "FAKE NEWS"
            else:
                prediction = "REAL NEWS"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text
    )


if __name__ == "__main__":
    app.run(debug=True)