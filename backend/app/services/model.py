import joblib
import json
from ml.src.preprocess import preprocess_text


class ScamModel:
    def __init__(self):
        # Load vectorizer
        self.vectorizer = joblib.load("ml/models/vectorizer.joblib")

        # Load primary classifier (Logistic Regression)
        self.logreg = joblib.load("ml/models/classifier.joblib")

        # Load Naive Bayes (secondary signal)
        self.nb = joblib.load("ml/models/nb.joblib")

        # Load threshold decided during training
        with open("ml/reports/threshold.json") as f:
            self.threshold = json.load(f)["threshold"]

    def predict(self, job: dict):
        TEXT_COLS = [
            "title",
            "description",
            "requirements",
            "company_profile",
            "benefits",
            "industry",
            "employment_type",
            "location",
            "salary_range"
        ]

        # Build text exactly like training
        raw_text = " ".join(job.get(col, "") for col in TEXT_COLS)
        pre_text = preprocess_text(raw_text)

        vec = self.vectorizer.transform([pre_text])

        # Logistic Regression probability
        fraud_prob = self.logreg.predict_proba(vec)[0][1]

        # Decision logic
        return float(fraud_prob), pre_text
