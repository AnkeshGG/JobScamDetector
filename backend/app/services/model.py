# backend/app/services/model.py
import joblib
from ml.src.preprocess import preprocess_text

class ScamModel:
    def __init__(self, vectorizer_path, classifier_path):
        self.vectorizer = joblib.load(vectorizer_path)
        self.clf = joblib.load(classifier_path)

    def predict(self, job: dict):
        """
        Predict whether a job is fake or real.
        job: dict containing all job fields
        """

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

        raw_text = " ".join([job.get(col, "") for col in TEXT_COLS])
        pre_text = preprocess_text(raw_text)
        vec = self.vectorizer.transform([pre_text])
        fraud_prob = self.clf.predict_proba(vec)[0][1]
        THRESHOLD = 0.65
        label = int(fraud_prob > THRESHOLD)
        confidence = float(fraud_prob)

        return label, confidence, pre_text

