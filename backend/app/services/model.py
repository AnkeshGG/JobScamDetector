# backend/app/services/model.py
import joblib
from ml.src.preprocess import preprocess_text

class ScamModel:
    def __init__(self, vectorizer_path, classifier_path):
        self.vectorizer = joblib.load(vectorizer_path)
        self.clf = joblib.load(classifier_path)

    def predict(self, text: str):
        pre_text = preprocess_text(text)
        vec = self.vectorizer.transform([pre_text])
        label = self.clf.predict(vec)[0]
        confidence = max(self.clf.predict_proba(vec)[0])
        return label, confidence, pre_text
