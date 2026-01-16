# ml/src/featurize.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from .utils import ensure_dir
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def build_vectorizer(max_features: int = 20000, ngram_range=(1,2)):
    """
    Create a TF-IDF vectorizer configured for short job-posting text.
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents="unicode",
        lowercase=True,
        norm="l2",
        sublinear_tf=True
    )
    return vec

def save_vectorizer(vectorizer, filename="vectorizer.joblib"):
    ensure_dir(MODELS_DIR)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(vectorizer, path)
    return path
