import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from preprocess import preprocess_text
from evaluate import evaluate_model
from utils import ensure_dir, save_json
from explain import top_terms_logreg

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

# Paths
DATA_PATH = "ml/data/raw/real_or_fake_jobs.csv"
MODEL_DIR = "ml/models"
REPORT_DIR = "ml/reports"

# Ensure output dirs
ensure_dir(MODEL_DIR)
ensure_dir(REPORT_DIR)

# Load dataset
logger.info("Loading dataset...")
df = pd.read_csv(DATA_PATH)
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

df[TEXT_COLS] = df[TEXT_COLS].fillna("")
df["text"] = df[TEXT_COLS].agg(" ".join, axis=1)
df["text"] = df["text"].apply(preprocess_text)

X = df["text"]
y = df["fraudulent"]

# Train/test split
logger.info("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Vectorize
logger.info("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train models
logger.info("Training Logistic Regression...")
logreg = LogisticRegression(
    max_iter=2000,
    class_weight={0: 1.5, 1: 1}
)
logreg.fit(X_train_vec, y_train)

logger.info("Training Naive Bayes...")
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_vec, y_train)

# Evaluate
logger.info("Evaluating all models...")
results = {}

models = {
    "logreg": logreg,
    "nb": nb
}

results = {}

for name, model in models.items():
    logger.info(f"Evaluating {name}...")

    report, cm = evaluate_model(
        model,
        X_test_vec,
        y_test,
        out_path=f"{REPORT_DIR}/{name}_metrics.json"
    )

    results[name] = {
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

# Save best model (LogReg)

logger.info("Saving trained models...")

# Primary production model
joblib.dump(logreg, f"{MODEL_DIR}/classifier.joblib")

# Explicit individual models
joblib.dump(logreg, f"{MODEL_DIR}/logreg.joblib")
joblib.dump(nb, f"{MODEL_DIR}/nb.joblib")

# Vectorizer
joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.joblib")

# Save top terms
logger.info("Extracting top terms...")
top_terms = top_terms_logreg(vectorizer, logreg, top_k=20)
save_json(top_terms, f"{REPORT_DIR}/top_terms.json")

# Save summary
logger.info("Saving summary report...")
save_json(results, f"{REPORT_DIR}/all_model_metrics.json")

logger.info("Training complete.")

# Save threshold for backend use
from utils import save_json

save_json(
    {"threshold": 0.65},
    f"{REPORT_DIR}/threshold.json"
)

