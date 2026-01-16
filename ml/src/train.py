import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
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
df = df.dropna(subset=["description", "fraudulent"])
df["text"] = df["description"].apply(preprocess_text)
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
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train_vec, y_train)

logger.info("Training Naive Bayes...")
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_vec, y_train)

logger.info("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_vec, y_train)

# Evaluate
logger.info("Evaluating models...")
results = {}
for name, model in [("logreg", logreg), ("nb", nb), ("rf", rf)]:
    logger.info(f"Evaluating {name}...")
    report, cm = evaluate_model(model, X_test_vec, y_test, out_path=f"{REPORT_DIR}/{name}_metrics.json")
    results[name] = {
        "report": report,
        "confusion_matrix": cm.tolist()
    }

# Save best model (LogReg)
logger.info("Saving best model...")
joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.joblib")
joblib.dump(logreg, f"{MODEL_DIR}/classifier.joblib")

# Save top terms
logger.info("Extracting top terms...")
top_terms = top_terms_logreg(vectorizer, logreg, top_k=20)
save_json(top_terms, f"{REPORT_DIR}/top_terms.json")

# Save summary
logger.info("Saving summary report...")
save_json(results, f"{REPORT_DIR}/metrics.json")

logger.info("Training complete.")
