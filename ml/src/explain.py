# ml/src/explain.py
import numpy as np

def top_terms_logreg(vectorizer, clf, top_k: int = 10):
    """
    For Logistic Regression models, return top terms most indicative of the positive (fraudulent) class.
    """
    if not hasattr(clf, "coef_"):
        return []
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]   # <-- FIX: use row 0
    top_idx = np.argsort(coefs)[-top_k:][::-1]
    return [(feature_names[i], float(coefs[i])) for i in top_idx]


RULE_WEIGHTS = {
    "payment_request": 0.4,
    "personal_info": 0.5,
    "whatsapp_only": 0.3,
    "urgency": 0.2
}

def keyword_risks(text: str):
    """
    Rule-based keyword risk detection.
    """
    if not text:
        return [], 0.0

    lower = text.lower()
    risks = []
    risk_score = 0.0

    rules = {
        "payment_request": [
            "registration fee", "training fee", "pay upfront"
        ],
        "whatsapp_only": ["whatsapp only"],
        "personal_info": [
            "aadhaar", "pan", "bank account", "credit card"
        ],
        "urgency": [
            "urgent hiring", "limited slots", "act now"
        ]
    }

    for label, kws in rules.items():
        hits = [k for k in kws if k in lower]
        if hits:
            risks.append({"rule": label, "keywords": hits})
            risk_score += RULE_WEIGHTS[label]

    return risks, min(risk_score, 1.0)
