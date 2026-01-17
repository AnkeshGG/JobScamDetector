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


def keyword_risks(text: str):
    """
    Rule-based keyword risk detection.
    """
    risks = []
    rules = {
        "payment_request": ["wire transfer","western union","pay fee","processing fee","training fee","fee", "payment", "deposit"],
        "whatsapp_only": "whatsapp" in text,
        "personal_info": ["ssn","aadhaar","pan","passport","bank account","credit card"],
        "contact_only": ["whatsapp","telegram","dm me","text only"],
        "urgent_hiring": "urgent" in text,
        "urgency": ["urgent","immediate joining","limited slots","act now"],
    }
    lower = text.lower()
    for label, kws in rules.items():
        hits = [k for k in kws if k in lower]
        if hits:
            risks.append({"rule": label, "keywords": hits})
    return risks
