from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from backend.app.services.model import ScamModel
from backend.app.services.storage import insert_prediction, list_history
from ml.src.explain import keyword_risks, top_terms_logreg

router = APIRouter()
model = ScamModel()


class PredictRequest(BaseModel):
    title: Optional[str] = ""
    description: Optional[str] = ""
    requirements: Optional[str] = ""
    company_profile: Optional[str] = ""
    benefits: Optional[str] = ""
    industry: Optional[str] = ""
    employment_type: Optional[str] = ""
    location: Optional[str] = ""
    salary_range: Optional[str] = ""


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")
@router.post("/predict")
def predict(req: PredictRequest):
    job_payload = req.dict()

    ml_prob, pre_text = model.predict(job_payload)

    risks, rule_score = keyword_risks(pre_text)

    final_score = 0.7 * ml_prob + 0.3 * rule_score

    label = 1 if final_score >= 0.5 else 0

    explanation = {
        "risks": risks
    }

    try:
        explanation["top_terms"] = top_terms_logreg(
            model.vectorizer,
            model.logreg
        )
    except Exception:
        explanation["top_terms"] = []

    insert_prediction(pre_text, label, final_score, explanation)

    return {
        "label": "Fake" if label == 1 else "Real",
        "confidence": round(final_score * 100, 2),
        "ml_probability": round(ml_prob * 100, 2),
        "rule_score": round(rule_score * 100, 2),
        "explanation": explanation
    }

@router.get("/history")
def history(limit: int = 50, offset: int = 0):
    rows = list_history(limit, offset)
    return [
        {
            "id": r[0],
            "text": r[1],
            "label": "Fake" if r[2] == 1 else "Real",
            "confidence": round(r[3] * 100, 2),
            "explanation": r[4],
            "created_at": r[5]
        }
        for r in rows
    ]
