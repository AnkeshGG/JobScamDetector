from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from backend.app.services.model import ScamModel
from backend.app.services.storage import insert_prediction, list_history
from ml.src.explain import keyword_risks, top_terms_logreg

router = APIRouter()
model = ScamModel("ml/models/vectorizer.joblib", "ml/models/classifier.joblib")

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
def predict(req: PredictRequest):
    # Convert Pydantic model → dict
    job_payload = req.dict()

    # ✅ NEW predict signature
    label, confidence, pre_text = model.predict(job_payload)

    # Explanation
    explanation = {
        "risks": keyword_risks(pre_text)
    }

    try:
        explanation["top_terms"] = top_terms_logreg(
            model.vectorizer,
            model.clf
        )
    except Exception:
        explanation["top_terms"] = []

    # Store in DB
    insert_prediction(
        pre_text,
        label,
        confidence,
        explanation
    )

    return {
        "label": "Fake" if label == 1 else "Real",
        "confidence": round(confidence * 100, 2),
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
