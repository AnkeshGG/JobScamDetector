#backend/app/api/routes.py
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.services.model import ScamModel
from backend.app.services.storage import insert_prediction, list_history
from ml.src.explain import keyword_risks, top_terms_logreg

router = APIRouter()
model = ScamModel("ml/models/vectorizer.joblib", "ml/models/classifier.joblib")

class PredictRequest(BaseModel):
    text: str = Field(min_length=20)

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict")
def predict(req: PredictRequest):
    label, confidence, pre_text = model.predict(req.text)
    explanation = {"risks": keyword_risks(req.text)}
    try:
        explanation["top_terms"] = top_terms_logreg(model.vectorizer, model.clf)
    except Exception:
        explanation["top_terms"] = []
    insert_prediction(req.text, label, confidence, explanation)
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
