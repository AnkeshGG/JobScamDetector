#backend/app/main.py
from fastapi import FastAPI
from backend.app.api.routes import router
from backend.app.services.storage import init_db

app = FastAPI(
    title="Job Scam Detector API",
    description="API for detecting fraudulent job postings",
    version="1.0.0"
)

app.include_router(router)

@app.on_event("startup")
def on_startup():
    init_db()
