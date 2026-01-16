# backend/app/services/validation.py
from typing import Tuple

def validate_job_text(text: str) -> Tuple[bool, str]:
    """
    Validate job posting text before sending to model.
    Returns (is_valid, error_message).
    """
    if not text or len(text.strip()) < 20:
        return False, "Job posting text must be at least 20 characters."
    if len(text) > 10000:
        return False, "Job posting text too long (max 10k chars)."
    return True, ""
