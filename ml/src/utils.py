# ml/src/utils.py
import os
import json

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    """Save Python object as JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    """Load JSON file into Python object."""
    with open(path) as f:
        return json.load(f)
