# backend/app/services/storage.py
import os
import sqlite3
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "jobs.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")


def init_db():
    """Initialize the SQLite database using schema.sql"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Load schema file safely
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema_sql = f.read()

    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()


def insert_prediction(text: str, label: int, confidence: float, explanation: dict):
    """Insert a prediction record into the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (text, label, confidence, explanation)
        VALUES (?, ?, ?, ?)
        """,
        (text, label, confidence, json.dumps(explanation)),
    )
    conn.commit()
    conn.close()


def list_history(limit: int = 50, offset: int = 0):
    """Retrieve prediction history"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, text, label, confidence, explanation, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows
