# backend/app/db/connection.py
import sqlite3

DB_PATH = "app.db"

def get_connection():
    """Return a new SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
