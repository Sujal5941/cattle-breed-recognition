"""
view_db.py — quick look at what's stored in the CattleVision AI database.
Run from the backend/ folder:
    python view_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "cattlevision.db")

if not os.path.exists(DB_PATH):
    print(f"No database found yet at {DB_PATH}")
    print("It's created the first time you start the server (uvicorn main:app).")
    raise SystemExit

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

print("=== Users ===")
for row in conn.execute("SELECT id, username, created_at FROM users"):
    print(dict(row))

print("\n=== Predictions ===")
for row in conn.execute(
    "SELECT id, user_id, predicted_breed, confidence, low_confidence, created_at "
    "FROM predictions ORDER BY created_at DESC LIMIT 20"
):
    print(dict(row))

conn.close()
