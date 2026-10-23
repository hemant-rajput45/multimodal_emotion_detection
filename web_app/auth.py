"""
Authentication & User Management Module
Supports SQLite database storage, password hashing, pre-seeded research accounts,
and session validation.
"""

import os
import sqlite3
import hashlib

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = "multimodal_emotion_salt_2026"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database and seed default researcher accounts"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'Researcher',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Check if default accounts exist
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]

    if count == 0:
        default_users = [
            ("Sai Bhadre", "sai.bhadre@indiraicem.ac.in", hash_password("sai123"), "Lead Researcher (SY MCA)"),
            ("Prof. Vaishali Hatkar", "vaishali.hatkar@indiraicem.ac.in", hash_password("guide123"), "Project Guide (Assistant Professor)"),
            ("Guest Evaluator", "evaluator@sppu.ac.in", hash_password("demo123"), "External Reviewer")
        ]
        cursor.executemany("""
            INSERT INTO users (name, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        """, default_users)
        conn.commit()

    conn.close()

def authenticate_user(email, password):
    """Authenticate user credentials, returns user dict or None"""
    conn = get_db_connection()
    cursor = conn.cursor()

    hashed = hash_password(password)
    cursor.execute("""
        SELECT id, name, email, role FROM users
        WHERE email = ? AND password_hash = ?
    """, (email.strip().lower(), hashed))

    user = cursor.fetchone()
    conn.close()

    if user:
        return dict(user)
    return None

def register_user(name, email, password, role="Researcher"):
    """Register a new user account"""
    conn = get_db_connection()
    cursor = conn.cursor()

    email_clean = email.strip().lower()
    hashed = hash_password(password)

    try:
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        """, (name.strip(), email_clean, hashed, role))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {"success": True, "user": {"id": user_id, "name": name, "email": email_clean, "role": role}}
    except sqlite3.IntegrityError:
        conn.close()
        return {"success": False, "message": "An account with this email address already exists."}
    except Exception as e:
        conn.close()
        return {"success": False, "message": str(e)}

# Auto-initialize DB on import
init_db()
