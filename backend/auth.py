"""
auth.py — password hashing and JWT session tokens.

Passwords are hashed with PBKDF2-HMAC-SHA256 from Python's standard library
`hashlib` (no compiled dependency like bcrypt, which avoids the exact kind
of Windows build-tool problems we hit earlier with `tokenizers`).

Sessions are JSON Web Tokens (PyJWT, pure Python). The signing key is
generated once and saved to backend/data/secret.key so tokens keep working
across server restarts; if that file is ever deleted, all existing sessions
are invalidated and everyone needs to log in again.
"""

import datetime
import hashlib
import hmac
import os
import secrets

import jwt
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from database import DATA_DIR, get_db
from db_models import User

SECRET_KEY_PATH = os.path.join(DATA_DIR, "secret.key")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24 * 14  # 2 weeks


def _get_secret_key():
    if os.path.exists(SECRET_KEY_PATH):
        with open(SECRET_KEY_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(SECRET_KEY_PATH, "w", encoding="utf-8") as f:
        f.write(key)
    return key


SECRET_KEY = _get_secret_key()


def hash_password(password, salt_hex=None):
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 100_000
    )
    return digest.hex(), salt_hex


def verify_password(password, password_hash, salt_hex):
    computed, _ = hash_password(password, salt_hex)
    return hmac.compare_digest(computed, password_hash)


def create_token(user: User):
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


def _extract_bearer_token(authorization):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[len("Bearer "):]


def get_current_user_optional(
    authorization: str = Header(None), db: Session = Depends(get_db)
):
    """Returns the User if a valid token was sent, otherwise None. Never raises."""
    token = _extract_bearer_token(authorization)
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    return db.query(User).filter(User.id == int(payload["sub"])).first()


def get_current_user_required(
    authorization: str = Header(None), db: Session = Depends(get_db)
):
    """Same as above, but raises 401 if there's no valid logged-in user."""
    user = get_current_user_optional(authorization, db)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    return user


def get_current_user_for_image(
    request: Request, authorization: str = Header(None), db: Session = Depends(get_db)
):
    """
    Same as get_current_user_required, but also accepts the token as a
    ?token= query parameter. Needed because <img> tags in the browser can't
    send a custom Authorization header, so stored images are loaded with a
    token in the URL instead.
    """
    token = _extract_bearer_token(authorization) or request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    return user
