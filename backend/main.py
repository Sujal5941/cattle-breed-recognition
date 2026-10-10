"""
main.py - CattleVision AI backend

Endpoints:
    GET  /api/breeds         -> list of the 17 breed names the model was trained on
    POST /api/predict        -> upload an image, get the real model prediction
                                 (+ attention rollout heatmap), saved to the DB
    GET  /api/health         -> basic status check
    POST /api/auth/register  -> create an account
    POST /api/auth/login     -> log in, get a session token
    GET  /api/auth/me        -> current logged-in user
    GET  /api/history        -> the logged-in user's past predictions
    GET  /api/uploads/{name} -> a stored uploaded image (owner-only)
    POST /api/chat           -> project chatbot (Ollama)

Run with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser - this app also serves the
frontend directory so everything runs from one process.
"""

import io
import json
import os
import uuid

import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from model_loader import load_model
from chatbot import ask_chatbot
from attention_viz import build_attention_overlay
from database import UPLOADS_DIR, get_db, init_db
from db_models import Prediction, User
from auth import (
    create_token,
    get_current_user_for_image,
    get_current_user_optional,
    get_current_user_required,
    hash_password,
    verify_password,
)

app = FastAPI(title="CattleVision AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10
LOW_CONFIDENCE_THRESHOLD = 0.5

_model = None
_processor = None
_id2label = None


@app.on_event("startup")
def _startup():
    global _model, _processor, _id2label
    _model, _processor, _id2label = load_model()
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": len(_id2label) if _id2label else 0}


@app.get("/api/breeds")
def breeds():
    """Returns the actual breed names read from the trained model's label map."""
    if _id2label is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    ordered = [name for _, name in sorted(_id2label.items(), key=lambda kv: kv[0])]
    return {"count": len(ordered), "breeds": ordered}


# --- Auth -------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    username = payload.username.strip()
    if len(username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if len(payload.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(400, "That username is already taken.")

    password_hash, salt = hash_password(payload.password)
    user = User(username=username, password_hash=password_hash, password_salt=salt)
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"token": create_token(user), "username": user.username}


@app.post("/api/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username.strip()).first()
    if not user or not verify_password(payload.password, user.password_hash, user.password_salt):
        raise HTTPException(401, "Incorrect username or password.")
    return {"token": create_token(user), "username": user.username}


@app.get("/api/auth/me")
def me(user: User = Depends(get_current_user_required)):
    return {"username": user.username}


# --- Prediction ---------------------------------------------------------


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user_optional),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a supported image format (JPG, PNG, or WEBP).",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Please choose a smaller image.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        inputs = _processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs, output_attentions=True)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=0)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[predict] inference error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to analyze this image right now. Please try again.",
        )

    scored = [
        {"breed": _id2label[i], "confidence": round(float(probs[i]), 6)}
        for i in range(len(_id2label))
    ]
    scored.sort(key=lambda x: x["confidence"], reverse=True)
    top3 = scored[:3]
    low_confidence = top3[0]["confidence"] < LOW_CONFIDENCE_THRESHOLD

    # Attention rollout heatmap — a real inspection of the model's
    # self-attention, not decorative. Wrapped so a visualization failure
    # never breaks the actual prediction the user came for.
    attention_map = None
    try:
        grid_size = _model.config.image_size // _model.config.patch_size
        attention_map = build_attention_overlay(image, outputs.attentions, grid_size)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[predict] attention visualization failed: {exc}")

    # Save the image + result to the database. Runs for every prediction,
    # logged in or not, building the project's image dataset; user_id is
    # only set when someone is logged in.
    saved_filename = f"{uuid.uuid4().hex}.jpg"
    try:
        image.save(os.path.join(UPLOADS_DIR, saved_filename), format="JPEG", quality=90)
        record = Prediction(
            user_id=user.id if user else None,
            image_filename=saved_filename,
            predicted_breed=top3[0]["breed"],
            confidence=top3[0]["confidence"],
            top_predictions_json=json.dumps(top3),
            low_confidence=low_confidence,
        )
        db.add(record)
        db.commit()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[predict] failed to save prediction record: {exc}")

    return {
        "predicted_breed": top3[0]["breed"],
        "confidence": top3[0]["confidence"],
        "top_predictions": top3,
        "low_confidence": low_confidence,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "attention_map": attention_map,
        "logged_in": user is not None,
    }


# --- History -------------------------------------------------------------


@app.get("/api/history")
def history(
    authorization: str = Header(None),
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    token = authorization[len("Bearer "):] if authorization else ""
    records = (
        db.query(Prediction)
        .filter(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc())
        .limit(50)
        .all()
    )
    return {
        "predictions": [
            {
                "id": r.id,
                "image_url": f"/api/uploads/{r.image_filename}?token={token}",
                "predicted_breed": r.predicted_breed,
                "confidence": r.confidence,
                "low_confidence": r.low_confidence,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]
    }


@app.get("/api/uploads/{filename}")
def get_uploaded_image(
    filename: str,
    user: User = Depends(get_current_user_for_image),
    db: Session = Depends(get_db),
):
    """Serves a stored image, only to the user who uploaded it."""
    record = db.query(Prediction).filter(Prediction.image_filename == filename).first()
    if not record or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found.")
    path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(path)


# --- Chat ------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@app.post("/api/chat")
def chat(payload: ChatRequest):
    history_msgs = [{"role": m.role, "content": m.content} for m in payload.history]
    try:
        reply = ask_chatbot(payload.message, history_msgs)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"reply": reply}


# --- Serve the frontend ---------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
