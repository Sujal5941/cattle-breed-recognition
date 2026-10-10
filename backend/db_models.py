"""
db_models.py — SQLAlchemy ORM models.

User          — a registered account (username + salted/hashed password).
Prediction    — one uploaded image + its model result. Saved for every
                 prediction, whether or not the person was logged in
                 (user_id is nullable); this is the "image database" that
                 accumulates real usage data over time, and logged-in users
                 can browse their own history from it.
"""

import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    password_salt = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    image_filename = Column(String(255), nullable=False)
    predicted_breed = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    top_predictions_json = Column(Text, nullable=False)
    low_confidence = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    user = relationship("User", back_populates="predictions")
