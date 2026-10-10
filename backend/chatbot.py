"""
chatbot.py — CattleVision AI project assistant, powered by a local Ollama model.

Requires Ollama running locally (ollama.com), with a model already pulled,
e.g.:
    ollama pull qwen2.5:7b

Configure which model/host to use with environment variables:
    OLLAMA_BASE_URL  (default: http://localhost:11434)
    OLLAMA_MODEL     (default: qwen2.5:7b)

Example (Windows PowerShell), before starting uvicorn:
    $env:OLLAMA_MODEL = "qwen2.5:7b"
"""

import os
import requests

from model_loader import load_model

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
REQUEST_TIMEOUT_SECONDS = 60

PROJECT_FACTS = """
Project name: CattleVision AI — an Image-Based Cattle Breed Recognition System.
Purpose: a user uploads a cattle photo and the site predicts the most likely
breed using a trained AI model, showing a confidence score and the top 3
possible breeds.

Model architecture: Vision Transformer (ViT-Base), pretrained checkpoint
google/vit-base-patch16-224-in21k (pretrained on ImageNet-21k), fine-tuned
for this project's 17-class cattle breed classification task.
Input resolution: 224x224 pixels, split into 16x16 patches.
Training data: 3,400 cattle images across 17 breed classes.
Training config: AdamW optimizer, learning rate 5e-5, Cross-Entropy Loss,
trained on CUDA GPU when available (otherwise CPU).

The 17 breeds this model was trained to recognize: {breed_list}

How the site works: Home introduces the project, About explains its goals
and limitations, How It Works shows the 4-step pipeline (upload, preprocess,
AI analysis, prediction), Breeds lists all 17 supported breeds, AI Model
explains the Vision Transformer architecture in detail, and Identify Breed
is where a user actually uploads a photo and gets a real prediction from the
model.

Important behavior to explain if asked: if the top prediction's confidence
is below 50%, the site tells the user the image likely doesn't clearly match
any of the 17 trained breeds, rather than presenting a low-confidence guess
as a real result.

Known limitations (be upfront about these if relevant): the model only
recognizes these 17 breeds and nothing else; prediction quality depends on
image quality, lighting, and camera angle; visually similar breeds can be
confused with each other; this is an AI-assisted tool, not a substitute for
expert identification; and confidence scores are statistical, not a
guarantee.
"""

SYSTEM_PROMPT_TEMPLATE = """You are the CattleVision AI project assistant, embedded in the CattleVision AI website.

Answer questions about this specific project only: what it does, how the Vision Transformer model works, which breeds it supports, how to use the site, confidence scores, and its limitations. Use the project facts below as your source of truth — do not invent breed origins, characteristics, accuracy numbers, or any detail not given to you. If someone asks for a specific breed fact you don't have (like a breed's origin or detailed characteristics), say plainly that this detail hasn't been added to the project yet, instead of making one up.

If someone asks something unrelated to this project, answer briefly and helpfully if it's simple and easy, but gently steer back to what you can actually help with here.

Keep answers conversational and concise — a few sentences unless the user asks for more detail.

PROJECT FACTS:
{project_facts}
"""


def _build_system_prompt():
    try:
        _, _, id2label = load_model()
        breed_list = ", ".join(
            name for _, name in sorted(id2label.items(), key=lambda kv: kv[0])
        )
    except Exception:
        breed_list = "(breed list unavailable — model not loaded)"

    facts = PROJECT_FACTS.format(breed_list=breed_list)
    return SYSTEM_PROMPT_TEMPLATE.format(project_facts=facts)


def ask_chatbot(user_message, history):
    """
    history: list of {"role": "user"|"assistant", "content": str}, oldest first.
    Returns the assistant's reply text.
    Raises RuntimeError with a user-friendly message on failure.
    """
    messages = [{"role": "system", "content": _build_system_prompt()}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Can't reach Ollama. Make sure it's running on this machine "
            "(open the Ollama app, or run `ollama serve`) and that the "
            f"'{OLLAMA_MODEL}' model has been pulled (`ollama pull {OLLAMA_MODEL}`)."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("The assistant took too long to respond. Please try again.")

    if response.status_code == 404:
        raise RuntimeError(
            f"Ollama doesn't have the model '{OLLAMA_MODEL}' pulled. Run "
            f"`ollama pull {OLLAMA_MODEL}` and try again, or set OLLAMA_MODEL "
            "to a model you already have."
        )
    if not response.ok:
        raise RuntimeError("Unable to reach the chatbot right now. Please try again.")

    data = response.json()
    reply = data.get("message", {}).get("content", "").strip()
    if not reply:
        raise RuntimeError("The assistant didn't return a response. Please try again.")
    return reply
