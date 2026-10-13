
"""
model_loader.py

Loads the CattleVision ViT model, label mapping, and image processor.

The trained model can be loaded from:
1. A local file: backend/models/vit_cattle_model.pth
2. A private Hugging Face repository, configured through:
   HF_MODEL_REPO=Sujal5941/cattlevision-model
   HF_TOKEN=<your read-only Hugging Face token>

The model is downloaded only when it is missing locally.
"""

import json
import os
import re

import torch
from huggingface_hub import hf_hub_download
from transformers import ViTForImageClassification, ViTImageProcessor

BASE_CHECKPOINT = "google/vit-base-patch16-224-in21k"

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "vit_cattle_model.pth")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
PROCESSOR_CONFIG_PATH = os.path.join(
    MODELS_DIR, "preprocessor_config.json"
)

HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "Sujal5941/cattlevision-model",
)
HF_TOKEN = os.getenv("HF_TOKEN")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_processor = None
_id2label = None


def _load_labels():
    """Load and normalize the 17-class label mapping."""
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"labels.json not found at {LABELS_PATH}."
        )

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, list):
        return {i: name for i, name in enumerate(raw)}

    if isinstance(raw, dict) and "id2label" in raw:
        raw = raw["id2label"]

    if not isinstance(raw, dict):
        raise ValueError("Unsupported labels.json format.")

    return {int(key): value for key, value in raw.items()}


_LEGACY_KEY_PATTERNS = [
    (
        r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.query\.(weight|bias)$",
        r"vit.layers.\1.attention.q_proj.\2",
    ),
    (
        r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.key\.(weight|bias)$",
        r"vit.layers.\1.attention.k_proj.\2",
    ),
    (
        r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.value\.(weight|bias)$",
        r"vit.layers.\1.attention.v_proj.\2",
    ),
    (
        r"^vit\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)$",
        r"vit.layers.\1.attention.o_proj.\2",
    ),
    (
        r"^vit\.encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)$",
        r"vit.layers.\1.mlp.fc1.\2",
    ),
    (
        r"^vit\.encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)$",
        r"vit.layers.\1.mlp.fc2.\2",
    ),
    (
        r"^vit\.encoder\.layer\.(\d+)\.layernorm_before\.(weight|bias)$",
        r"vit.layers.\1.layernorm_before.\2",
    ),
    (
        r"^vit\.encoder\.layer\.(\d+)\.layernorm_after\.(weight|bias)$",
        r"vit.layers.\1.layernorm_after.\2",
    ),
]


def _remap_legacy_vit_keys(state_dict, model_keys):
    """Remap legacy checkpoint parameter names when necessary."""
    ckpt_keys = set(state_dict.keys())

    if ckpt_keys <= model_keys:
        return state_dict

    remapped = {}

    for key, tensor in state_dict.items():
        new_key = key

        for pattern, replacement in _LEGACY_KEY_PATTERNS:
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                break

        remapped[new_key] = tensor

    still_missing = model_keys - set(remapped.keys())

    if still_missing:
        print(
            "[model_loader] Warning: unresolved model keys: "
            f"{sorted(still_missing)[:10]}"
        )
    else:
        print("[model_loader] Remapped legacy ViT checkpoint keys.")

    return remapped


def _get_weights_path():
    """
    Use the local model when available; otherwise download it
    from the configured Hugging Face repository.
    """
    if os.path.isfile(WEIGHTS_PATH):
        print("[model_loader] Using local model weights.")
        return WEIGHTS_PATH

    if not HF_TOKEN:
        raise RuntimeError(
            "Model weights are not available locally. "
            "Configure HF_TOKEN in the backend environment."
        )

    if not HF_MODEL_REPO:
        raise RuntimeError(
            "Configure HF_MODEL_REPO with your Hugging Face "
            "repository ID."
        )

    print(f"[model_loader] Downloading model from {HF_MODEL_REPO}...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="vit_cattle_model.pth",
            repo_type="model",
            token=HF_TOKEN,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download the private Hugging Face model. "
            "Check HF_MODEL_REPO, HF_TOKEN, repository permissions, "
            "and available disk space."
        ) from exc

    print("[model_loader] Model download completed.")
    return downloaded_path


def load_model():
    """Load and cache the trained model, processor, and labels."""
    global _model, _processor, _id2label

    if _model is not None:
        return _model, _processor, _id2label

    _id2label = _load_labels()
    label2id = {name: index for index, name in _id2label.items()}
    num_labels = len(_id2label)

    weights_path = _get_weights_path()

    model = ViTForImageClassification.from_pretrained(
        BASE_CHECKPOINT,
        num_labels=num_labels,
        id2label={str(k): v for k, v in _id2label.items()},
        label2id={k: str(v) for k, v in label2id.items()},
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )

    state_dict = torch.load(
        weights_path,
        map_location=_device,
        weights_only=True,
    )

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    state_dict = _remap_legacy_vit_keys(
        state_dict,
        set(model.state_dict().keys()),
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print(
            f"[model_loader] Warning: {len(missing)} model keys missing: "
            f"{missing[:10]}"
        )

    if unexpected:
        print(
            f"[model_loader] Warning: {len(unexpected)} unexpected keys: "
            f"{unexpected[:10]}"
        )

    model.to(_device)
    model.eval()

    if os.path.isfile(PROCESSOR_CONFIG_PATH):
        processor = ViTImageProcessor.from_pretrained(MODELS_DIR)
    else:
        processor = ViTImageProcessor.from_pretrained(BASE_CHECKPOINT)

    _model = model
    _processor = processor

    print(
        f"[model_loader] Loaded model on {_device} "
        f"with {num_labels} classes."
    )

    return _model, _processor, _id2label


def get_device():
    return _device
