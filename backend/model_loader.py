"""
model_loader.py

Loads the trained CattleVision ViT model, its label mapping, and the image
processor used for inference. This module is the ONLY place that touches the
model weights - everything else in the backend calls into it.

Expected files (see README.md for exact placement):
    backend/models/vit_cattle_model.pth   -> trained weights (state_dict)
    backend/models/labels.json            -> id2label mapping for the 17 breeds

Optional file:
    backend/models/preprocessor_config.json
        If present, this is used to build the image processor so that
        inference preprocessing exactly matches training. If absent, the
        processor is downloaded from the base checkpoint
        "google/vit-base-patch16-224-in21k" the first time the server starts
        (requires internet access on that first run; it is then cached
        locally by Hugging Face and no longer needs the network).
"""

import json
import os
import re
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

BASE_CHECKPOINT = "google/vit-base-patch16-224-in21k"

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "vit_cattle_model.pth")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
PROCESSOR_CONFIG_PATH = os.path.join(MODELS_DIR, "preprocessor_config.json")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_processor = None
_id2label = None


def _load_labels():
    """
    Reads backend/models/labels.json and normalizes it into an
    {int: str} id2label dict, regardless of which common shape it was
    exported in:
        {"0": "Amritmahal", "1": "Ayrshire", ...}
        {"id2label": {"0": "Amritmahal", ...}}
        ["Amritmahal", "Ayrshire", ...]
    """
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"labels.json not found at {LABELS_PATH}. "
            "Copy your labels file there before starting the server."
        )

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return {i: name for i, name in enumerate(raw)}

    if isinstance(raw, dict) and "id2label" in raw:
        raw = raw["id2label"]

    # dict with string or int keys -> normalize to int keys
    return {int(k): v for k, v in raw.items()}


_LEGACY_KEY_PATTERNS = [
    (r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.query\.(weight|bias)$",
     r"vit.layers.\1.attention.q_proj.\2"),
    (r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.key\.(weight|bias)$",
     r"vit.layers.\1.attention.k_proj.\2"),
    (r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.value\.(weight|bias)$",
     r"vit.layers.\1.attention.v_proj.\2"),
    (r"^vit\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)$",
     r"vit.layers.\1.attention.o_proj.\2"),
    (r"^vit\.encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)$",
     r"vit.layers.\1.mlp.fc1.\2"),
    (r"^vit\.encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)$",
     r"vit.layers.\1.mlp.fc2.\2"),
    (r"^vit\.encoder\.layer\.(\d+)\.layernorm_before\.(weight|bias)$",
     r"vit.layers.\1.layernorm_before.\2"),
    (r"^vit\.encoder\.layer\.(\d+)\.layernorm_after\.(weight|bias)$",
     r"vit.layers.\1.layernorm_after.\2"),
]


def _remap_legacy_vit_keys(state_dict, model_keys):
    """
    Some checkpoints were saved against an older `transformers` release that
    named ViT encoder submodules differently (e.g. `encoder.layer.0.attention
    .attention.query` and separate q/k/v Linear layers) than newer releases
    (e.g. `layers.0.attention.q_proj`, fused-style naming). If the checkpoint
    already matches the installed model's key names, this is a no-op.
    Otherwise it rewrites the checkpoint's keys to match, so training-time
    weights still land in the right tensors regardless of which
    `transformers` version is installed at inference time.
    """
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
            "[model_loader] Warning - key remap left some keys unresolved: "
            f"{sorted(still_missing)[:10]}"
        )
    else:
        print(
            "[model_loader] Remapped checkpoint key names to match the "
            "installed transformers version's ViT layer naming."
        )
    return remapped


def load_model():
    """
    Loads the model + processor once and caches them in module-level
    globals. Safe to call multiple times (subsequent calls are no-ops).
    """
    global _model, _processor, _id2label

    if _model is not None:
        return _model, _processor, _id2label

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH}. "
            "Copy vit_cattle_model.pth there before starting the server."
        )

    _id2label = _load_labels()
    label2id = {v: k for k, v in _id2label.items()}
    num_labels = len(_id2label)

    # Build the ViT architecture matching the fine-tuning setup, then load
    # the trained weights into it.
    model = ViTForImageClassification.from_pretrained(
        BASE_CHECKPOINT,
        num_labels=num_labels,
        id2label={str(k): v for k, v in _id2label.items()},
        label2id={k: str(v) for k, v in label2id.items()},
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )

    state_dict = torch.load(WEIGHTS_PATH, map_location=_device)
    # Some training scripts save {"model_state_dict": ...} or wrap in "state_dict"
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    state_dict = _remap_legacy_vit_keys(state_dict, set(model.state_dict().keys()))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[model_loader] Warning - missing keys when loading weights: {missing}")
    if unexpected:
        print(f"[model_loader] Warning - unexpected keys when loading weights: {unexpected}")

    model.to(_device)
    model.eval()

    if os.path.exists(PROCESSOR_CONFIG_PATH):
        processor = ViTImageProcessor.from_pretrained(MODELS_DIR)
    else:
        processor = ViTImageProcessor.from_pretrained(BASE_CHECKPOINT)

    _model, _processor = model, processor
    print(f"[model_loader] Loaded model on {_device} with {num_labels} classes.")
    return _model, _processor, _id2label


def get_device():
    return _device
