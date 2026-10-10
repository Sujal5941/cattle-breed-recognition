"""
diagnose_model.py — run this from backend/ to check whether the trained
weights are actually landing in the right places.

Usage:
    python diagnose_model.py
"""

import json
import os
import torch
from transformers import ViTForImageClassification

from model_loader import _remap_legacy_vit_keys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "vit_cattle_model.pth")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
BASE_CHECKPOINT = "google/vit-base-patch16-224-in21k"

print(f"Loading labels from {LABELS_PATH}")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    raw_labels = json.load(f)
if isinstance(raw_labels, list):
    id2label = {i: n for i, n in enumerate(raw_labels)}
elif "id2label" in raw_labels:
    id2label = {int(k): v for k, v in raw_labels["id2label"].items()}
else:
    id2label = {int(k): v for k, v in raw_labels.items()}
num_labels = len(id2label)
print(f"-> {num_labels} classes: {list(id2label.values())}\n")

print(f"Loading checkpoint from {WEIGHTS_PATH}")
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")

print(f"Checkpoint top-level type: {type(checkpoint)}")
if isinstance(checkpoint, dict):
    print(f"Checkpoint top-level keys (first 15): {list(checkpoint.keys())[:15]}")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        print("-> unwrapped 'state_dict'")
    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
        print("-> unwrapped 'model_state_dict'")
else:
    print("!! Checkpoint is not a dict — it looks like a full pickled model object,")
    print("   not a state_dict. If so, this is very likely the bug: main.py expects")
    print("   torch.load() to return a state_dict, but it's returning a full model.")

print()
if isinstance(checkpoint, dict):
    ckpt_keys = list(checkpoint.keys())
    print(f"Total keys in checkpoint state_dict: {len(ckpt_keys)}")
    print("First 10 keys:", ckpt_keys[:10])
    print("Last 10 keys:", ckpt_keys[-10:])
    classifier_keys = [k for k in ckpt_keys if "classifier" in k.lower()]
    print(f"\nKeys containing 'classifier': {classifier_keys}")
    for k in classifier_keys:
        print(f"  {k}: shape {tuple(checkpoint[k].shape)}")

print("\nBuilding fresh ViTForImageClassification for comparison...")
model = ViTForImageClassification.from_pretrained(
    BASE_CHECKPOINT,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
)
model_keys = set(model.state_dict().keys())
print(f"Fresh model expects {len(model_keys)} keys, e.g. classifier keys:")
for k in model_keys:
    if "classifier" in k.lower():
        print(f"  {k}: shape {tuple(model.state_dict()[k].shape)}")

if isinstance(checkpoint, dict):
    ckpt_key_set = set(checkpoint.keys())
    missing = model_keys - ckpt_key_set
    unexpected = ckpt_key_set - model_keys
    print(f"\nBEFORE REMAP")
    print(f"Keys the model needs but checkpoint doesn't have ({len(missing)}):")
    for k in sorted(missing)[:10]:
        print(f"  MISSING: {k}")
    print(f"Keys in checkpoint the model doesn't recognize ({len(unexpected)}):")
    for k in sorted(unexpected)[:10]:
        print(f"  UNEXPECTED: {k}")

    print("\nApplying legacy key remap (same logic model_loader.py uses)...")
    remapped = _remap_legacy_vit_keys(checkpoint, model_keys)
    remapped_key_set = set(remapped.keys())
    missing_after = model_keys - remapped_key_set
    unexpected_after = remapped_key_set - model_keys

    print(f"\nAFTER REMAP")
    print(f"Keys the model needs but checkpoint doesn't have ({len(missing_after)}):")
    for k in sorted(missing_after)[:10]:
        print(f"  STILL MISSING: {k}")
    print(f"Keys in checkpoint the model doesn't recognize ({len(unexpected_after)}):")
    for k in sorted(unexpected_after)[:10]:
        print(f"  STILL UNEXPECTED: {k}")

    if not missing_after and not unexpected_after:
        print("\n*** All keys match after remap. model_loader.py will load your")
        print("*** trained weights correctly. Start the server and re-test.")
    elif any("classifier" in k for k in missing_after):
        print("\n*** Your classifier head is still not matching after remap.")
        print("*** Check 'STILL UNEXPECTED' above for what it's actually called")
        print("*** and tell me so I can extend the remap.")
    else:
        print("\n*** Remap left some keys unresolved — paste this full output")
        print("*** and I'll extend the pattern list in model_loader.py.")
