"""
attention_viz.py — Attention rollout visualization for the ViT model.

Implements "Attention Rollout" (Abnar & Zuidema, 2020, "Quantifying
Attention Flow in Transformers"): traces how attention propagates through
every transformer layer, accounting for the residual connection around each
attention block, to show which image patches most influenced the [CLS]
token — the actual representation the classification head uses. This is a
genuine inspection of the trained model's attention weights, computed fresh
on every request, not a static or decorative visualization.
"""

import base64
import io

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# The site's own accent color (--hide, a warm ochre/hide-brown), used for the
# focus brackets and glow so the visualization matches the rest of the
# product instead of looking like a bolted-on diagnostic tool.
_ACCENT_COLOR = (230, 150, 60)


def compute_attention_rollout(attentions):
    """
    attentions: tuple of per-layer attention tensors from a transformers ViT
        model called with output_attentions=True, each shaped
        (batch=1, num_heads, seq_len, seq_len).
    Returns: 1D numpy array of length (seq_len - 1) — the rollout relevance
        of each patch token to the [CLS] token, after propagating through
        every layer.
    """
    seq_len = attentions[0].shape[-1]
    rollout = torch.eye(seq_len)
    identity = torch.eye(seq_len)

    for attn in attentions:
        # Fuse attention heads by averaging.
        attn_heads_avg = attn[0].mean(dim=0)  # (seq_len, seq_len)

        # Attention Rollout accounts for the residual connection wrapped
        # around each attention block: without this, the rollout would
        # ignore how much of each token's representation simply passes
        # through unchanged from the previous layer.
        a = attn_heads_avg + identity
        a = a / a.sum(dim=-1, keepdim=True)

        rollout = a @ rollout

    cls_to_patches = rollout[0, 1:]  # relevance of each patch to [CLS]
    return cls_to_patches.detach().numpy()


def _find_hotspot_boxes(norm_grid, image_size, grid_size, top_percentile=72, max_boxes=2, min_patches=2, pad_frac=0.05):
    """
    Groups the patch grid's highest-attention cells into connected clusters
    (simple flood fill — the grid is only ~14x14, no need for scipy), then
    returns pixel-space bounding boxes for the strongest 1-2 clusters.
    """
    threshold = np.percentile(norm_grid, top_percentile)
    mask = norm_grid >= threshold
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for i in range(grid_size):
        for j in range(grid_size):
            if not mask[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) >= min_patches:
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                weight = sum(norm_grid[y, x] for y, x in cells)
                components.append({"weight": weight, "y0": min(ys), "y1": max(ys), "x0": min(xs), "x1": max(xs)})

    components.sort(key=lambda c: c["weight"], reverse=True)
    components = components[:max_boxes]

    img_w, img_h = image_size
    patch_w, patch_h = img_w / grid_size, img_h / grid_size
    pad_x, pad_y = img_w * pad_frac, img_h * pad_frac

    boxes = []
    for c in components:
        x0 = max(0, c["x0"] * patch_w - pad_x)
        y0 = max(0, c["y0"] * patch_h - pad_y)
        x1 = min(img_w, (c["x1"] + 1) * patch_w + pad_x)
        y1 = min(img_h, (c["y1"] + 1) * patch_h + pad_y)
        boxes.append((x0, y0, x1, y1))
    return boxes


def _draw_focus_bracket(draw, box, color=_ACCENT_COLOR, bracket_frac=0.22, width=5):
    """Draws camera-autofocus-style corner brackets around a box, rather than a full rectangle."""
    x0, y0, x1, y1 = box
    bracket_len = min(x1 - x0, y1 - y0) * bracket_frac
    corners = [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]
    for cx, cy, dx, dy in corners:
        draw.line([(cx, cy), (cx + dx * bracket_len, cy)], fill=color, width=width)
        draw.line([(cx, cy), (cx, cy + dy * bracket_len)], fill=color, width=width)


def build_attention_overlay(original_image, attentions, grid_size):
    """
    original_image: PIL.Image (RGB), the exact image the user uploaded.
    attentions: the `attentions` tuple from the model's output.
    grid_size: patches per side (e.g. 14 for a 224px image / 16px patches).
    Returns: a base64-encoded PNG data URL. The background gets a light,
        subtle fade toward grayscale, and camera-autofocus-style corner
        brackets are drawn around the 1-2 regions the model attended to
        most — clean bounding shapes rather than a full-image heatmap wash.
    """
    relevance = compute_attention_rollout(attentions).reshape(grid_size, grid_size)
    rmin, rmax = float(relevance.min()), float(relevance.max())
    norm = (relevance - rmin) / (rmax - rmin + 1e-8)

    # Very light background fade for context — most of the visual weight
    # now comes from the bracket boxes below, not this.
    mask_img = Image.fromarray((norm * 255).astype(np.uint8)).resize(
        original_image.size, Image.BICUBIC
    )
    blur_radius = max(2, int(max(original_image.size) * 0.02))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask = np.asarray(mask_img).astype(np.float32) / 255.0

    base = np.asarray(original_image.convert("RGB")).astype(np.float32)
    gray = np.asarray(ImageOps.grayscale(original_image.convert("RGB")).convert("RGB")).astype(np.float32)
    muted = gray * 0.88

    spotlight = mask[..., None] ** 0.9
    result = muted * (1 - spotlight) + base * spotlight
    overlay = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    # Draw bracket boxes around the actual hotspot cluster(s).
    boxes = _find_hotspot_boxes(norm, original_image.size, grid_size)
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        _draw_focus_bracket(draw, box)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
