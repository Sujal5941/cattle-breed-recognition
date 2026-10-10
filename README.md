# CattleVision AI

Image-based cattle breed recognition — a FastAPI backend running your trained
Vision Transformer, plus a professional frontend to explore the project and
test the model.

## 1. Where to put your model files

Copy your two files into `backend/models/`:

```
cattlevision-ai/
└── backend/
    └── models/
        ├── vit_cattle_model.pth   ← your trained weights
        └── labels.json            ← your id2label file
```

That's it — the backend automatically loads whatever is in that folder on
startup. You don't need to rename anything as long as the filenames match
exactly (`vit_cattle_model.pth` and `labels.json`).

If you also have a `preprocessor_config.json` from the same training run,
drop that in `backend/models/` too — the backend will use it automatically
for exact preprocessing parity with training. If it's not there, the backend
downloads the default processor for `google/vit-base-patch16-224-in21k`
from Hugging Face the first time you start the server (needs internet just
that once).

## 2. Install Python dependencies

Requires Python 3.10+. From the `backend` folder:

```bash
cd cattlevision-ai/backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

This installs PyTorch on CPU by default. If you have an NVIDIA GPU and want
CUDA acceleration, install the matching CUDA build of torch from
https://pytorch.org/get-started/locally/ instead of the plain `torch` line
in requirements.txt, then run `pip install -r requirements.txt` again for
the rest.

## 3. Run the app

Still inside `backend/`, with your venv active:

```bash
uvicorn main:app --reload --port 8000
```

You should see a line like:

```
[model_loader] Loaded model on cpu with 17 classes.
```

That confirms your model and labels loaded correctly.

## 4. Open the website

Go to **http://localhost:8000** in your browser. The FastAPI server serves
the frontend directly, so this one command runs the whole app — no separate
frontend server needed.

## 5. Using it

- **Identify Breed** page → upload a cattle photo → **Analyze Image** → see
  the real prediction, confidence, and top-3 breeds from your model.
- **Breeds** page → pulls the breed list live from `labels.json` via
  `/api/breeds`.

## Troubleshooting

- **"Model weights not found"** — check the file is at exactly
  `backend/models/vit_cattle_model.pth`.
- **"labels.json not found"** — check the file is at exactly
  `backend/models/labels.json`.
- **Missing/unexpected keys warning on startup** — this means the checkpoint
  structure doesn't line up 1:1 with a plain `ViTForImageClassification`
  state_dict (e.g. it was saved with a custom wrapper class). The server
  still starts, but check the printed key names against your Colab training
  script to confirm nothing important failed to load.
- **Port already in use** — run with a different port, e.g.
  `uvicorn main:app --reload --port 8001`, and open that port instead.
- **CORS / fetch errors in the browser console** — make sure you're opening
  `http://localhost:8000` (the backend's own address), not a separate static
  file server, since `/api/...` calls are relative.

## Project structure

```
cattlevision-ai/
├── backend/
│   ├── main.py            FastAPI app: /api/breeds, /api/predict, serves frontend
│   ├── model_loader.py    Loads the ViT model, weights, and labels
│   ├── requirements.txt
│   └── models/            ← put your .pth and labels.json here
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/{api.js, app.js}
└── README.md
```
