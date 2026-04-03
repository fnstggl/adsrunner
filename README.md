# Green Screen Phone Compositor

A localhost web app that takes a greenscreen phone scene and a UI screenshot,
uses Claude Vision to detect the phone screen corners, and composites the UI
into the screen via perspective warp.

## Setup

```bash
# 1. Clone / navigate into the project
cd adsrunner

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 5. Run the server
python app.py
```

Then open **http://localhost:5000** in your browser.

## Usage

1. **Drag & drop** (or click) the *Greenscreen Scene* zone — upload your phone photo.
2. **Drag & drop** the *UI Screenshot* zone — upload the mobile screen you want composited in.
3. Click **Composite**.
4. The app calls Claude Vision to detect the four phone screen corners (~2–5 s).
5. The composited image appears — click **Download PNG** to save it.

> **Corner caching:** detected corners are cached by scene image hash in
> `/tmp/corners_cache/`. Uploading the same scene a second time skips the
> Vision API call entirely.

## File structure

```
adsrunner/
├── app.py                    # Flask server
├── compositor_v4_final.py    # Perspective-warp compositor
├── vision_corner_detector.py # Claude Vision corner detector
├── templates/
│   └── index.html            # Single-page frontend
├── requirements.txt
├── .env.example
└── .gitignore
```

## Limits

| Setting | Value |
|---------|-------|
| Max upload size | 20 MB per file |
| Supported formats | JPG, PNG, WEBP |
| Corner cache location | `/tmp/corners_cache/` |

## Troubleshooting

- **"Vision API error"** — check that `ANTHROPIC_API_KEY` is set correctly in `.env`.
- **Blurry composite** — ensure the UI screenshot matches the phone screen aspect ratio.
- **Wrong corners detected** — delete the cached JSON for that image from `/tmp/corners_cache/` to force re-detection.
