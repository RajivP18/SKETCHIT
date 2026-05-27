# ✏️ SketchAI — Image to Sketch Converter

Transform any photo into stunning hand-drawn artwork using OpenCV and Streamlit.

## Features

- **5 Sketch Styles**: Pencil, Charcoal, Ink, Hatching, Watercolor
- **Adjustable Intensity**: Fine-tune the line strength (1–25)
- **Instant Download**: Save your sketch as PNG
- **Beautiful Dark UI**: Elegant art-inspired design

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## How It Works

| Style | Technique |
|-------|-----------|
| **Pencil** | Gaussian blur dodge on inverted grayscale |
| **Charcoal** | High-intensity dodge + histogram equalization |
| **Ink** | Canny edge detection, inverted |
| **Hatching** | Pencil base + horizontal erosion kernel |
| **Watercolor** | Bilateral filter smoothing + soft pencil |

## Project Structure

```
sketch_app/
├── app.py           # Main Streamlit application
├── style.css        # Custom dark UI styling
├── components.html  # Animated canvas background (JS)
└── requirements.txt # Python dependencies
```
