import streamlit as st
import numpy as np
from PIL import Image
import cv2
import base64
import io
import os

# Page config
st.set_page_config(
    page_title="SketchIT",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom HTML/JS
with open("components.html") as f:
    st.markdown(f.read(), unsafe_allow_html=True)


def odd(v: int) -> int:
    """Ensure value is odd and >= 1."""
    v = max(1, int(v))
    return v if v % 2 == 1 else v + 1


def pencil_sketch_gray(gray: np.ndarray, blur_k: int) -> np.ndarray:
    """Core dodge-blend pencil effect on a grayscale image."""
    inv = 255 - gray
    blurred = cv2.GaussianBlur(inv, (blur_k, blur_k), 0)
    # Avoid divide-by-zero
    denom = (255 - blurred).astype(np.float32)
    denom[denom == 0] = 1
    sketch = np.clip(gray.astype(np.float32) / denom * 255.0, 0, 255).astype(np.uint8)
    return sketch


def to_sketch(image: Image.Image, style: str, intensity: int, color: bool = False) -> Image.Image:
    """Convert PIL Image to sketch using OpenCV."""
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_k = odd(intensity * 2)

    if style == "Pencil":
        sketch_gray = pencil_sketch_gray(gray, blur_k)
        if color:
            # Blend pencil edges with a desaturated colour version
            color_layer = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            color_layer = cv2.cvtColor(color_layer, cv2.COLOR_BGR2RGB)
            sketch_3ch = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2RGB)
            result = cv2.addWeighted(sketch_3ch, 0.6, color_layer, 0.4, 0)
            return Image.fromarray(result)
        return Image.fromarray(sketch_gray)

    elif style == "Charcoal":
        big_k = odd(intensity * 4)
        sketch_gray = pencil_sketch_gray(gray, big_k)
        sketch_gray = cv2.equalizeHist(sketch_gray)
        # Darken & add grain
        sketch_gray = np.clip(sketch_gray.astype(np.int32) - 20, 0, 255).astype(np.uint8)
        if color:
            color_layer = img_array.copy()
            sketch_3ch = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2RGB)
            result = cv2.addWeighted(sketch_3ch, 0.55, color_layer, 0.45, 0)
            return Image.fromarray(result)
        return Image.fromarray(sketch_gray)

    elif style == "Ink":
        # Adaptive thresholding gives much cleaner ink lines than Canny
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        block = odd(intensity + 10)  # block size for adaptive threshold
        ink = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block, intensity // 2 + 2
        )
        if color:
            # Tint the white areas with the original colour
            ink_3ch = cv2.cvtColor(ink, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            color_norm = img_array.astype(np.float32) / 255.0
            # Where ink is white (1.0), blend in colour; black lines stay black
            result = (ink_3ch * color_norm * 255).astype(np.uint8)
            return Image.fromarray(result)
        return Image.fromarray(ink)

    elif style == "Hatching":
        # Use pencil base, then draw diagonal hatch lines modulated by darkness
        sketch_gray = pencil_sketch_gray(gray, blur_k)
        h, w = sketch_gray.shape
        hatch = sketch_gray.copy()
        # Draw diagonal lines every `spacing` pixels; suppress where image is bright
        spacing = max(3, 28 - intensity)
        for i in range(-h, w, spacing):
            cv2.line(hatch, (i, 0), (i + h, h), 180, 1)
        # Cross-hatch for dark regions
        dark_mask = sketch_gray < 120
        for i in range(-h, w, spacing * 2):
            line_img = np.zeros_like(hatch)
            cv2.line(line_img, (i + h, 0), (i, h), 150, 1)
            hatch[dark_mask & (line_img > 0)] = 150
        hatch = np.clip(hatch, 0, 255).astype(np.uint8)
        if color:
            color_layer = img_array.copy()
            hatch_3ch = cv2.cvtColor(hatch, cv2.COLOR_GRAY2RGB)
            result = cv2.addWeighted(hatch_3ch, 0.5, color_layer, 0.5, 0)
            return Image.fromarray(result)
        return Image.fromarray(hatch)

    elif style == "Watercolor":
        # Multiple bilateral passes = painterly smooth look
        smooth = img_array.copy()
        passes = max(1, intensity // 5 + 1)
        for _ in range(passes):
            smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
        # Edge mask from original
        edges = cv2.Canny(gray, 50, 150)
        edges_inv = cv2.bitwise_not(edges)
        edges_3ch = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)
        # Blend smooth colour + edge lines
        result = cv2.bitwise_and(smooth, edges_3ch)
        # Boost saturation for watercolor pop
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if not color:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            return Image.fromarray(result)
        return Image.fromarray(result)

    else:
        return Image.fromarray(gray)


def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
  <div class="header-inner">
    <div class="logo-mark">✏</div>
    <div>
      <h1 class="app-title">SketchIT</h1>
      <p class="app-sub">Transform any photo into stunning art</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ─────────────────────────────────────────────────────────────────
st.markdown('<div class="controls-bar">', unsafe_allow_html=True)
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 1, 1])

with ctrl1:
    style = st.selectbox(
        "🎨 Sketch Style",
        ["Pencil", "Charcoal", "Ink", "Hatching", "Watercolor"],
        key="style"
    )

with ctrl2:
    intensity = st.slider(
        "✏️ Line Intensity",
        min_value=1, max_value=25, value=13,
        key="intensity"
    )

with ctrl3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    color_mode = st.toggle("🎨 Color", value=False, key="color_mode",
                           help="Blend original colors into the sketch")

with ctrl4:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    convert_btn = st.button("🖊 Sketch It!", use_container_width=True, key="convert")

st.markdown('</div>', unsafe_allow_html=True)

# ── Upload + Preview ──────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="panel-label">📸 Original Photo</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        label_visibility="collapsed",
        key="upload"
    )

    if uploaded:
        img = Image.open(uploaded)
        b64_orig = pil_to_b64(img)
        st.markdown(f"""
        <div class="img-frame">
          <img src="data:image/png;base64,{b64_orig}" class="preview-img"/>
          <div class="img-badge">Original</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="img-meta">{img.width} × {img.height} px &nbsp;|&nbsp; {uploaded.type}</p>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="drop-zone" id="dropZone">
          <div class="drop-icon">🖼</div>
          <p class="drop-title">Drop your image here</p>
          <p class="drop-sub">PNG · JPG · WEBP · BMP</p>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel-label">✏️ Sketch Output</div>', unsafe_allow_html=True)

    if uploaded and convert_btn:
        with st.spinner(""):
            st.markdown('<div class="processing-overlay"><span class="proc-text">Sketching…</span></div>', unsafe_allow_html=True)
            sketch_img = to_sketch(img, style, intensity, color=color_mode)

        b64_sketch = pil_to_b64(sketch_img)
        sketch_bytes = pil_to_bytes(sketch_img)

        st.markdown(f"""
        <div class="img-frame sketch-frame">
          <img src="data:image/png;base64,{b64_sketch}" class="preview-img sketch-reveal"/>
          <div class="img-badge sketch-badge">{style}</div>
        </div>
        """, unsafe_allow_html=True)

        fname = f"sketch_{style.lower()}_{os.path.splitext(uploaded.name)[0]}.png"
        st.download_button(
            label="⬇️  Download Sketch",
            data=sketch_bytes,
            file_name=fname,
            mime="image/png",
            use_container_width=True,
            key="dl"
        )

    elif uploaded and not convert_btn:
        st.markdown("""
        <div class="drop-zone waiting-zone">
          <div class="drop-icon anim-pencil">✏️</div>
          <p class="drop-title">Ready to sketch</p>
          <p class="drop-sub">Hit <strong>Sketch It!</strong> to transform</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="drop-zone empty-zone">
          <div class="drop-icon">🎨</div>
          <p class="drop-title">No image yet</p>
          <p class="drop-sub">Upload a photo on the left first</p>
        </div>
        """, unsafe_allow_html=True)

# ── Style Guide Cards ─────────────────────────────────────────────────────────
st.markdown("""
<div class="style-section">
  <h3 class="section-title">Style Guide</h3>
  <div class="style-cards">
    <div class="style-card">
      <div class="card-icon">✏️</div>
      <div class="card-name">Pencil</div>
      <div class="card-desc">Classic graphite look with soft tonal gradients</div>
    </div>
    <div class="style-card">
      <div class="card-icon">🖤</div>
      <div class="card-name">Charcoal</div>
      <div class="card-desc">Bold, dramatic strokes with high contrast</div>
    </div>
    <div class="style-card">
      <div class="card-icon">🖊</div>
      <div class="card-name">Ink</div>
      <div class="card-desc">Sharp edge-detected lines, comic-book style</div>
    </div>
    <div class="style-card">
      <div class="card-icon">📐</div>
      <div class="card-name">Hatching</div>
      <div class="card-desc">Fine cross-hatch lines for etching depth</div>
    </div>
    <div class="style-card">
      <div class="card-icon">💧</div>
      <div class="card-name">Watercolor</div>
      <div class="card-desc">Smooth, soft-washed tones with light strokes</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
  Designed By Rajiv Pillalamarri
</div>
""", unsafe_allow_html=True)