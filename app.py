import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Preprocessor",
    page_icon="🖼️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
    }
    .main { background-color: #0f1117; }

    .stButton>button {
        background: linear-gradient(135deg, #6c63ff, #3ecfcf);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        opacity: 0.88;
        transform: translateY(-2px);
    }

    .stat-card {
        background: #1a1d2e;
        border: 1px solid #2d3154;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        text-align: center;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #6c7293;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'Space Mono', monospace;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #3ecfcf;
        font-family: 'Space Mono', monospace;
    }

    .section-header {
        color: #6c63ff;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.8rem;
        font-family: 'Space Mono', monospace;
    }

    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #a0a3b1 !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stSidebar"] {
        background-color: #12141f;
        border-right: 1px solid #2d3154;
    }

    .upload-area {
        border: 2px dashed #3d4166;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: #1a1d2e;
        color: #6c7293;
    }

    .badge {
        display: inline-block;
        background: #6c63ff22;
        color: #6c63ff;
        border: 1px solid #6c63ff55;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-family: 'Space Mono', monospace;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def apply_preprocessing(img: Image.Image, opts: dict) -> Image.Image:
    """Apply all selected preprocessing steps and return the result."""
    result = img.copy()

    # 1. Resize
    if opts.get("resize"):
        w, h = opts["resize_w"], opts["resize_h"]
        result = result.resize((w, h), Image.LANCZOS)

    # 2. Color mode
    mode = opts.get("color_mode", "Original")
    if mode == "Grayscale":
        result = result.convert("L").convert("RGB")
    elif mode == "RGB":
        result = result.convert("RGB")
    elif mode == "RGBA":
        result = result.convert("RGBA")

    # 3. Rotate / Flip
    if opts.get("rotate", 0) != 0:
        result = result.rotate(opts["rotate"], expand=True)
    if opts.get("flip_h"):
        result = ImageOps.mirror(result)
    if opts.get("flip_v"):
        result = ImageOps.flip(result)

    # 4. Brightness / Contrast / Saturation / Sharpness
    result = ImageEnhance.Brightness(result).enhance(opts.get("brightness", 1.0))
    result = ImageEnhance.Contrast(result).enhance(opts.get("contrast", 1.0))
    result = ImageEnhance.Color(result).enhance(opts.get("saturation", 1.0))
    result = ImageEnhance.Sharpness(result).enhance(opts.get("sharpness", 1.0))

    # 5. Filters
    filt = opts.get("filter", "None")
    if filt == "Blur":
        result = result.filter(ImageFilter.GaussianBlur(radius=opts.get("blur_radius", 2)))
    elif filt == "Sharpen":
        result = result.filter(ImageFilter.SHARPEN)
    elif filt == "Edge Enhance":
        result = result.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif filt == "Emboss":
        result = result.filter(ImageFilter.EMBOSS)
    elif filt == "Contour":
        result = result.filter(ImageFilter.CONTOUR)
    elif filt == "Median":
        result = result.filter(ImageFilter.MedianFilter(size=opts.get("median_size", 3)))

    # 6. Normalization (numpy)
    if opts.get("normalize"):
        arr = np.array(result).astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        result = Image.fromarray(arr.astype(np.uint8))

    # 7. Histogram equalization
    if opts.get("hist_eq"):
        gray = np.array(result.convert("L"))
        eq = cv2.equalizeHist(gray)
        result = Image.fromarray(eq).convert("RGB")

    # 8. Canny edge detection
    if opts.get("canny"):
        gray = np.array(result.convert("L"))
        edges = cv2.Canny(gray, opts.get("canny_t1", 100), opts.get("canny_t2", 200))
        result = Image.fromarray(edges).convert("RGB")

    # 9. Thresholding
    thresh = opts.get("threshold", "None")
    if thresh != "None":
        gray = np.array(result.convert("L"))
        if thresh == "Binary":
            _, out = cv2.threshold(gray, opts.get("thresh_val", 127), 255, cv2.THRESH_BINARY)
        elif thresh == "Otsu":
            _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif thresh == "Adaptive":
            out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        result = Image.fromarray(out).convert("RGB")

    # 10. Crop
    if opts.get("crop"):
        W, H = result.size
        l = int(W * opts["crop_l"] / 100)
        t = int(H * opts["crop_t"] / 100)
        r = int(W * opts["crop_r"] / 100)
        b = int(H * opts["crop_b"] / 100)
        if r > l and b > t:
            result = result.crop((l, t, r, b))

    # 11. Padding
    if opts.get("pad"):
        result = ImageOps.expand(result, border=opts.get("pad_size", 10),
                                 fill=opts.get("pad_color", "black"))

    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🖼️ Image Preprocessor")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Supports JPG, PNG, BMP, TIFF, WEBP",
    )

    if uploaded:
        st.markdown("---")

        # ── Resize ────────────────────────────────────────
        st.markdown('<p class="section-header"> Resize</p>', unsafe_allow_html=True)
        do_resize = st.checkbox("Enable Resize")
        resize_w = st.number_input("Width (px)", 1, 4096, 224, disabled=not do_resize)
        resize_h = st.number_input("Height (px)", 1, 4096, 224, disabled=not do_resize)

        st.markdown("---")

        # ── Color ─────────────────────────────────────────
        st.markdown('<p class="section-header"> Color Mode</p>', unsafe_allow_html=True)
        color_mode = st.selectbox("Mode", ["Original", "Grayscale", "RGB", "RGBA"])

        st.markdown("---")

        # ── Transforms ────────────────────────────────────
        st.markdown('<p class="section-header"> Transforms</p>', unsafe_allow_html=True)
        rotate = st.slider("Rotate (°)", -180, 180, 0)
        flip_h = st.checkbox("Flip Horizontal")
        flip_v = st.checkbox("Flip Vertical")

        st.markdown("---")

        # ── Enhancements ──────────────────────────────────
        st.markdown('<p class="section-header"> Enhancements</p>', unsafe_allow_html=True)
        brightness = st.slider("Brightness", 0.1, 3.0, 1.0, 0.05)
        contrast   = st.slider("Contrast",   0.1, 3.0, 1.0, 0.05)
        saturation = st.slider("Saturation", 0.0, 3.0, 1.0, 0.05)
        sharpness  = st.slider("Sharpness",  0.0, 3.0, 1.0, 0.05)

        st.markdown("---")

        # ── Filters ───────────────────────────────────────
        st.markdown('<p class="section-header"> Filter</p>', unsafe_allow_html=True)
        filt = st.selectbox("Apply Filter",
                            ["None", "Blur", "Sharpen", "Edge Enhance",
                             "Emboss", "Contour", "Median"])
        blur_radius  = st.slider("Blur Radius",  1, 15, 2, disabled=(filt != "Blur"))
        median_size  = st.select_slider("Median Size", [3, 5, 7, 9], disabled=(filt != "Median"))

        st.markdown("---")

        # ── Advanced ──────────────────────────────────────
        st.markdown('<p class="section-header">Advanced</p>', unsafe_allow_html=True)
        normalize = st.checkbox("Normalize (min-max)")
        hist_eq   = st.checkbox("Histogram Equalization")

        do_canny  = st.checkbox("Canny Edge Detection")
        canny_t1  = st.slider("Canny Threshold 1", 0, 500, 100, disabled=not do_canny)
        canny_t2  = st.slider("Canny Threshold 2", 0, 500, 200, disabled=not do_canny)

        threshold = st.selectbox("Thresholding", ["None", "Binary", "Otsu", "Adaptive"])
        thresh_val = st.slider("Threshold Value", 0, 255, 127,
                               disabled=(threshold != "Binary"))

        st.markdown("---")

        # ── Crop ──────────────────────────────────────────
        st.markdown('<p class="section-header"> Crop (%)</p>', unsafe_allow_html=True)
        do_crop = st.checkbox("Enable Crop")
        crop_l  = st.slider("Left %",   0, 100,   0, disabled=not do_crop)
        crop_t  = st.slider("Top %",    0, 100,   0, disabled=not do_crop)
        crop_r  = st.slider("Right %",  0, 100, 100, disabled=not do_crop)
        crop_b  = st.slider("Bottom %", 0, 100, 100, disabled=not do_crop)

        st.markdown("---")

        # ── Padding ───────────────────────────────────────
        st.markdown('<p class="section-header">Padding</p>', unsafe_allow_html=True)
        do_pad   = st.checkbox("Add Padding")
        pad_size = st.number_input("Padding (px)", 0, 200, 10, disabled=not do_pad)
        pad_color = st.color_picker("Pad Color", "#000000", disabled=not do_pad)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# Image Preprocessor")
st.markdown("Upload an image, configure preprocessing steps in the sidebar, and download the result.")
st.markdown("---")

if not uploaded:
    st.markdown("""
    <div class="upload-area">
        <h3 style="color:#6c7293; font-family:'Space Mono',monospace;">← Upload an Image</h3>
        <p>Use the sidebar uploader to get started.<br>
        Supports JPG · PNG · BMP · TIFF · WEBP</p>
        <span class="badge">Resize</span>
        <span class="badge">Grayscale</span>
        <span class="badge">Rotate</span>
        <span class="badge">Brightness</span>
        <span class="badge">Blur</span>
        <span class="badge">Normalize</span>
        <span class="badge">Canny</span>
        <span class="badge">Threshold</span>
        <span class="badge">Crop</span>
        <span class="badge">Padding</span>
    </div>
    """, unsafe_allow_html=True)
else:
    original = Image.open(uploaded)

    # Build options dict
    opts = dict(
        resize=do_resize, resize_w=resize_w, resize_h=resize_h,
        color_mode=color_mode,
        rotate=rotate, flip_h=flip_h, flip_v=flip_v,
        brightness=brightness, contrast=contrast,
        saturation=saturation, sharpness=sharpness,
        filter=filt, blur_radius=blur_radius, median_size=median_size,
        normalize=normalize, hist_eq=hist_eq,
        canny=do_canny, canny_t1=canny_t1, canny_t2=canny_t2,
        threshold=threshold, thresh_val=thresh_val,
        crop=do_crop, crop_l=crop_l, crop_t=crop_t, crop_r=crop_r, crop_b=crop_b,
        pad=do_pad, pad_size=pad_size, pad_color=pad_color,
    )

    processed = apply_preprocessing(original, opts)

    # ── Image stats ──────────────────────────────────────────────────────────
    ow, oh = original.size
    pw, ph = processed.size
    orig_kb   = len(pil_to_bytes(original)) / 1024
    proc_kb   = len(pil_to_bytes(processed)) / 1024

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Original Size",  f"{ow}×{oh}"),
        (c2, "Output Size",    f"{pw}×{ph}"),
        (c3, "Original (KB)",  f"{orig_kb:.1f}"),
        (c4, "Output (KB)",    f"{proc_kb:.1f}"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Side-by-side preview ─────────────────────────────────────────────────
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("####  Original")
        st.image(original, use_container_width=True)

    with right:
        st.markdown("#### Processed")
        st.image(processed, use_container_width=True)

    # ── Histogram ────────────────────────────────────────────────────────────
    with st.expander(" Pixel Intensity Histogram"):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3), facecolor="#0f1117")
        for ax, img, title in [(axes[0], original, "Original"),
                               (axes[1], processed, "Processed")]:
            arr = np.array(img.convert("L"))
            ax.hist(arr.ravel(), bins=256, color="#6c63ff", alpha=0.85)
            ax.set_title(title, color="#c0c3d9", fontsize=10)
            ax.set_facecolor("#1a1d2e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3154")
            ax.tick_params(colors="#6c7293")

        st.pyplot(fig)
        plt.close(fig)

    # ── Numpy stats ───────────────────────────────────────────────────────────
    with st.expander("Pixel Statistics"):
        arr = np.array(processed.convert("L")).astype(np.float32)
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Min",  f"{arr.min():.1f}")
        s2.metric("Max",  f"{arr.max():.1f}")
        s3.metric("Mean", f"{arr.mean():.2f}")
        s4.metric("Std",  f"{arr.std():.2f}")
        s5.metric("Mode", f"{float(np.bincount(arr.ravel().astype(int)).argmax()):.0f}")

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    dl_fmt = st.radio("Download Format", ["PNG", "JPEG", "WEBP"], horizontal=True)
    mime_map = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
    dl_bytes = pil_to_bytes(processed.convert("RGB"), dl_fmt)

    st.download_button(
        label=f"⬇️ Download Processed Image ({dl_fmt})",
        data=dl_bytes,
        file_name=f"preprocessed.{dl_fmt.lower()}",
        mime=mime_map[dl_fmt],
        use_container_width=True,
    )