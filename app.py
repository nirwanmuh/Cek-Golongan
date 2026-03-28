import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import base64
import io

from golongan_rules import GOLONGAN_RULES

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Deteksi Golongan Kendaraan I–XII",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS (FULLSCREEN & RESPONSIVE CAMERA)
# =====================================================
st.markdown("""
<style>
.block-container {
    padding-left: 0.8rem;
    padding-right: 0.8rem;
}

div[data-testid="stCameraInput"],
div[data-testid="stFileUploader"] {
    width: 100% !important;
}

div[data-testid="stCameraInput"] video {
    width: 100% !important;
    border-radius: 12px;
}

@media (max-width: 768px) {
    div[data-testid="stCameraInput"] video {
        height: 70vh !important;
        object-fit: cover;
    }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.title("🚗 Deteksi Golongan Kendaraan I–XII")
st.caption(
    "Sistem mendeteksi kendaraan dari kamera atau upload foto, "
    "menentukan golongan berdasarkan PM 66/2019, dan menampilkan crop tiap kendaraan."
)

# =====================================================
# LOAD YOLO
# =====================================================
model = YOLO("models/yolov8n.pt")

CLASS_COLORS = {
    "car": "red",
    "motorbike": "blue",
    "bus": "green",
    "truck": "yellow",
    "bicycle": "orange",
}

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def estimate_size(box, frame_width):
    x1, y1, x2, y2 = box
    rel = (x2 - x1) / frame_width
    if rel < 0.25:
        return "small"
    elif rel < 0.40:
        return "medium"
    elif rel < 0.65:
        return "large"
    elif rel < 0.85:
        return "xlarge"
    return "xxlarge"

def classify_vehicle(label, size):
    for rule in GOLONGAN_RULES:
        if label in rule["yolo"]:
            if rule["size"] == "any" or rule["size"] == size:
                return rule["golongan"]
    return "Tidak diketahui"

# =====================================================
# INPUT SECTION (KAMERA → UPLOAD DI BAWAH)
# =====================================================
st.subheader("📸 Ambil foto kendaraan")
camera_img = st.camera_input("")

st.subheader("📤 Atau upload foto kendaraan")
uploaded_img = st.file_uploader("", type=["jpg", "jpeg", "png"])

image_source = camera_img if camera_img else uploaded_img

# =====================================================
# PROCESS IMAGE
# =====================================================
if image_source:
    img = Image.open(image_source).convert("RGB")
    img_np = np.array(img)
    frame_w = img_np.shape[1]

    results = model(img_np)[0]
    draw = ImageDraw.Draw(img)

    rows = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        size = estimate_size(xyxy, frame_w)
        gol = classify_vehicle(label, size)

        # CROP
        crop = img.crop(xyxy)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()

        rows.append({
            "Gambar Kendaraan": f'<img src="data:image/png;base64,{encoded}" width="120"/>',
            "Kendaraan": label,
            "Golongan": gol,
            "Confidence": round(conf, 3)
        })

        # DRAW BOX
        x1, y1, x2, y2 = xyxy
        color = CLASS_COLORS.get(label, "white")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        text = f"{label} · {gol} · {conf:.2f}"
        tw = draw.textlength(text)
        draw.rectangle([x1, y1 - 24, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - 22), text, fill="black")

    # =================================================
    # RESULT TABLE
    # =================================================
    st.subheader("📊 Hasil Deteksi")
    if rows:
        df = pd.DataFrame(rows)
        st.write(df.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.info("Tidak ada kendaraan terdeteksi.")

    # =================================================
    # FINAL IMAGE
    # =================================================
    st.subheader("📸 Gambar Dengan Bounding Box")
    st.image(img, use_column_width=True)
