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

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Deteksi Golongan Kendaraan I–XII",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CSS =================
st.markdown("""
<style>
.block-container { padding: 0.8rem; }

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

# ================= HEADER =================
st.title("🚗 Deteksi Golongan Kendaraan I–XII")
st.caption("Golongan IVA diperbaiki: mobil penumpang tidak lagi salah diklasifikasikan.")

# ================= LOAD MODEL =================
model = YOLO("models/yolov8n.pt")

CLASS_COLORS = {
    "car": "red",
    "motorbike": "blue",
    "bus": "green",
    "truck": "yellow",
    "bicycle": "orange",
}

# ================= HELPERS =================
def estimate_size(box, frame_width):
    x1, _, x2, _ = box
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

# ================= INPUT =================
st.subheader("📸 Ambil foto kendaraan")
camera_img = st.camera_input("")

st.subheader("📤 Atau upload foto kendaraan")
uploaded_img = st.file_uploader("", type=["jpg", "jpeg", "png"])

image_source = camera_img if camera_img else uploaded_img

# ================= PROCESS =================
if image_source:
    img = Image.open(image_source).convert("RGB")
    np_img = np.array(img)
    frame_w = np_img.shape[1]

    results = model(np_img)[0]
    draw = ImageDraw.Draw(img)

    rows = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        size = estimate_size(xyxy, frame_w)
        golongan = classify_vehicle(label, size)

        crop = img.crop(xyxy)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()

        rows.append({
            "Gambar": f'data:image/png;base64,{encoded}',
            "Kendaraan": label,
            "Golongan": golongan,
            "Confidence": round(conf, 3)
        })

        color = CLASS_COLORS.get(label, "white")
        x1, y1, x2, y2 = xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        text = f"{label} · {golongan} · {conf:.2f}"
        draw.rectangle([x1, y1-24, x1+len(text)*9, y1], fill=color)
        draw.text((x1+3, y1-22), text, fill="black")

    st.subheader("📊 Hasil Deteksi")
    df = pd.DataFrame(rows)
    st.write(df.to_html(escape=False), unsafe_allow_html=True)

    st.subheader("📸 Gambar Dengan Bounding Box")
    st.image(img, use_column_width=True)
