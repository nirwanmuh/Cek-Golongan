import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import io
import base64

# Import aturan golongan
from golongan_rules import GOLONGAN_RULES

st.set_page_config(page_title="Deteksi Golongan Kendaraan I–XII", layout="wide")

st.title("🚗 Deteksi Golongan Kendaraan I–XII")
st.write("Sistem mendeteksi kendaraan dari kamera atau upload foto, menentukan golongan, dan menampilkan crop tiap kendaraan.")

# ===============================
#  Load YOLO
# ===============================
model = YOLO("models/yolov8n.pt")

# Warna per kelas YOLO
CLASS_COLORS = {
    "car": "red",
    "motorbike": "blue",
    "bus": "green",
    "truck": "yellow",
    "bicycle": "orange",
}

# ===============================
#  Estimate size from bounding box
# ===============================
def estimate_size(box, frame_width):
    x1, y1, x2, y2 = box
    w = x2 - x1
    rel = w / frame_width

    if rel < 0.25:
        return "small"
    elif rel < 0.40:
        return "medium"
    elif rel < 0.65:
        return "large"
    elif rel < 0.85:
        return "xlarge"
    else:
        return "xxlarge"

# ===============================
#  Golongan classification
# ===============================
def classify_vehicle(label, size):
    for rule in GOLONGAN_RULES:
        if label in rule["yolo"]:
            if rule["size"] == "any" or rule["size"] == size:
                return rule["golongan"]
    return "Tidak diketahui"

# =====================================================
# INPUT: Kamera ATAU Upload Foto
# =====================================================

col1, col2 = st.columns(2)

with col1:
    img_data = st.camera_input("Ambil foto kendaraan")

with col2:
    uploaded_file = st.file_uploader("Atau upload foto kendaraan", type=["jpg", "jpeg", "png"])

# Tentukan gambar mana yang dipakai
image_source = None

if img_data:
    image_source = img_data
elif uploaded_file:
    image_source = uploaded_file

# =====================================================
# Proses gambar jika ada input
# =====================================================
if image_source:
    img = Image.open(image_source).convert("RGB")
    img_np = np.array(img)
    frame_w = img_np.shape[1]

    results = model(img_np)[0]

    draw = ImageDraw.Draw(img)
    detected_rows = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]
        conf = float(box.conf[0])

        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = xyxy

        # Tentukan ukuran + golongan
        size = estimate_size(xyxy, frame_w)
        golongan = classify_vehicle(label, size)

        # ---------------------
        #  CROP KENDARAAN
        # ---------------------
        cropped = img.crop((x1, y1, x2, y2))

        # Convert crop to PNG for HTML
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()

        # Masukkan ke tabel
        detected_rows.append({
            "Gambar Kendaraan": f'<img src="data:image/png;base64,{encoded}" width="120">',
            "Kendaraan": label,
            "Golongan": golongan,
            "Confidence": round(conf, 3),
        })

        # ---------------------
        #  Gambar bounding box
        # ---------------------
        color = CLASS_COLORS.get(label, "white")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        text = f"{label} · {golongan} · {conf:.2f}"
        tw = draw.textlength(text)
        th = 22

        draw.rectangle([x1, y1 - th, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th + 2), text, fill="black")

    # =========================
    # TABEL HASIL DETEKSI
    # =========================
    st.subheader("📊 Hasil Deteksi")

    if len(detected_rows) > 0:
        df = pd.DataFrame(detected_rows)
        st.write(df.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.info("Tidak ada kendaraan terdeteksi.")

    # =========================
    # GAMBAR BOUNDING BOX
    # =========================
    st.subheader("📸 Gambar Dengan Bounding Box:")
    st.image(img, use_column_width=True)
