import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

# Import aturan golongan
from golongan_rules import GOLONGAN_RULES

st.set_page_config(page_title="Deteksi Golongan Kendaraan I–XII", layout="wide")

st.title("🚗 Deteksi Golongan Kendaraan (I–XII) - PM 66/2019")
st.write("Sistem mendeteksi kendaraan dan menentukan golongan berdasarkan label + ukuran.")

# =======================================
# Muat model YOLO
# =======================================
model = YOLO("models/yolov8n.pt")

# Warna per kelas YOLO
CLASS_COLORS = {
    "car": "red",
    "motorbike": "blue",
    "bus": "green",
    "truck": "yellow",
    "bicycle": "orange",
}

# =======================================
# Estimasi ukuran kendaraan dari bounding box
# =======================================
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

# =======================================
# Tentukan golongan
# =======================================
def classify_vehicle(label, size):
    for rule in GOLONGAN_RULES:
        if label in rule["yolo"]:
            if rule["size"] == "any" or rule["size"] == size:
                return rule["golongan"]
    return "Tidak diketahui"

# =======================================
# Ambil gambar
# =======================================
img_data = st.camera_input("Ambil foto kendaraan")

if img_data:
    img = Image.open(img_data).convert("RGB")
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
        size = estimate_size(xyxy, frame_w)
        gol = classify_vehicle(label, size)

        detected_rows.append({
            "Kendaraan": label,
            "Ukuran": size,
            "Golongan": gol,
            "Confidence": round(conf, 3),
        })

        # Warna box
        color = CLASS_COLORS.get(label, "white")

        x1, y1, x2, y2 = xyxy

        # BOUNDING BOX
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        # Label
        text_label = f"{label} · {gol} · {conf:.2f}"
        tw = draw.textlength(text_label)
        th = 22

        draw.rectangle([x1, y1 - th, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th + 2), text_label, fill="black")

    # TABEL HASIL
    st.subheader("📊 Hasil Deteksi:")
    df = pd.DataFrame(detected_rows)
    st.table(df)

    # GAMBAR
    st.subheader("📸 Gambar dengan Bounding Box:")
    st.image(img, use_column_width=True)
