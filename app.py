import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

st.set_page_config(page_title="Deteksi Golongan Kendaraan", layout="wide")

st.title("🚗 Deteksi Golongan Kendaraan")
st.write("Ambil gambar kendaraan menggunakan kamera, lalu sistem akan mendeteksi jenis & golongannya, lengkap dengan bounding box berwarna.")

# Load model YOLO
model = YOLO("models/yolov8n.pt")

# Warna untuk setiap kelas
CLASS_COLORS = {
    "car": "red",
    "motorbike": "blue",
    "bus": "green",
    "truck": "yellow",
}

# Golongan sederhana
def classify_vehicle(label):
    if label in ["car", "motorbike"]:
        return "Golongan 1"
    elif label == "bus":
        return "Golongan 2"
    elif label == "truck":
        return "Golongan 3"
    return "Tidak diketahui"

# Ambil gambar dari kamera
img_data = st.camera_input("Ambil foto kendaraan")

if img_data:
    # Load image
    img = Image.open(img_data).convert("RGB")
    img_np = np.array(img)

    # Inference YOLO
    results = model(img_np)[0]

    # Untuk bounding box
    draw = ImageDraw.Draw(img)

    # List hasil deteksi
    detected_rows = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]
        conf = float(box.conf[0])
        gol = classify_vehicle(label)

        # Buat row untuk tabel
        detected_rows.append({
            "Kendaraan": label,
            "Golongan": gol,
            "Confidence": round(conf, 3)
        })

        # Ambil koordinat
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Warna bounding box
        color = CLASS_COLORS.get(label, "white")

        # Kotak
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        # Label + confidence
        text_label = f"{label} ({conf:.2f})"
        text_width = draw.textlength(text_label)
        text_height = 20

        draw.rectangle([x1, y1 - text_height, x1 + text_width + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - text_height + 2), text_label, fill="black")

    # ==========================
    #   Tampilkan Tabel Hasil
    # ==========================
    st.subheader("📊 Hasil Deteksi:")
    
    if len(detected_rows) > 0:
        df = pd.DataFrame(detected_rows)
        st.table(df)
    else:
        st.info("Tidak ada kendaraan terdeteksi.")

    # ==========================
    #   Tampilkan Gambar
    # ==========================
    st.subheader("📸 Gambar yang Diambil + Bounding Box:")
    st.image(img, use_column_width=True)
