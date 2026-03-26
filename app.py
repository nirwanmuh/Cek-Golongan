import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(page_title="Deteksi Golongan Kendaraan", layout="wide")

st.title("🚗 Deteksi Golongan Kendaraan")
st.write("Ambil gambar kendaraan menggunakan kamera, lalu model akan mendeteksi jenisnya.")

# Load YOLO
model = YOLO("models/yolov8n.pt")

# Golongan sederhana
def classify_vehicle(label):
    if label in ["car", "motorbike"]:
        return "Golongan 1"
    elif label in ["bus"]:
        return "Golongan 2"
    elif label in ["truck"]:
        return "Golongan 3"
    return "Tidak diketahui"

# Ambil gambar
img_data = st.camera_input("Ambil foto kendaraan")

if img_data:
    img = Image.open(img_data).convert("RGB")
    img_np = np.array(img)

    # Inference
    results = model(img_np)[0]

    detected = []

    # Gambar bounding box
    draw = ImageDraw.Draw(img)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]
        gol = classify_vehicle(label)

        detected.append({"kendaraan": label, "golongan": gol})

        # Ambil koordinat xyxy
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Gambar kotak
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

        # Tambahkan label di atas kotak
        draw.text((x1, y1 - 10), f"{label} ({gol})", fill="red")

    st.subheader("Hasil Deteksi:")
    st.json(detected)

    st.subheader("Gambar yang Diambil + Bounding Box:")
    st.image(img, use_column_width=True)
