import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Deteksi Golongan Kendaraan", layout="wide")

st.title("🚗 Deteksi Golongan Kendaraan")
st.write("Ambil gambar kendaraan menggunakan kamera, lalu model akan mendeteksi jenisnya.")

# Load model
model = YOLO("models/yolov8n.pt")

# Klasifikasi golongan sederhana
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
    img = Image.open(img_data)
    img_np = np.array(img)

    # Model inference
    results = model(img_np)[0]

    detected = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]
        gol = classify_vehicle(label)
        detected.append({"kendaraan": label, "golongan": gol})

    st.subheader("Hasil Deteksi:")
    st.json(detected)

    st.image(img_np, caption="Gambar yang diambil", use_column_width=True)
