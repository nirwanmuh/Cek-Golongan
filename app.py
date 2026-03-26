import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Deteksi Golongan Kendaraan", layout="wide")

st.title("🚗 Deteksi Golongan Kendaraan (Real-Time)")
st.write("Aplikasi ini mendeteksi jenis kendaraan dan menentukan golongan secara otomatis.")

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

# Ambil gambar dari webcam
img_data = st.camera_input("Ambil gambar kendaraan")

if img_data:
    # Convert ke array
    img = Image.open(img_data)
    img = np.array(img)

    # Model inference
    results = model(img)[0]

    st.subheader("Hasil Deteksi:")

    vehicle_list = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]
        gol = classify_vehicle(label)

        vehicle_list.append((label, gol))

    st.json([
        {"kendaraan": v[0], "golongan": v[1]}
        for v in vehicle_list
    ])

    # Tampilkan gambar
    st.image(img, caption="Input Image", use_column_width=True)
