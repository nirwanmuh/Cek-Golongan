import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Realtime Polling Deteksi Kendaraan", layout="wide")

st.title("🚗 Deteksi Golongan Kendaraan (Polling Streamlit)")
st.write("Mengambil gambar otomatis setiap beberapa detik untuk deteksi semi-realtime.")

# Load model YOLO
model = YOLO("models/yolov8n.pt")

# Golongan kendaraan sederhana
def classify_vehicle(label):
    if label in ["car", "motorbike"]:
        return "Golongan 1"
    elif label in ["bus"]:
        return "Golongan 2"
    elif label in ["truck"]:
        return "Golongan 3"
    else:
        return "Tidak diketahui"

# Tempat menaruh hasil
result_placeholder = st.empty()
image_placeholder = st.empty()

# Interval polling (detik)
poll_interval = 1

st.write("📸 Kamera akan mengambil gambar otomatis setiap", poll_interval, "detik.")

while True:
    # Key unik setiap loop agar Streamlit mengambil gambar baru
    img_data = st.camera_input("Kamera", key=str(time.time()))

    if img_data:
        # Process frame
        img = Image.open(img_data)
        img_np = np.array(img)

        # YOLO inference
        results = model(img_np)[0]

        detected = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.model.names[cls_id]
            gol = classify_vehicle(label)
            detected.append({"kendaraan": label, "golongan": gol})

        # Tampilkan hasil
        result_placeholder.json(detected)
        image_placeholder.image(img_np, caption="Frame terbaru", use_column_width=True)

    time.sleep(poll_interval)
