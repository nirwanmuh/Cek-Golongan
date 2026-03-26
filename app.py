import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

st.set_page_config(page_title="Deteksi Golongan Kendaraan", layout="wide")

st.title("🚗 Deteksi Golongan Kendaraan")
st.write("Ambil gambar kendaraan menggunakan kamera, lalu model akan mendeteksi jenisnya dengan bounding box warna per kelas.")

# Load YOLO
model = YOLO("models/yolov8n.pt")

# Warna per kelas
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

# Ambil gambar
img_data = st.camera_input("Ambil foto kendaraan")

if img_data:
    img = Image.open(img_data).convert("RGB")
    img_np = np.array(img)

    # Inference
    results = model(img_np)[0]

    # Siapkan draw
    draw = ImageDraw.Draw(img)

    detected = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]
        conf = float(box.conf[0])
        gol = classify_vehicle(label)

        detected.append({
            "kendaraan": label,
            "golongan": gol,
            "confidence": round(conf, 3)
        })

        # Koordinat bounding box
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Pilih warna
        color = CLASS_COLORS.get(label, "white")

        # Gambar bounding box (lebih rapi, tebal)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        # Label text: "label (conf)"
        text_label = f"{label} ({conf:.2f})"

        # Background text
        text_width = draw.textlength(text_label)
        text_height = 20

        draw.rectangle([x1, y1 - text_height, x1 + text_width + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - text_height + 2), text_label, fill="black")

    st.subheader("Hasil Deteksi:")
    st.json(detected)

    st.subheader("Gambar yang Diambil + Bounding Box:")
    st.image(img, use_column_width=True)
