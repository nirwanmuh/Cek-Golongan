import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import io

# Import aturan golongan
from golongan_rules import GOLONGAN_RULES

st.set_page_config(page_title="Deteksi Golongan Kendaraan I–XII", layout="wide")

st.title("🚗 Deteksi Golongan Kendaraan I–XII")
st.write("Sistem mendeteksi kendaraan, mengklasifikasi golongan, dan menampilkan crop tiap objek.")

# ===============================
#  Load YOLO
# ===============================
model = YOLO("models/yolov8n.pt")

# Warna berdasarkan kelas YOLO
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

# ===============================
#  Start capture
# ===============================
img_data = st.camera_input("Ambil foto kendaraan")

if img_data:
    img = Image.open(img_data).convert("RGB")
    img_np = np.array(img)
    frame_w = img_np.shape[1]

    results = model(img_np)[0]

    draw = ImageDraw.Draw(img)
    detected_rows = []

    for box in results.boxes:
        # Extract YOLO box data
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]
        conf = float(box.conf[0])

        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = xyxy

        # Determine size & golongan
        size = estimate_size(xyxy, frame_w)
        gol = classify_vehicle(label, size)

        # -------------------------
        # CROP IMAGE
        # -------------------------
        cropped = img.crop((x1, y1, x2, y2))

        # Convert crop ke PNG untuk display di DataFrame
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        buf.seek(0)

        # Row untuk tabel
        detected_rows.append({
            "Gambar Kendaraan": buf,
            "Kendaraan": label,
            "Golongan": gol,
            "Confidence": round(conf, 3),
        })

        # -------------------------
        # BOUNDING BOX DRAWING
        # -------------------------
        color = CLASS_COLORS.get(label, "white")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        text_label = f"{label} · {gol} · {conf:.2f}"
        tw = draw.textlength(text_label)
        th = 22

        draw.rectangle([x1, y1 - th, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th + 2), text_label, fill="black")

    # ===============================
    #  Display Results Table
    # ===============================
    st.subheader("📊 Hasil Deteksi:")

    if len(detected_rows) > 0:
        # Convert rows menjadi DataFrame yang bisa menampilkan gambar
        df = pd.DataFrame(detected_rows)

        # Render gambar di tabel
        def image_formatter(img_bytes):
            return f'<img src="data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}" width="120"/>'

        import base64
        st.write(
            df.to_html(escape=False, formatters={"Gambar Kendaraan": image_formatter}),
            unsafe_allow_html=True
        )
    else:
        st.info("Tidak ada kendaraan terdeteksi.")

    # ===============================
    #  Display annotated image
    # ===============================
    st.subheader("📸 Gambar dengan Bounding Box:")
    st.image(img, use_column_width=True)
