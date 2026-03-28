# Golongan Kendaraan I–XII
# Versi AKURAT & REALISTIS (IVA tidak pakai ukuran)

GOLONGAN_RULES = [
    # Kendaraan tidak bermotor
    {"golongan": "Golongan I", "yolo": ["bicycle"], "size": "any"},

    # Sepeda motor
    {"golongan": "Golongan II", "yolo": ["motorbike"], "size": "small"},
    {"golongan": "Golongan III", "yolo": ["motorbike"], "size": "medium"},

    # ✅ MOBIL PENUMPANG: SELALU IVA
    {"golongan": "Golongan IVA", "yolo": ["car"], "size": "any"},

    # Bus
    {"golongan": "Golongan VA", "yolo": ["bus"], "size": "small"},
    {"golongan": "Golongan VIA", "yolo": ["bus"], "size": "medium"},

    # Truk
    {"golongan": "Golongan VB", "yolo": ["truck"], "size": "small"},
    {"golongan": "Golongan VIB", "yolo": ["truck"], "size": "medium"},
    {"golongan": "Golongan VII", "yolo": ["truck"], "size": "large"},
    {"golongan": "Golongan VIII", "yolo": ["truck"], "size": "xlarge"},
    {"golongan": "Golongan IX", "yolo": ["truck"], "size": "xxlarge"},
]
