# =======================================
# Golongan I–XII (Reformulasi PM 66/2019)
# =======================================

# Klasifikasi ukuran (hasil estimate_size)
# - small
# - medium
# - large
# - xlarge
# - xxlarge

GOLONGAN_RULES = [
    # 1. KENDARAAN TIDAK BERMOTOR
    {"golongan": "Golongan I", "yolo": ["bicycle"], "size": "any"},

    # 2. SEPEDA MOTOR KECIL
    {"golongan": "Golongan II", "yolo": ["motorbike"], "size": "small"},

    # 3. SEPEDA MOTOR BESAR
    {"golongan": "Golongan III", "yolo": ["motorbike"], "size": "medium"},

    # 4A. MOBIL PENUMPANG KECIL
    {"golongan": "Golongan IVA", "yolo": ["car"], "size": "small"},

    # 4B. MOBIL PENUMPANG BESAR / DOUBLE CABIN
    {"golongan": "Golongan IVB", "yolo": ["car"], "size": "medium"},

    # 5A. BUS / MINIBUS KECIL
    {"golongan": "Golongan VA", "yolo": ["bus"], "size": "small"},

    # 5B. TRUK KECIL
    {"golongan": "Golongan VB", "yolo": ["truck"], "size": "small"},

    # 6A. BUS SEDANG
    {"golongan": "Golongan VIA", "yolo": ["bus"], "size": "medium"},

    # 6B. TRUK SEDANG
    {"golongan": "Golongan VIB", "yolo": ["truck"], "size": "medium"},

    # 7. TRUK BESAR (3 SUMBU)
    {"golongan": "Golongan VII", "yolo": ["truck"], "size": "large"},

    # 8. KENDARAAN GANDENGAN
    {"golongan": "Golongan VIII", "yolo": ["truck"], "size": "xlarge"},

    # 9. TRAILER PANJANG
    {"golongan": "Golongan IX", "yolo": ["truck"], "size": "xxlarge"},
]
