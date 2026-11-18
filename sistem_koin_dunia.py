import os
import cv2
import json
import base64
import numpy as np
from collections import Counter
from ultralytics import YOLO

try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

KATALOG_KOIN = {
    "ID_IDR_100":  {"negara": "Indonesia", "mata_uang": "IDR", "simbol": "Rp ", "nominal": 100, "kurs_idr": 1.0},
    "ID_IDR_200":  {"negara": "Indonesia", "mata_uang": "IDR", "simbol": "Rp ", "nominal": 200, "kurs_idr": 1.0},
    "ID_IDR_500":  {"negara": "Indonesia", "mata_uang": "IDR", "simbol": "Rp ", "nominal": 500, "kurs_idr": 1.0},
    "ID_IDR_1000": {"negara": "Indonesia", "mata_uang": "IDR", "simbol": "Rp ", "nominal": 1000, "kurs_idr": 1.0},
    "US_USD_025":  {"negara": "Amerika Serikat", "mata_uang": "USD", "simbol": "$", "nominal": 0.25, "kurs_idr": 16200.0},
    "EU_EUR_100":  {"negara": "Uni Eropa", "mata_uang": "EUR", "simbol": "€", "nominal": 1.0, "kurs_idr": 17500.0},
    "JP_JPY_100":  {"negara": "Jepang", "mata_uang": "JPY", "simbol": "¥", "nominal": 100, "kurs_idr": 106.0},
    "MY_MYR_050":  {"negara": "Malaysia", "mata_uang": "MYR", "simbol": "RM ", "nominal": 0.50, "kurs_idr": 3650.0},
}

DEFAULT_WEIGHTS = "hasil_training_koin/model_fleksibel_v1/weights/best.pt"
_loaded_model = None

def muat_model(path_model=DEFAULT_WEIGHTS):
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model
    if os.path.exists(path_model):
        _loaded_model = YOLO(path_model)
    else:
        print(f"[INFO] Bobot kustom '{path_model}' belum tersedia, menggunakan fallback 'yolov8s-seg.pt'.")
        _loaded_model = YOLO("yolov8s-seg.pt")
    return _loaded_model

def proses_array_gambar(img, target_w=800, target_h=520, path_model=DEFAULT_WEIGHTS):
    """
    Menjalankan inferensi YOLOv8-Seg & fitEllipse serta mengembalikan hasil
    dalam format kanvas (x, y, rx, ry, code) untuk integrasi front-end.
    """
    model = muat_model(path_model)
    orig_h, orig_w = img.shape[:2]
    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)

    results = model.predict(source=img, conf=0.35, iou=0.45, verbose=False)[0]
    annotated = img.copy()

    total_koin = 0
    terklasifikasi = 0
    tidak_terklasifikasi = 0
    total_estimasi_idr = 0.0
    rincian_jenis = Counter()
    daftar_koin_kanvas = []

    boxes = results.boxes
    masks = results.masks.xy if results.masks is not None else [None] * len(boxes)

    for box, poly in zip(boxes, masks):
        total_koin += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        kode_kelas = model.names.get(cls_id, "UNCLASSIFIED")

        rasio_elips = 1.0
        cx_orig = (x1 + x2) / 2.0
        cy_orig = (y1 + y2) / 2.0
        rx_orig = (x2 - x1) / 2.0
        ry_orig = (y2 - y1) / 2.0

        if poly is not None and len(poly) >= 5:
            pts = np.array(poly, dtype=np.int32)
            (center, (ma, MA), _) = cv2.fitEllipse(pts)
            rasio_elips = min(ma, MA) / (max(ma, MA) + 1e-5)
            cx_orig, cy_orig = center
            rx_orig = max(ma, MA) / 2.0
            ry_orig = min(ma, MA) / 2.0

        if conf >= 0.55 and rasio_elips >= 0.35 and kode_kelas in KATALOG_KOIN:
            terklasifikasi += 1
            info = KATALOG_KOIN[kode_kelas]
            nilai_idr = info["nominal"] * info["kurs_idr"]
            total_estimasi_idr += nilai_idr
            rincian_jenis[f"{info['negara']} ({info['simbol']}{info['nominal']})"] += 1
            warna = (16, 185, 129)
            label = f"#{total_koin} {info['simbol']}{info['nominal']}"
            kode_final = kode_kelas
        else:
            tidak_terklasifikasi += 1
            warna = (11, 158, 245)
            label = f"#{total_koin} Unclassified"
            kode_final = "UNCLASSIFIED"

        daftar_koin_kanvas.append({
            "x": int(round(cx_orig * scale_x)),
            "y": int(round(cy_orig * scale_y)),
            "rx": max(10, int(round(rx_orig * scale_x))),
            "ry": max(6, int(round(ry_orig * scale_y))),
            "code": kode_final
        })

        if poly is not None and len(poly) >= 5:
            cv2.polylines(annotated, [np.array(poly, dtype=np.int32)], True, warna, 2)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), warna, 2)

        cv2.putText(annotated, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)

    laporan = {
        "total_semua_koin": total_koin,
        "berhasil_diklasifikasi": terklasifikasi,
        "tidak_terklasifikasi": tidak_terklasifikasi,
        "total_estimasi_idr": round(total_estimasi_idr),
        "rincian_per_jenis": dict(rincian_jenis),
        "daftar_koin": daftar_koin_kanvas
    }
    return laporan, annotated

def analisis_gambar_koin(path_gambar, path_model=DEFAULT_WEIGHTS):
    img = cv2.imread(path_gambar)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path_gambar}")

    h_canvas = max(260, int(round((img.shape[0] / float(img.shape[1])) * 800)))
    laporan, annotated = proses_array_gambar(img, target_w=800, target_h=h_canvas, path_model=path_model)

    print(json.dumps(laporan, indent=4, ensure_ascii=False))
    cv2.imwrite("hasil_deteksi_koin.jpg", annotated)
    return laporan

if FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        return send_from_directory(".", "aplikasi_deteksi_koin_v10.html")

    @app.route("/api/catalog", methods=["GET", "POST"])
    def handle_catalog():
        if request.method == "POST":
            data = request.get_json(force=True)
            kode = data.get("kode")
            if kode:
                KATALOG_KOIN[kode] = {
                    "negara": data.get("negara", "Unknown"),
                    "mata_uang": data.get("mata_uang", "UNK"),
                    "simbol": data.get("simbol", ""),
                    "nominal": float(data.get("nominal", 0)),
                    "kurs_idr": float(data.get("kurs_ke_idr", 1.0))
                }
        return jsonify({"status": "ok", "katalog": KATALOG_KOIN})

    @app.route("/api/detect", methods=["POST"])
    def detect_endpoint():
        payload = request.get_json(force=True)
        img_b64 = payload.get("image_base64", "")
        target_w = int(payload.get("canvas_w", 800))
        target_h = int(payload.get("canvas_h", 520))

        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Gagal membaca gambar"}), 400

        laporan, _ = proses_array_gambar(img, target_w=target_w, target_h=target_h)
        return jsonify({"status": "ok", "laporan": laporan})

if __name__ == "__main__":
    contoh_path = "contoh_tumpukan_koin.jpg"
    if os.path.exists(contoh_path):
        analisis_gambar_koin(contoh_path)

    if FLASK_AVAILABLE:
        print("[SERVER] Menjalankan server integrasi di http://127.0.0.1:5000")
        app.run(host="127.0.0.1", port=5000, debug=False)
    else:
        print("[INFO] Instal flask & flask-cors (`pip install flask flask-cors`) untuk mengaktifkan server API.")