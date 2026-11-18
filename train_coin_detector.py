import os
from ultralytics import YOLO

def latih_model_koin_universal(yaml_path="data_koin_global.yaml"):
    """
    Melatih model YOLOv8 instance segmentation secara fleksibel untuk mendeteksi
    koin dari berbagai negara, berbagai sudut kemiringan, dan tumpukan acak.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"File konfigurasi dataset tidak ditemukan: {yaml_path}")

    model = YOLO("yolov8s-seg.pt")

    model.train(
        data=yaml_path,
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        # Augmentasi geometris untuk sudut miring (perspektif elips) & tumpukan
        degrees=180.0,
        shear=20.0,
        perspective=0.001,
        scale=0.7,
        translate=0.1,
        mosaic=1.0,
        mixup=0.15,
        # Augmentasi warna untuk kilau logam (perak, emas, kuningan, tembaga)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.6,
        optimizer="AdamW",
        lr0=1e-3,
        save=True,
        project="hasil_training_koin",
        name="model_fleksibel_v1"
    )

    print("[SUKSES] Bobot model terbaik tersimpan di: hasil_training_koin/model_fleksibel_v1/weights/best.pt")

if __name__ == "__main__":
    latih_model_koin_universal()