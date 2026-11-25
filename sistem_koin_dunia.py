# -*- coding: utf-8 -*-
"""
sistem_koin_dunia.py
====================
Deteksi, penghitungan, dan estimasi nilai Koin Rupiah Indonesia (IDR).
100% Murni Pengolahan Citra Digital (OpenCV).

Pipeline OpenCV:
  1. Estimasi warna/permukaan alas di ruang warna CIE Lab (mode histogram dari tepi + sudut,
     lalu fit bidang linear agar tahan gradasi cahaya / lipatan kain / serat kayu).
  2. Foreground = jarak Lab terhadap permukaan alas, dengan ambang adaptif (Otsu + noise alas).
  3. Isi lubang HANYA jika lubangnya cembung (kilau/relief di dalam koin). Celah di antara
     koin-koin yang bersentuhan berbentuk cekung sehingga TIDAK diisi -> tidak ada koin hantu.
  4. Distance Transform -> puncak lokal -> NMS berbasis radius (1 koin = 1 puncak).
  5. Watershed dengan marker besar (lingkaran dalam koin) memisahkan koin yang bersentuhan.
  6. cv2.fitEllipse pada busur luar yang bebas (titik potong watershed dibuang), fallback momen.
  7. Validasi (soliditas, rasio sumbu, ukuran) + dedup tumpang tindih -> 1 koin = tepat 1 elips.

Pecahan Koin Rupiah yang dikenali:
  - Rp 100   (aluminium/perak, diameter 23.0 mm)
  - Rp 200   (aluminium/perak, diameter 25.0 mm)
  - Rp 500 Perak  (aluminium/perak, diameter 27.0 mm)
  - Rp 500 Kuningan (kuning emas, diameter 24.0 mm)
  - Rp 1000  (nikel/perak tebal, diameter 24.1 mm)

Endpoint:
  /            -> halaman HTML frontend
  /api/detect  -> deteksi koin (POST)
  /api/health  -> status server (GET)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import json
import math
import base64
import io
import numpy as np
from collections import Counter

try:
    from flask import Flask, request, jsonify, send_from_directory, make_response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Cek ketersediaan rembg (opsional)
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    rembg_remove = None
    REMBG_AVAILABLE = False


# ========================================================================================
# KATALOG KOIN RUPIAH INDONESIA
# ========================================================================================
KATALOG_KOIN = {
    "IDR_100":        {"nominal": 100,  "simbol": "Rp ", "diameter_mm": 23.0, "warna": "perak",    "label": "Rp 100"},
    "IDR_200":        {"nominal": 200,  "simbol": "Rp ", "diameter_mm": 25.0, "warna": "perak",    "label": "Rp 200"},
    "IDR_500_PERAK":  {"nominal": 500,  "simbol": "Rp ", "diameter_mm": 27.0, "warna": "perak",    "label": "Rp 500"},
    "IDR_500_KUNING": {"nominal": 500,  "simbol": "Rp ", "diameter_mm": 24.0, "warna": "kuningan", "label": "Rp 500 (Kuningan)"},
    "IDR_1000":       {"nominal": 1000, "simbol": "Rp ", "diameter_mm": 24.1, "warna": "perak",    "label": "Rp 1000"},
}


# ========================================================================================
# KONSTANTA PROSES OPENCV
# ========================================================================================
PANJANG_PROSES = 720                                       # sisi terpanjang gambar saat diproses
BOBOT_LAB = np.array([0.85, 1.0, 1.0], dtype=np.float32)  # bobot jarak (L, a, b)
RADIUS_LATAR = 24.0                                        # radius klaster warna alas (satuan Lab)


# ========================================================================================
# UTILITAS UMUM
# ========================================================================================
def _ganjil(n, minimum=3):
    n = int(round(n))
    n = max(minimum, n)
    return n if n % 2 == 1 else n + 1


def _kernel_elips(ukuran):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ukuran, ukuran))


def _normalisasi_sudut(rad):
    """Normalisasi sudut elips ke rentang [-pi/2, pi/2) (elips simetris 180 derajat)."""
    return (rad + math.pi / 2.0) % math.pi - math.pi / 2.0


def _ke_int(nilai, default):
    try:
        return int(float(nilai))
    except Exception:
        return default


def _ke_bool(nilai, default=False):
    """Parse boolean dari form/JSON: 'true'/'1'/True -> True."""
    if isinstance(nilai, bool):
        return nilai
    if isinstance(nilai, str):
        return nilai.lower() in ("true", "1", "yes")
    return default


def _param_elips(ex, ey, a1, a2, sudut_deg):
    """
    Konversi keluaran cv2.fitEllipse (panjang sumbu penuh a1, a2 + sudut derajat) ke
    (cx, cy, rx, ry, sudut_rad) dengan rx = setengah sumbu PANJANG.
    """
    if a1 >= a2:
        rx, ry, sud = a1 / 2.0, a2 / 2.0, math.radians(sudut_deg)
    else:
        rx, ry, sud = a2 / 2.0, a1 / 2.0, math.radians(sudut_deg + 90.0)
    return float(ex), float(ey), float(rx), float(ry), _normalisasi_sudut(sud)


def _elips_dari_momen(m):
    """Elips ekuivalen dari momen kontur (fallback bila fitEllipse tidak masuk akal)."""
    if m["m00"] <= 1e-6:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    akar = math.sqrt(max(((mu20 - mu02) / 2.0) ** 2 + mu11 ** 2, 0.0))
    l1 = (mu20 + mu02) / 2.0 + akar
    l2 = (mu20 + mu02) / 2.0 - akar
    rx = 2.0 * math.sqrt(max(l1, 1e-6))
    ry = 2.0 * math.sqrt(max(l2, 1e-6))
    sudut = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    return float(cx), float(cy), float(rx), float(ry), _normalisasi_sudut(sudut)


def elips_ke_kanvas(cx, cy, rx, ry, angle, sx, sy):
    """
    Transformasi elips (rx, ry, angle) di bawah penskalaan anisotropik diag(sx, sy).
    Hasilnya tetap elips yang tepat (sumbu = nilai singular, rotasi = vektor singular kiri),
    sehingga overlay di kanvas frontend tetap pas walau gambar diregangkan.
    """
    c, s = math.cos(angle), math.sin(angle)
    m = np.array([[c * rx, -s * ry], [s * rx, c * ry]], dtype=np.float64)
    ms = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64) @ m
    u, sv, _ = np.linalg.svd(ms)
    rx_n, ry_n = float(sv[0]), float(sv[1])
    ang_n = _normalisasi_sudut(math.atan2(u[1, 0], u[0, 0]))
    return cx * sx, cy * sy, rx_n, ry_n, ang_n


def _raster_elips(e, x_min, y_min, f, ukuran):
    m = np.zeros(ukuran, dtype=np.uint8)
    pusat = (int(round((e["cx"] - x_min) / f)), int(round((e["cy"] - y_min) / f)))
    sumbu = (max(1, int(round(e["rx"] / f))), max(1, int(round(e["ry"] / f))))
    cv2.ellipse(m, pusat, sumbu, math.degrees(e["angle"]), 0, 360, 1, -1)
    return m


def fraksi_tumpang_tindih(e1, e2):
    """Luas irisan dua elips dibagi luas elips yang lebih kecil (0..1)."""
    if math.hypot(e1["cx"] - e2["cx"], e1["cy"] - e2["cy"]) >= e1["rx"] + e2["rx"]:
        return 0.0
    x_min = int(math.floor(min(e1["cx"] - e1["rx"], e2["cx"] - e2["rx"]))) - 1
    y_min = int(math.floor(min(e1["cy"] - e1["rx"], e2["cy"] - e2["rx"]))) - 1
    x_max = int(math.ceil(max(e1["cx"] + e1["rx"], e2["cx"] + e2["rx"]))) + 1
    y_max = int(math.ceil(max(e1["cy"] + e1["rx"], e2["cy"] + e2["rx"]))) + 1
    lebar = x_max - x_min + 1
    tinggi = y_max - y_min + 1
    if lebar <= 0 or tinggi <= 0:
        return 0.0
    f = max(1.0, max(lebar, tinggi) / 400.0)
    ukuran = (int(math.ceil(tinggi / f)) + 1, int(math.ceil(lebar / f)) + 1)
    m1 = _raster_elips(e1, x_min, y_min, f, ukuran)
    m2 = _raster_elips(e2, x_min, y_min, f, ukuran)
    a1 = int(np.count_nonzero(m1))
    a2 = int(np.count_nonzero(m2))
    if a1 == 0 or a2 == 0:
        return 0.0
    irisan = int(np.count_nonzero(m1 & m2))
    return irisan / float(min(a1, a2))


def urutkan_baca(daftar):
    """Urutkan koin per baris (atas ke bawah) lalu kiri ke kanan."""
    if not daftar:
        return daftar
    tinggi_baris = max(1.0, 1.8 * float(np.median([c["ry"] for c in daftar])))
    return sorted(daftar, key=lambda c: (int(round(c["cy"] / tinggi_baris)), c["cx"]))


def _laporan_kosong(engine):
    return {
        "total_semua_koin": 0,
        "berhasil_diklasifikasi": 0,
        "tidak_terklasifikasi": 0,
        "total_estimasi_idr": 0,
        "engine_digunakan": engine,
        "rincian_per_jenis": {},
        "daftar_koin": [],
        "pipeline_images": {},
    }


# ========================================================================================
# PREPROCESSING: PNG TRANSPARAN & REMBG & CLAHE
# ========================================================================================
def _handle_alpha_channel(img_raw):
    """Jika gambar punya alpha channel, timpa area transparan jadi hitam pekat."""
    if img_raw is None:
        return None
    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        alpha = img_raw[:, :, 3].astype(np.float32) / 255.0
        bgr = img_raw[:, :, :3].astype(np.float32)
        for c in range(3):
            bgr[:, :, c] *= alpha
        return bgr.astype(np.uint8)
    return img_raw


def _apply_rembg(img_bgr):
    """Gunakan rembg untuk mengganti latar jadi hitam pekat. Return (result, success)."""
    if not REMBG_AVAILABLE or rembg_remove is None:
        return img_bgr, False
    try:
        # rembg bekerja dengan PIL/bytes, kita konversi BGR -> PNG bytes -> rembg -> decode
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            return img_bgr, False
        result_bytes = rembg_remove(buf.tobytes())
        np_arr = np.frombuffer(result_bytes, np.uint8)
        result_rgba = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if result_rgba is None:
            return img_bgr, False
        # Konversi RGBA -> BGR dengan latar hitam
        return _handle_alpha_channel(result_rgba), True
    except Exception:
        return img_bgr, False


def _apply_clahe(img_bgr):
    """Terapkan CLAHE pada channel L (CIE Lab) untuk meredam kilau logam (anti-glare)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ========================================================================================
# UTILITAS X-RAY: Encode gambar pipeline ke base64 JPEG (maks lebar 480px)
# ========================================================================================
def _encode_pipeline_image(img, max_width=480):
    """Encode gambar (grayscale atau BGR) ke base64 JPEG string yang ringan."""
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        new_w = max_width
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Pastikan 3-channel untuk encode
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    return base64.b64encode(buf).decode("ascii")


# ========================================================================================
# 1. ESTIMASI ALAS (CIE Lab)
# ========================================================================================
def konversi_lab_cie(bgr_uint8):
    """BGR uint8 -> Lab skala CIE sebenarnya (L 0..100, a/b sekitar -128..127)."""
    lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def jarak_lab(lab, ref):
    """Jarak warna Lab berbobot (L sedikit diredam agar tahan perbedaan pencahayaan)."""
    selisih = (lab - ref) * BOBOT_LAB
    return np.sqrt(np.sum(selisih * selisih, axis=-1))


def _mode_warna(pix):
    """Warna dominan (mode) dari kumpulan piksel Lab lewat histogram 3D terkuantisasi + mean-shift ringan."""
    q = np.floor((pix + np.array([0.0, 128.0, 128.0], dtype=np.float32)) / 8.0).astype(np.int32)
    q = np.clip(q, 0, 40)
    kunci = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    nilai, hitung = np.unique(kunci, return_counts=True)
    kunci_top = nilai[int(np.argmax(hitung))]
    pusat = np.median(pix[kunci == kunci_top], axis=0)
    for _ in range(4):
        d = jarak_lab(pix, pusat)
        anggota = d <= 14.0
        if np.count_nonzero(anggota) < 10:
            break
        pusat_baru = np.median(pix[anggota], axis=0)
        if np.allclose(pusat_baru, pusat, atol=0.05):
            pusat = pusat_baru
            break
        pusat = pusat_baru
    return pusat.astype(np.float32)


def kumpulkan_sampel_latar(lab):
    """
    Ambil piksel dari strip tepi gambar + 4 patch sudut (dijarangkan).
    Patch sudut penting untuk foto close-up: tepi bisa penuh koin, tapi sudut biasanya masih alas.
    """
    h, w = lab.shape[:2]
    sisi = min(h, w)
    tebal = max(1, min(max(3, int(sisi * 0.012)), h // 4, w // 4))
    sudut = max(tebal * 2, int(sisi * 0.10))
    sudut = max(1, min(sudut, h // 2, w // 2))

    m_tepi = np.zeros((h, w), dtype=bool)
    m_tepi[:tebal, :] = True
    m_tepi[h - tebal:, :] = True
    m_tepi[:, :tebal] = True
    m_tepi[:, w - tebal:] = True

    m_sudut = np.zeros((h, w), dtype=bool)
    m_sudut[:sudut, :sudut] = True
    m_sudut[:sudut, w - sudut:] = True
    m_sudut[h - sudut:, :sudut] = True
    m_sudut[h - sudut:, w - sudut:] = True
    grid = np.zeros((h, w), dtype=bool)
    grid[::2, ::2] = True
    m_sudut &= grid
    m_sudut &= ~m_tepi

    ys_t, xs_t = np.nonzero(m_tepi)
    ys_s, xs_s = np.nonzero(m_sudut)
    xs = np.concatenate([xs_t, xs_s])
    ys = np.concatenate([ys_t, ys_s])
    pix = lab[ys, xs].astype(np.float32)
    is_sudut = np.concatenate([np.zeros(len(xs_t), dtype=bool), np.ones(len(xs_s), dtype=bool)])
    return pix, xs, ys, is_sudut


def estimasi_model_latar(lab):
    """
    Estimasi model alas: warna pusat, noise, dan permukaan (bidang linear Lab) seukuran gambar.
    Beberapa kandidat mode dinilai dari porsi di tepi + (lebih berat) di sudut, lalu dipilih
    yang paling mirip alas. Ini mencegah koin besar di tepi dikira alas pada foto close-up.
    """
    h, w = lab.shape[:2]
    pix, xs, ys, is_sudut = kumpulkan_sampel_latar(lab)
    total = len(pix)
    n_tepi = max(1, int(np.count_nonzero(~is_sudut)))
    n_sudut = max(1, int(np.count_nonzero(is_sudut)))

    sisa = np.ones(total, dtype=bool)
    terbaik = None
    for _ in range(3):
        if np.count_nonzero(sisa) < max(30, int(0.05 * total)):
            break
        pusat = _mode_warna(pix[sisa])
        d = jarak_lab(pix, pusat)
        anggota = d <= RADIUS_LATAR
        frak_tepi = np.count_nonzero(anggota & ~is_sudut) / float(n_tepi)
        frak_sudut = np.count_nonzero(anggota & is_sudut) / float(n_sudut)
        skor = frak_tepi + 2.5 * frak_sudut
        if terbaik is None or skor > terbaik[0]:
            terbaik = (skor, pusat, anggota)
        sisa &= ~anggota

    if terbaik is None:
        pusat = np.median(pix, axis=0).astype(np.float32)
        terbaik = (0.0, pusat, jarak_lab(pix, pusat) <= RADIUS_LATAR)

    _, pusat, anggota = terbaik
    if np.count_nonzero(anggota) < 10:
        anggota = np.ones(total, dtype=bool)

    # Permukaan alas: default konstan; ganti bidang linear bila sampel cukup tersebar & masuk akal.
    permukaan = np.empty((h, w, 3), dtype=np.float32)
    permukaan[:, :] = pusat

    xa = xs[anggota].astype(np.float32) / max(w - 1, 1)
    ya = ys[anggota].astype(np.float32) / max(h - 1, 1)
    if xa.size >= 200 and float(xa.std()) >= 0.15 and float(ya.std()) >= 0.15:
        a_mat = np.stack([np.ones_like(xa), xa, ya], axis=1)
        pa = pix[anggota]
        koef = np.linalg.lstsq(a_mat, pa, rcond=None)[0]
        res = np.sqrt(np.sum(((pa - a_mat @ koef) * BOBOT_LAB) ** 2, axis=1))
        pakai = res <= max(5.0, 2.5 * float(np.std(res)))
        if np.count_nonzero(pakai) >= 100:
            koef = np.linalg.lstsq(a_mat[pakai], pa[pakai], rcond=None)[0]
        gx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
        gy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        kand = koef[0][None, None, :] + gx * koef[1][None, None, :] + gy * koef[2][None, None, :]
        dev = np.abs(kand - pusat[None, None, :])
        if dev[..., 0].max() <= 28.0 and dev[..., 1].max() <= 16.0 and dev[..., 2].max() <= 16.0:
            permukaan = kand.astype(np.float32)

    ref = permukaan[ys[anggota], xs[anggota]]
    d_ang = jarak_lab(pix[anggota], ref)
    noise = float(np.percentile(d_ang, 90)) if d_ang.size >= 10 else 4.0
    return {"pusat": pusat, "permukaan": permukaan, "noise": noise}


# ========================================================================================
# 2. MASK FOREGROUND
# ========================================================================================
def buat_mask_foreground(lab, model, sensitivitas):
    diff = jarak_lab(lab, model["permukaan"])
    d8 = np.clip(diff * 2.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = float(otsu) / 2.0

    thr_noise = float(np.clip(1.3 * model["noise"] + 4.0, 9.0, 40.0))
    thr = float(np.clip(0.65 * otsu, thr_noise, 45.0))
    p98 = float(np.percentile(diff, 98))
    thr = min(thr, max(thr_noise, 0.45 * p98))

    faktor = 1.20 - 0.004 * float(sensitivitas)  # sensitivitas tinggi -> ambang lebih rendah
    thr = float(np.clip(thr * faktor, thr_noise * 0.9, 60.0))

    fg = (diff > thr).astype(np.uint8) * 255
    return fg, thr


def isi_lubang_cembung(mask, soliditas_min=0.78, luas_noise=12):
    """
    Isi lubang di dalam foreground HANYA jika lubang itu bintik kecil atau berbentuk cembung
    (kilau logam / relief di dalam satu koin). Celah di antara koin-koin yang bersentuhan
    berbentuk cekung (soliditas rendah) sehingga dibiarkan -> tidak muncul koin hantu.
    """
    terpadding = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inv = cv2.bitwise_not(terpadding)
    n, label, stat, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    luar = label[0, 0]
    id_isi = []
    for i in range(1, n):
        if i == luar:
            continue
        luas = int(stat[i, cv2.CC_STAT_AREA])
        if luas <= luas_noise:
            id_isi.append(i)
            continue
        x = int(stat[i, cv2.CC_STAT_LEFT])
        y = int(stat[i, cv2.CC_STAT_TOP])
        bw = int(stat[i, cv2.CC_STAT_WIDTH])
        bh = int(stat[i, cv2.CC_STAT_HEIGHT])
        sub = (label[y:y + bh, x:x + bw] == i).astype(np.uint8)
        kontur, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not kontur:
            continue
        cnt = max(kontur, key=cv2.contourArea)
        luas_hull = cv2.contourArea(cv2.convexHull(cnt))
        if luas / max(luas_hull, 1.0) >= soliditas_min:
            id_isi.append(i)
    hasil = terpadding.copy()
    if id_isi:
        hasil[np.isin(label, id_isi)] = 255
    return np.ascontiguousarray(hasil[1:-1, 1:-1])


def bersihkan_mask(fg):
    h, w = fg.shape[:2]
    sisi = min(h, w)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _kernel_elips(_ganjil(sisi * 0.009)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _kernel_elips(_ganjil(sisi * 0.006)))
    return isi_lubang_cembung(fg)


# ========================================================================================
# 3. DISTANCE TRANSFORM -> PUNCAK KOIN
# ========================================================================================
def hitung_distance_transform(solid):
    """Distance transform dengan tepi gambar dianggap dinding (koin terpotong tidak bergeser pusatnya)."""
    terpadding = cv2.copyMakeBorder(solid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(terpadding, cv2.DIST_L2, 5)
    return np.ascontiguousarray(dist[1:-1, 1:-1])


def cari_puncak_koin(dist, min_r):
    """
    Puncak lokal Distance Transform (cepat, via dilasi) + NMS berbasis radius.
    Satu koin menghasilkan tepat satu puncak; dua koin bersentuhan menghasilkan dua puncak.
    """
    dist_s = cv2.GaussianBlur(dist, (0, 0), 1.2)
    k = _ganjil(min_r * 1.2)
    dil = cv2.dilate(dist_s, _kernel_elips(k))
    lokal = ((dist_s >= dil - 1e-4) & (dist_s >= min_r)).astype(np.uint8)
    n, label, stat, _ = cv2.connectedComponentsWithStats(lokal, connectivity=8)

    kandidat = []
    for i in range(1, n):
        x = int(stat[i, cv2.CC_STAT_LEFT])
        y = int(stat[i, cv2.CC_STAT_TOP])
        bw = int(stat[i, cv2.CC_STAT_WIDTH])
        bh = int(stat[i, cv2.CC_STAT_HEIGHT])
        anggota = (label[y:y + bh, x:x + bw] == i)
        nilai_sub = dist_s[y:y + bh, x:x + bw]
        nilai = float(nilai_sub[anggota].max())
        atas = anggota & (nilai_sub >= nilai - 0.25)
        ys_a, xs_a = np.nonzero(atas)
        if xs_a.size == 0:
            continue
        kandidat.append((x + float(xs_a.mean()), y + float(ys_a.mean()), nilai))

    kandidat.sort(key=lambda t: -t[2])
    puncak = []
    for (px, py, pr) in kandidat:
        terlalu_dekat = False
        for (ex, ey, er) in puncak:
            if math.hypot(px - ex, py - ey) < 0.85 * max(pr, er):
                terlalu_dekat = True
                break
        if not terlalu_dekat:
            puncak.append((px, py, pr))

    if puncak:
        r_ref = float(np.percentile([p[2] for p in puncak], 90))
        puncak = [p for p in puncak if p[2] >= max(min_r, 0.30 * r_ref)]
    return puncak


# ========================================================================================
# 4. WATERSHED
# ========================================================================================
def _partisi_terdekat(solid, puncak):
    """Fallback bila cv2.watershed gagal: partisi berbobot-radius (jarak ke tepi lingkaran terdekat)."""
    h, w = solid.shape[:2]
    bg_id = len(puncak) + 1
    yy, xx = np.indices((h, w))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    terbaik = np.full((h, w), np.inf, dtype=np.float32)
    indeks = np.zeros((h, w), dtype=np.int32)
    for i, (px, py, pr) in enumerate(puncak):
        d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2) - pr
        m = d < terbaik
        terbaik[m] = d[m]
        indeks[m] = i + 1
    return np.where(solid > 0, indeks, bg_id).astype(np.int32)


def segmentasi_watershed(citra_bgr, solid, puncak):
    """
    Watershed dengan marker besar (0.8 x radius lingkaran dalam tiap koin) dan marker alas pasti.
    Koin yang bersentuhan terpisah tepat di leher pertemuannya.
    """
    h, w = solid.shape[:2]
    bg_id = len(puncak) + 1
    penanda = np.zeros((h, w), dtype=np.int32)
    alas_pasti = cv2.dilate(solid, _kernel_elips(3)) == 0
    penanda[alas_pasti] = bg_id
    for i, (px, py, pr) in enumerate(puncak):
        cv2.circle(penanda, (int(round(px)), int(round(py))), max(2, int(round(pr * 0.8))), i + 1, -1)
    try:
        cv2.watershed(citra_bgr, penanda)
    except Exception:
        penanda = _partisi_terdekat(solid, puncak)
    return penanda, bg_id


# ========================================================================================
# 5. ELIPS PER KOIN
# ========================================================================================
def _celah_sudut_maks(pts, cx, cy):
    ang = np.sort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    celah = np.diff(np.concatenate([ang, [ang[0] + 2.0 * math.pi]]))
    return float(celah.max())


def analisis_wilayah(penanda, solid, hsv, lab_cie, puncak_i, label_id, bg_id):
    """
    Ambil wilayah satu koin hasil watershed, lalu fit elips pada busur luarnya.
    Juga hitung fitur warna: rata-rata saturasi/hue (HSV) dan skor is_gold (Lab).
    """
    px, py, pr = puncak_i
    h, w = solid.shape[:2]
    r = int(math.ceil(pr * 2.4)) + 4
    x0 = max(0, int(px) - r)
    x1 = min(w, int(px) + r + 1)
    y0 = max(0, int(py) - r)
    y1 = min(h, int(py) + r + 1)

    sub = penanda[y0:y1, x0:x1]
    reg = ((sub == label_id) & (solid[y0:y1, x0:x1] > 0)).astype(np.uint8) * 255
    reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, _kernel_elips(_ganjil(pr * 0.25)))

    kontur, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not kontur:
        return None
    cnt = max(kontur, key=cv2.contourArea)
    luas = float(cv2.contourArea(cnt))
    if luas < 30.0 or len(cnt) < 8:
        return None

    hull = cv2.convexHull(cnt)
    soliditas = luas / max(float(cv2.contourArea(hull)), 1.0)
    m = cv2.moments(cnt)
    if m["m00"] <= 1e-6:
        return None
    mx = m["m10"] / m["m00"]
    my = m["m01"] / m["m00"]

    pts = hull.reshape(-1, 2).astype(np.float32)
    lain = ((sub > 0) & (sub != label_id) & (sub != bg_id)).astype(np.uint8)
    lain = cv2.dilate(lain, np.ones((7, 7), dtype=np.uint8))
    ix = np.clip(pts[:, 0].astype(np.int32), 0, sub.shape[1] - 1)
    iy = np.clip(pts[:, 1].astype(np.int32), 0, sub.shape[0] - 1)
    terpotong = lain[iy, ix] > 0
    gx_abs = ix + x0
    gy_abs = iy + y0
    terpotong |= (gx_abs <= 1) | (gy_abs <= 1) | (gx_abs >= w - 2) | (gy_abs >= h - 2)

    bebas = pts[~terpotong]
    titik_fit = pts
    if len(bebas) >= 8 and _celah_sudut_maks(bebas, mx, my) <= math.radians(130.0):
        titik_fit = bebas

    elips = None
    if len(titik_fit) >= 6:
        try:
            (ex, ey), (a1, a2), sd = cv2.fitEllipse(titik_fit.reshape(-1, 1, 2).astype(np.float32))
            kand = _param_elips(ex, ey, a1, a2, sd)
            r_ekuiv = math.sqrt(luas / math.pi)
            r_fit = math.sqrt(max(kand[2] * kand[3], 1e-6))
            if 0.65 * r_ekuiv <= r_fit <= 1.45 * r_ekuiv and kand[3] / max(kand[2], 1e-6) >= 0.25:
                elips = kand
        except cv2.error:
            elips = None
    if elips is None:
        elips = _elips_dari_momen(m)
    if elips is None:
        return None

    ex, ey, rx, ry, sudut = elips

    # --- Fitur warna: HSV (saturasi, hue) + Lab (skor is_gold) ---
    inner = cv2.erode(reg, _kernel_elips(_ganjil(pr * 0.4)))
    if cv2.countNonZero(inner) < 20:
        inner = reg
    hsv_sub = np.ascontiguousarray(hsv[y0:y1, x0:x1])
    rata_hsv = cv2.mean(hsv_sub, mask=inner)

    # Skor is_gold: di Lab, kuningan punya b* positif tinggi & a* sedikit positif
    lab_sub = np.ascontiguousarray(lab_cie[y0:y1, x0:x1])
    rata_lab = cv2.mean(lab_sub, mask=inner)
    # rata_lab: (L_cie_scaled, a_cie, b_cie, 0) -- tapi ingat lab_cie sudah diskala
    lab_b = rata_lab[2]  # b* channel (positif = kuning)
    lab_a = rata_lab[1]  # a* channel

    # Heuristik is_gold: saturasi HSV tinggi + hue di zona kuning-oranye + b* Lab positif
    sat_val = rata_hsv[1]
    hue_val = rata_hsv[0]
    is_gold_score = 0.0
    if sat_val > 35 and 12 <= hue_val <= 40 and lab_b > 5:
        is_gold_score = min(1.0, (sat_val - 35) / 80.0 + (lab_b - 5) / 30.0)
    is_gold_score = round(float(np.clip(is_gold_score, 0.0, 1.0)), 3)

    return {
        "cx": ex + x0,
        "cy": ey + y0,
        "rx": rx,
        "ry": ry,
        "angle": sudut,
        "aspect": ry / (rx + 1e-6),
        "soliditas": soliditas,
        "luas": luas,
        "sat": rata_hsv[1],
        "hue": rata_hsv[0],
        "val": rata_hsv[2],
        "lab_L": rata_lab[0],
        "lab_a": lab_a,
        "lab_b": lab_b,
        "is_gold": is_gold_score,
        "radius_px": float((rx + ry) / 2.0),
    }


# ========================================================================================
# KLASIFIKASI KOIN RUPIAH
# ========================================================================================
def klasifikasi_koin_rupiah(koin_list):
    """
    Klasifikasi koin berdasarkan ukuran relatif dan skor warna logam.

    Diameter resmi (mm):
      Rp 100 = 23.0  (perak, terkecil)
      Rp 200 = 25.0  (perak)
      Rp 500 Kuningan = 24.0  (kuning emas)
      Rp 500 Perak = 27.0  (perak, terbesar)
      Rp 1000 = 24.1  (perak tebal)

    Rasio ukuran relatif terhadap Rp 500 Perak (27mm):
      Rp 100:   23.0/27.0 = 0.852
      Rp 200:   25.0/27.0 = 0.926
      Rp 500K:  24.0/27.0 = 0.889
      Rp 1000:  24.1/27.0 = 0.893
      Rp 500P:  27.0/27.0 = 1.000
    """
    if not koin_list:
        return []

    # Ambil koin bulat saja untuk referensi ukuran
    koin_bulat = [c for c in koin_list if c["aspect"] >= 0.76]
    if not koin_bulat:
        koin_bulat = koin_list

    # Radius terbesar diasumsikan Rp 500 Perak (27mm) sebagai referensi
    max_r = max(c["radius_px"] for c in koin_bulat)

    hasil = []
    for c in koin_list:
        if c["aspect"] < 0.76:
            hasil.append("UNCLASSIFIED")
            continue

        rel = c["radius_px"] / max(max_r, 1e-6)
        is_gold = c.get("is_gold", 0.0)

        # Koin kuningan (Rp 500 Kuningan): deteksi berdasarkan warna
        if is_gold >= 0.35:
            hasil.append("IDR_500_KUNING")
            continue

        # Klasifikasi berdasarkan ukuran relatif (koin perak)
        if rel >= 0.96:
            # Terbesar: Rp 500 Perak
            hasil.append("IDR_500_PERAK")
        elif rel >= 0.90:
            # Sedang-besar: Rp 200 (25mm) atau Rp 1000 (24.1mm)
            # Rp 1000 cenderung lebih tebal/gelap di gambar, tapi sulit dibedakan dari ukuran saja
            # Gunakan kecerahan: Rp 1000 (nikel) cenderung lebih gelap dari Rp 200 (aluminium)
            if c.get("lab_L", 70) < 62:
                hasil.append("IDR_1000")
            else:
                hasil.append("IDR_200")
        elif rel >= 0.86:
            # Zona ambang Rp 200 / Rp 1000
            if c.get("lab_L", 70) < 62:
                hasil.append("IDR_1000")
            else:
                hasil.append("IDR_200")
        else:
            # Terkecil: Rp 100
            hasil.append("IDR_100")

    return hasil


# ========================================================================================
# DETEKSI OPENCV ADAPTIF (PIPELINE UTAMA)
# ========================================================================================
def deteksi_opencv_adaptif(img, target_w=800, target_h=520, sensitivitas=55,
                           use_clahe=False, remove_bg=False):
    engine = "OpenCV Lab-Alas + Distance Transform + Watershed + fitEllipse"
    orig_h, orig_w = img.shape[:2]
    if min(orig_h, orig_w) < 40:
        return _laporan_kosong(engine), img.copy()

    sensitivitas = int(max(0, min(100, sensitivitas)))

    # --- Preprocessing opsional -------------------------------------------------------
    rembg_used = False
    if remove_bg:
        img, rembg_used = _apply_rembg(img)

    if use_clahe:
        img = _apply_clahe(img)

    # --- Skala proses (isotropik) -----------------------------------------------------
    skala = min(PANJANG_PROSES / float(max(orig_h, orig_w)), 2.0)
    proc_w = max(32, int(round(orig_w * skala)))
    proc_h = max(32, int(round(orig_h * skala)))
    interp = cv2.INTER_AREA if skala < 1.0 else cv2.INTER_LINEAR
    kecil = cv2.resize(img, (proc_w, proc_h), interpolation=interp)
    k_blur = _ganjil(min(proc_w, proc_h) * 0.01, minimum=5)
    buram = cv2.GaussianBlur(kecil, (k_blur, k_blur), 0)

    lab = konversi_lab_cie(buram)
    hsv = cv2.cvtColor(buram, cv2.COLOR_BGR2HSV)

    # --- 1-3. Alas, foreground, pembersihan ------------------------------------------
    model_latar = estimasi_model_latar(lab)
    fg, _ = buat_mask_foreground(lab, model_latar, sensitivitas)
    solid = bersihkan_mask(fg)

    # === Pipeline images: mask_foreground & morfologi ===
    pipeline_images = {}
    pipeline_images["mask_foreground"] = _encode_pipeline_image(fg)
    pipeline_images["morfologi"] = _encode_pipeline_image(solid)

    # --- 4. Distance Transform + puncak ----------------------------------------------
    dist = hitung_distance_transform(solid)
    min_r = max(6.0, 0.016 * min(proc_h, proc_w))
    puncak = cari_puncak_koin(dist, min_r)

    # === Pipeline image: distance_transform (heatmap JET/TURBO) ===
    dist_vis = None
    if dist.max() > 0:
        dist_norm = np.clip(dist / dist.max() * 255, 0, 255).astype(np.uint8)
        try:
            dist_vis = cv2.applyColorMap(dist_norm, cv2.COLORMAP_TURBO)
        except Exception:
            dist_vis = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    pipeline_images["distance_transform"] = _encode_pipeline_image(dist_vis)

    if not puncak:
        result = _laporan_kosong(engine)
        result["pipeline_images"] = pipeline_images
        result["rembg_used"] = rembg_used
        return result, img.copy()

    r_ref = float(np.percentile([p[2] for p in puncak], 90))
    r_min_final = max(min_r * 0.8, 0.28 * r_ref)

    # --- 5. Watershed + fitEllipse ----------------------------------------------------
    penanda, bg_id = segmentasi_watershed(buram, solid, puncak)

    kandidat_mentah = []
    for i, p in enumerate(puncak):
        hasil = analisis_wilayah(penanda, solid, hsv, lab, p, i + 1, bg_id)
        if hasil is None:
            continue
        if hasil["soliditas"] < 0.75:      # bentuk cekung = celah antar koin / bukan koin
            continue
        if hasil["aspect"] < 0.38:         # terlalu lonjong untuk koin
            continue
        if math.sqrt(hasil["rx"] * hasil["ry"]) < r_min_final:
            continue
        kandidat_mentah.append(hasil)

    # --- 6. Dedup: 1 koin = 1 elips ---------------------------------------------------
    kandidat_mentah.sort(key=lambda c: -c["luas"])
    diterima = []
    for c in kandidat_mentah:
        if any(fraksi_tumpang_tindih(c, d) > 0.30 for d in diterima):
            continue
        diterima.append(c)

    if not diterima:
        result = _laporan_kosong(engine)
        result["pipeline_images"] = pipeline_images
        result["rembg_used"] = rembg_used
        return result, img.copy()

    # --- 7. Konversi koordinat: proses -> asli (untuk anotasi) -> kanvas (untuk frontend)
    fx = orig_w / float(proc_w)
    fy = orig_h / float(proc_h)
    sx_kanvas = target_w / float(proc_w)
    sy_kanvas = target_h / float(proc_h)

    koin = []
    for c in diterima:
        cx_o, cy_o, rx_o, ry_o, ang_o = elips_ke_kanvas(c["cx"], c["cy"], c["rx"], c["ry"], c["angle"], fx, fy)
        cx_k, cy_k, rx_k, ry_k, ang_k = elips_ke_kanvas(c["cx"], c["cy"], c["rx"], c["ry"], c["angle"],
                                                        sx_kanvas, sy_kanvas)
        koin.append({
            "cx": cx_o, "cy": cy_o, "rx": rx_o, "ry": ry_o, "angle": ang_o,
            "aspect": c["aspect"], "sat": c["sat"], "hue": c["hue"],
            "val": c.get("val", 0), "is_gold": c.get("is_gold", 0.0),
            "radius_px": c.get("radius_px", 0.0),
            "lab_L": c.get("lab_L", 0), "lab_a": c.get("lab_a", 0), "lab_b": c.get("lab_b", 0),
            "k_cx": cx_k, "k_cy": cy_k, "k_rx": rx_k, "k_ry": ry_k, "k_ang": ang_k,
        })
    koin = urutkan_baca(koin)

    # --- 8. Klasifikasi Rupiah + laporan -----------------------------------------------
    kode_list = klasifikasi_koin_rupiah(koin)

    # Visualisasi watershed_final
    ws_vis = buram.copy()
    # Gambar batas watershed (garis -1)
    ws_vis[penanda == -1] = [0, 0, 255]

    annotated = img.copy()
    tebal = max(2, int(round(max(orig_h, orig_w) * 0.003)))
    skala_font = max(0.45, min(1.3, max(orig_h, orig_w) / 1600.0))

    total_koin = 0
    terklasifikasi = 0
    tidak_terklasifikasi = 0
    total_estimasi_idr = 0.0
    rincian_jenis = Counter()
    daftar_koin_kanvas = []

    for idx, (c, kode_final) in enumerate(zip(koin, kode_list)):
        total_koin += 1

        if kode_final in KATALOG_KOIN:
            terklasifikasi += 1
            info = KATALOG_KOIN[kode_final]
            total_estimasi_idr += info["nominal"]
            rincian_jenis[info["label"]] += 1
            warna = (16, 185, 129)
            label = f"#{total_koin} {info['label']}"
        else:
            tidak_terklasifikasi += 1
            warna = (11, 158, 245)
            label = f"#{total_koin} Unclassified"

        # Radius rata-rata di koordinat asli (piksel)
        radius_px_orig = float((c["rx"] + c["ry"]) / 2.0)

        daftar_koin_kanvas.append({
            "x": int(round(c["k_cx"])),
            "y": int(round(c["k_cy"])),
            "rx": max(6, int(round(c["k_rx"]))),
            "ry": max(5, int(round(c["k_ry"]))),
            "angle": round(c["k_ang"], 4),
            "code": kode_final,
            "radius_px": round(radius_px_orig, 2),
            "is_gold": c.get("is_gold", 0.0),
        })

        # Gambar elips pada gambar visualisasi watershed
        cv2.ellipse(
            ws_vis,
            (int(round(c["cx"] / fx * skala)), int(round(c["cy"] / fy * skala))),
            (max(2, int(round(c["rx"] / fx * skala))), max(2, int(round(c["ry"] / fy * skala)))),
            math.degrees(c["angle"]), 0, 360, (0, 255, 0), 2
        )

        # Gambar elips pada annotated (gambar asli)
        cv2.ellipse(
            annotated,
            (int(round(c["cx"])), int(round(c["cy"]))),
            (max(3, int(round(c["rx"]))), max(3, int(round(c["ry"])))),
            math.degrees(c["angle"]), 0, 360, warna, tebal
        )
        cv2.putText(
            annotated, label,
            (int(round(c["cx"] - c["rx"] * 0.5)), max(20, int(round(c["cy"])))),
            cv2.FONT_HERSHEY_SIMPLEX, skala_font, warna, max(1, tebal - 1)
        )

    pipeline_images["watershed_final"] = _encode_pipeline_image(ws_vis)

    laporan = {
        "total_semua_koin": total_koin,
        "berhasil_diklasifikasi": terklasifikasi,
        "tidak_terklasifikasi": tidak_terklasifikasi,
        "total_estimasi_idr": round(total_estimasi_idr),
        "engine_digunakan": engine,
        "rincian_per_jenis": dict(rincian_jenis),
        "daftar_koin": daftar_koin_kanvas,
        "pipeline_images": pipeline_images,
        "rembg_used": rembg_used,
    }
    return laporan, annotated


# ========================================================================================
# PEMROSES UTAMA (100% OpenCV)
# ========================================================================================
def proses_array_gambar(img, target_w=800, target_h=520, sensitivitas=55,
                        use_clahe=False, remove_bg=False):
    return deteksi_opencv_adaptif(
        img, target_w=target_w, target_h=target_h,
        sensitivitas=sensitivitas, use_clahe=use_clahe, remove_bg=remove_bg
    )


# ========================================================================================
# FLASK SERVER
# ========================================================================================
if FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)

    @app.after_request
    def disable_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.route("/")
    def index():
        resp = make_response(send_from_directory(".", "aplikasi_deteksi_koin_v10.html"))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    @app.route("/api/health", methods=["GET"])
    def health_endpoint():
        return jsonify({
            "status": "ok",
            "engine": "OpenCV (100% PCD)",
            "rembg_available": REMBG_AVAILABLE,
            "pecahan_rupiah": list(KATALOG_KOIN.keys()),
        })

    @app.route("/api/detect", methods=["POST"])
    def detect_endpoint():
        img = None
        target_w = 800
        target_h = 520
        sensitivitas = 55
        use_clahe = False
        remove_bg = False

        if "image" in request.files:
            file_storage = request.files["image"]
            img_bytes = file_storage.read()
            np_arr = np.frombuffer(img_bytes, np.uint8)
            # Gunakan IMREAD_UNCHANGED untuk mendukung PNG transparan
            img_raw = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            img = _handle_alpha_channel(img_raw)
            target_w = _ke_int(request.form.get("canvas_w", 800), 800)
            target_h = _ke_int(request.form.get("canvas_h", 520), 520)
            sensitivitas = _ke_int(request.form.get("sensitivitas", 55), 55)
            use_clahe = _ke_bool(request.form.get("use_clahe", False))
            remove_bg = _ke_bool(request.form.get("remove_bg", False))
        else:
            payload = request.get_json(silent=True) or {}
            img_b64 = payload.get("image_base64", "")
            target_w = _ke_int(payload.get("canvas_w", 800), 800)
            target_h = _ke_int(payload.get("canvas_h", 520), 520)
            sensitivitas = _ke_int(payload.get("sensitivitas", 55), 55)
            use_clahe = _ke_bool(payload.get("use_clahe", False))
            remove_bg = _ke_bool(payload.get("remove_bg", False))
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            if img_b64:
                try:
                    img_bytes = base64.b64decode(img_b64)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    img_raw = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    img = _handle_alpha_channel(img_raw)
                except Exception:
                    img = None

        if img is None:
            return jsonify({"status": "error", "message": "Gagal membaca file gambar"}), 400

        # Pastikan 3-channel BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        target_w = max(1, target_w)
        target_h = max(1, target_h)
        sensitivitas = max(0, min(100, sensitivitas))

        try:
            laporan, _ = proses_array_gambar(
                img, target_w=target_w, target_h=target_h,
                sensitivitas=sensitivitas, use_clahe=use_clahe, remove_bg=remove_bg
            )
        except Exception as e:
            return jsonify({"status": "error", "message": f"Gagal memproses gambar: {e}"}), 500

        laporan["rembg_available"] = REMBG_AVAILABLE
        return jsonify({"status": "ok", "laporan": laporan})

if __name__ == "__main__":
    if FLASK_AVAILABLE:
        print("=" * 65)
        print("  SISTEM DETEKSI KOIN RUPIAH INDONESIA")
        print("  100% Murni Pengolahan Citra Digital (OpenCV)")
        print("  Pecahan: Rp 100, Rp 200, Rp 500 (Perak/Kuningan), Rp 1000")
        print(f"  rembg tersedia: {REMBG_AVAILABLE}")
        print("=" * 65)
        print("[SERVER] Menjalankan di http://127.0.0.1:5000")
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    else:
        print("[ERROR] Flask / flask-cors belum terpasang: pip install flask flask-cors")
