# Sistem Deteksi Koin Multi-Sudut & Tumpukan (v10.0)

Sistem deteksi jumlah, jenis, dan nilai mata uang koin dunia dari berbagai sudut pengambilan gambar serta berbagai kondisi susunan (beraturan, menara tumpukan samping, maupun tumpukan acak padat).

## Arsitektur Algoritma Tri-Mode Vision Engine (v10.0)

Sistem dirancang untuk mengatasi kelemahan deteksi lingkaran klasik ketika berhadapan dengan sudut kamera miring (koin berbentuk elips), pantulan cahaya logam (*specular highlight*), serta koin yang saling menumpuk:

1. **Exterior Background Flood-Fill (Penutup Lubang Internal):**
   Menyebarkan *flood-fill* hanya dari piksel latar belakang di tepi luar citra. Seluruh pantulan cahaya putih atau relief ukiran di dalam batas koin otomatis tertutup rapat menjadi satu piringan solid (*solid foreground mask*).
2. **Mode 1 — Koin Beraturan / Terpisah (`REGULAR`):**
   Menggunakan analisis komponen terhubung (*Connected Component Analysis*) di mana 1 blob utuh dikunci tepat menjadi 1 keping koin tanpa mencacah ukiran Garuda atau angka nominal di dalamnya.
3. **Mode 2 — Menara Tumpukan Samping (`TOWER_STACK`):**
   * Menggunakan proyeksi kepadatan vertikal yang dihaluskan (*Smoothed 1D Column Profile*) dengan jarak minimal antar sumbu menara >= 144 px untuk mencegah munculnya kolom bayangan di celah antar tumpukan.
   * Memasang 1 elips **Terklasifikasi (Hijau)** tepat pada piringan koin paling atas (*Top-Cap Lock*), lalu menghitung lekukan garis kepingan di bawahnya menggunakan *1D Vertical Gradient Peak Counter* sebagai koin **Tidak Terklasifikasi (Oranye)**.
   * Mendeteksi koin yang bersandar miring atau menghadap kamera menggunakan *Elliptical Rim Finder*.
4. **Mode 3 — Tumpukan Acak Padat (`IRREGULAR_PILE`):**
   Menerapkan penghalusan tekstur relief internal (*3x3 Spatial Smoothing*) diikuti operator Sobel dan validasi keseragaman busur 8-sektor mata angin (*8-Sector Rim Uniformity*). Koin lapisan atas dengan >= 6 sektor aktif diklasifikasikan nominalnya, sedangkan koin lapisan bawah yang tertutup sebagian (>= 4 sektor aktif) dihitung sebagai *Unclassified*.

## Cara Menjalankan Sistem Terintegrasi

1. Instal dependensi Python:
   ```bash
   pip install ultralytics opencv-python numpy flask flask-cors