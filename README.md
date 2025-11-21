# Andalens-TensorRT-Deepface(Retinaface and Facenet512)

Proyek ini mengoptimasi model **FaceNet** dan **RetinaFace** menggunakan **TensorFlow-TensorRT (TF-TRT)** untuk mempercepat inferensi deteksi dan embedding wajah pada GPU NVIDIA.

---

## 📋 Daftar Isi

- [Instalasi](#-instalasi)
- [Struktur Proyek](#-struktur-proyek)
- [Cara Penggunaan](#-cara-penggunaan)
- [Hasil Benchmark](#-hasil-benchmark)
- [Troubleshooting](#-troubleshooting)

---

## 📦 Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd <project-folder>
```

### 2. Setup Docker

**Di Windows (WSL2):**
```bash
# Install Docker Desktop + enable WSL2 backend
# Install NVIDIA Driver di Windows (bukan di WSL!)
# Restart PC

# Cek GPU terdeteksi di WSL
wsl
nvidia-smi
```

### 3. Build & Run Container
```bash
# Build Docker image
docker-compose build

# Start container
docker-compose up -d

# Lihat logs untuk dapat token JupyterLab
docker-compose logs jupyter-cuda
```

### 4. Akses JupyterLab
Buka browser dan akses:
```
http://localhost:8888
```
Masukkan token yang muncul di logs.

---

## 📁 Struktur Proyek

Untuk folder dan file yang tidak ada, bisa cari di: https://drive.google.com/drive/folders/1vapX9ea_K-7_ny9qGTFnEMmIyxrJqqFl?usp=sharing

```
.
├── Dockerfile                      # CUDA 11.8 + Python 3.11 + TensorFlow 2.14
├── docker-compose.yml              # Docker service config
├── TFRT.ipynb                      # Notebook utama untuk konversi & benchmark
├── facenetlib/                     # Library FaceNet
│   ├── Facenet_standalone.py
│   ├── facenet512_weights.h5
│   ├── preprocessing.py
│   └── image_utils.py
├── retina.py                       # RetinaFace detection API
├── retinafacelib/                  # RetinaFace core library
│   ├── preprocess.py
│   └── postprocess.py
├── saved_model/                    # Model original (SavedModel format)
│   ├── facenet512/
│   └── retinaface_saved_model/
├── facenet512_saved_model_TFTRT_FP16/    # Model TensorRT FP16 (hasil konversi)
├── retinaface_saved_model_TFTRT_FP16/
├── embedding/                      # Hasil embedding JSON
└── anjay/                          # Folder contoh gambar input
```

---

## 🎯 Cara Penggunaan

### A. Konversi Model ke TensorRT

Buka `TFRT.ipynb` di JupyterLab dan jalankan cell berikut:

#### 1️⃣ Convert FaceNet
```python
# Cell: Convert TensorRT (Facenet)
print('Converting to TF-TRT FP16...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16
)
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir=FACENET_TRT_DIR, 
   conversion_params=conversion_params
)
converter.convert()
converter.save(output_saved_model_dir='facenet512_saved_model_TFTRT_FP16')
```

#### 2️⃣ Convert RetinaFace
```python
# Cell: Convert TensorRT (Retinaface)
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=RETINA_TRT_DIR,
    conversion_params=conversion_params
)
converter.convert()
converter.save(output_saved_model_dir="retinaface_saved_model_TFTRT_FP16")
```

### B. Batch Processing

#### Tanpa TensorRT (Baseline)
```python
# Cell: Gapake TensorRT (Facenet)
results = batch_represent(img_files)
# Output: embedding/embeddings_batchf.json
```

#### Dengan TensorRT FP16
```python
# Cell: Pake Facenet TensorRT
results = batch_representfp16(img_files)
# Output: embedding/embeddings_fp16.json
```

---

## 📊 Hasil Benchmark

### FaceNet Embedding (1 gambar, 100 iterasi)

| Model | Avg Time/Inference | Speedup |
|-------|-------------------|---------|
| Original TF | 0.0091 sec | 1x |
| TF-TRT FP16 | 0.0020 sec | **4.5x faster** 🚀 |

### RetinaFace Detection (1 gambar, 50 iterasi)

| Model | Avg Time/Inference |
|-------|-------------------|
| TF-TRT FP16 | 0.0218 sec |

### Batch Processing (beberapa wajah)

| Pipeline | Total Time | Breakdown |
|----------|-----------|-----------|
| **Tanpa TensorRT** | 56.54 sec | Load: 11.91s, Detect: 36.86s, Embed: 7.53s |
| **Dengan TensorRT (FaceNet only)** | 59.51 sec | Load: 18.02s, Detect: 37.07s, Embed: 3.45s |

> **Note**: Speedup paling terasa di embedding stage (7.53s → 3.45s = 2.2x faster)

---

## 📝 Catatan Penting

1. **TensorRT Engine Caching**: Pertama kali inference akan lambat (build engine). Run berikutnya jauh lebih cepat.
2. **FP16 Precision**: Kehilangan akurasi ~0.1% dibanding FP32, tapi 2-4x lebih cepat.
3. **GPU Utilization**: Monitor dengan `nvidia-smi` saat benchmark untuk validasi GPU usage.

---

## 🐛 Known Issues

- RetinaFace TensorRT masih belum fully optimized (hanya 83.87% ops converted)
- Batch inference RetinaFace butuh implementasi custom (saat ini loop per gambar)

### Warmup dengan input stabil 640x640
```
[INFO] Warming up RetinaFace...

[INFO] Benchmarking RetinaFace TF-TRT FP16...
[RESULT] RetinaFace TF-TRT FP16 Avg Time: 0.0085 sec per inference 
```
TensorRT punya aturan main
 - Engine hanya valid untuk satu kombinasi input shape tertentu.
 - Kalau pertama kali kamu kasih 640x640, engine akan dikompilasi untuk itu. (saat build)
 - Kalau kemudian kamu kasih 1920x1080, TensorRT harus bikin engine baru.
 - Hal ini tidak menjadi masalah pada Facenet TensorRT karena sudah dipreprocess sedemikian rupa sehingga input pasti adalah (1,160,160,3)
 - Menjadi masalah pada retinaface karena input foto yang kita gunakan bisa banyak ukuran
---


