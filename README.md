# 🥬 Lettuce Fresh Weight Prediction using ANN + PSO

> Prediksi berat segar selada (gram) dari gambar menggunakan **Artificial Neural Network (MLPRegressor)** yang dioptimalkan dengan **Particle Swarm Optimization (PSO)** dan ekstraksi fitur berbasis **classical image processing**.

---

## 📌 Latar Belakang

Monitoring berat tanaman selada secara manual membutuhkan waktu dan tenaga yang besar. Project ini membangun sistem prediksi berat selada berbasis computer vision dan machine learning yang dapat mengestimasi berat segar selada hanya dari satu foto, tanpa timbangan fisik.

---

## 🎯 Problem Statement

| | |
|---|---|
| **Input** | Gambar satu tanaman selada (JPG/JPEG) |
| **Output** | Prediksi berat segar dalam gram (termasuk akar) |
| **Metode** | Classical image processing + ANN (tanpa CNN / deep learning) |

---

## 📁 Struktur Project

```
lettuce-weight-prediction/
│
├── data/
│   ├── raw/
│   │   ├── images/                  # 100 gambar asli (S001.jpeg – S100.jpeg)
│   │   └── measurements_raw.csv     # Label berat (id;foto;bobot_segar_gram)
│   ├── augmented_images/            # 200 gambar (100 asli + 100 augmented) [auto]
│   ├── dataset.csv                  # Hasil ekstraksi 17 fitur + label [auto]
│   └── preprocessing_config.pkl     # Config preprocessing [auto]
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb       # Load, segmentasi, augmentasi
│   ├── 02_Feature_Extraction.ipynb       # Ekstraksi 17 fitur per gambar
│   ├── 03_Exploratory_Data_Analysis.ipynb # EDA, heatmap, distribusi
│   ├── 04_ANN_Model.ipynb                # Training ANN dasar (64-32-16)
│   ├── 05_PSO_Hyperparameter_Tuning.ipynb # PSO tuning 4D (baseline)
│   ├── 06_Model_Evaluation.ipynb         # Evaluasi 5-Fold CV, perbandingan
│   ├── 07_Improved_ANN_PSO.ipynb         # ⭐ Improved pipeline (model terbaik)
│   └── 08_Final_Pipeline.ipynb           # 🎯 Demo presentasi (load model terbaik)
│
├── output/
│   ├── models/                      # Semua file .pkl model [auto]
│   └── plots/                       # Semua visualisasi [auto]
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🗂️ Dataset

| Info | Detail |
|------|--------|
| Jumlah gambar asli | 100 (S001–S100) |
| Jumlah setelah augmentasi | 200 |
| Format gambar | JPEG |
| Label | `measurements_raw.csv` (separator `;`) |
| Range berat | 18 – 45 gram |
| Rata-rata berat | ~31.97 gram |
| **Definisi berat** | **Berat segar total = daun + batang + akar** |

### Format CSV Label
```
id_tanaman;foto;bobot_segar_gram
S001;S001.jpeg;27
S002;S002.jpeg;37
...
```

---

## 🔬 Segmentasi & Feature Extraction

### Segmentasi HSV – Daun + Akar

Label berat mencakup seluruh tanaman (daun + akar), sehingga segmentasi menggunakan **dua mask** yang digabungkan:

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Mask 1: Daun hijau
mask_g = cv2.inRange(hsv, np.array([20, 30,  30]),
                          np.array([95, 255, 255]))

# Mask 2: Akar gelap (V < 75, S < 80 → tanah/akar, bukan background putih)
mask_r = cv2.inRange(hsv, np.array([0,  0,   5]),
                          np.array([180, 80, 75]))

# Gabungkan
mask = cv2.bitwise_or(mask_g, mask_r)
```

**Alasan pemisahan background vs akar:**

| Objek | V (Value/Brightness) | S (Saturation) |
|-------|---------------------|----------------|
| Background putih/abu | 103 – 142 | 26 – 41 |
| **Akar (tanah gelap)** | **5 – 74** | **0 – 80** |
| Daun hijau | 32 – 194 | 25 – 255 |

> Kunci deteksi akar: **V < 75** (gelap) **AND S < 80** (tidak jenuh/hijau)

### 17 Fitur yang Diekstrak

#### Morfologi (5 fitur) — dari area daun + akar
| Fitur | Deskripsi |
|-------|-----------|
| `leaf_area` | Luas area segmentasi (px²) |
| `perimeter` | Keliling kontur |
| `aspect_ratio` | Lebar / Tinggi bounding box |
| `extent` | Rasio area / bounding box |
| `equivalent_diameter` | √(4A/π) |

#### Warna (7 fitur)
| Fitur | Deskripsi |
|-------|-----------|
| `mean_R`, `mean_G`, `mean_B` | Rata-rata kanal RGB |
| `mean_H`, `mean_S`, `mean_V` | Rata-rata kanal HSV |
| `std_G` | Standar deviasi kanal hijau |

#### Tekstur GLCM (5 fitur)
`contrast`, `energy`, `homogeneity`, `correlation`, `entropy`

---

## 🧠 Arsitektur ANN

```
Input Layer      →  N neuron  (N fitur terpilih)
     ↓ ReLU
Hidden Layer 1   →  n1 neuron  (dicari via PSO)
     ↓ ReLU
Hidden Layer 2   →  n2 neuron  (dicari via PSO)
     ↓ ReLU
Hidden Layer 3   →  n3 neuron  (opsional, dicari via PSO)
     ↓ Linear
Output Layer     →   1 neuron  (berat gram)
```

---

## 🐝 PSO Optimization

| | Baseline (NB05) | Improved (NB07) |
|---|---|---|
| Dimensi | 4 (n1, n2, lr, α) | **5 (n1, n2, n3, lr, α)** |
| Partikel | 15 | **25** |
| Iterasi | 30 | **50** |
| c1 / c2 | 0.5 / 0.3 | **1.5 / 1.5** |
| Inertia (w) | 0.9 | **0.7** |
| BOUNDS n1 | 16–128 | **8–256** |
| BOUNDS lr | 0.0001–0.05 | **0.0001–0.1** |
| BOUNDS alpha | 0.001–0.1 | **0.0001–0.5** |
| max_iter (eval) | 300 | **500** |
| Objective | RMSE 5-Fold CV | RMSE 5-Fold CV |

---

## 🚀 Cara Menjalankan

### Prerequisites
```bash
pip install -r requirements.txt
```

### Alur Notebook

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

| Notebook | Fungsi | Prasyarat |
|----------|--------|-----------|
| `01` | Load data, segmentasi (daun+akar), augmentasi | Data di `data/raw/` |
| `02` | Ekstrak 17 fitur → `dataset.csv` | NB01 selesai |
| `03` | EDA: distribusi, korelasi, outlier | NB02 selesai |
| `04` | Training ANN dasar | NB02 selesai |
| `05` | PSO tuning baseline (4D) | NB02 selesai |
| `06` | Evaluasi 5-Fold CV, perbandingan | NB04 + NB05 selesai |
| `07` | ⭐ Improved pipeline (model terbaik) | NB02 selesai |
| `08` | 🎯 Demo presentasi (load model terbaik) | **NB07 selesai** |

> **Shortcut presentasi:** Jalankan `01` → `02` → `07` → `08`

### Cek Root Detection (NB01)

Setelah mendefinisikan fungsi segmentasi, jalankan **Cell "Cek Root Detection"** di NB01 untuk memverifikasi akar terdeteksi sebelum lanjut ke augmentasi:

```
Status yang diharapkan:
  S014  18g   ✅ Terdeteksi
  S080  29g   ✅ Terdeteksi
  S023  35g   ✅ Terdeteksi
  S067  45g   ✅ Terdeteksi
```

---

## 📊 Metrik Evaluasi

| Metrik | Formula |
|--------|---------|
| **MAE** | Σ\|y − ŷ\| / n |
| **RMSE** | √(Σ(y − ŷ)² / n) |
| **R²** | 1 − SS_res/SS_tot |
| **MAPE** | Σ\|y − ŷ\|/\|y\| × 100% |

**Cross Validation:** `KFold(n_splits=5, shuffle=True, random_state=42)`

---

## 📈 Hasil Eksperimen

Evaluasi menggunakan **5-Fold Cross Validation**.

### Perbandingan Model (5-Fold CV, mean ± std)

| Metrik | ANN Base | ANN + PSO | **ANN Improved** ⭐ |
|--------|----------|-----------|---------------------|
| MAE (gram) | 4.6216 ± 1.060 | 4.0299 ± 1.138 | **2.6914 ± 0.194** |
| RMSE (gram) | 5.7213 ± 1.181 | 5.0872 ± 1.458 | **3.5640 ± 0.459** |
| R² | -0.0277 ± 0.363 | 0.1003 ± 0.615 | **0.5887 ± 0.165** |
| MAPE (%) | 15.51 ± 4.115 | 13.24 ± 3.585 | **8.94 ± 1.247** |

> Pipeline dengan **segmentasi daun + akar** (dual HSV mask + multi-contour selection) dan **PSO 5D yang dioptimalkan** (25 partikel, 50 iterasi, c1=c2=1.5).

### Peningkatan ANN Improved vs ANN+PSO Baseline

| Metrik | Δ | % |
|--------|---|---|
| MAE | ↓ 1.3385 gram | **+33.2%** |
| RMSE | ↓ 1.5232 gram | **+29.9%** |
| MAPE | ↓ 4.3072% | **+32.5%** |
| R² | ↑ +0.4884 | — |

---

## ⚠️ Keterbatasan

1. **Segmentasi akar tidak sempurna** — Warna akar bervariasi antar sampel (tergantung media tanam dan pencahayaan). Segmentasi HSV berbasis V < 75 mendeteksi sebagian besar akar gelap, namun tidak sempurna untuk semua kondisi. Ini merupakan limitasi inherent dari classical image processing.

2. **Dataset kecil** — 100 sampel asli (200 setelah augmentasi) membatasi generalisasi model.

3. **Augmentasi terbatas** — Augmentasi hanya geometric + photometric, tidak mencakup variasi background atau sudut pengambilan.

---

## 🔧 Environment

| Library | Versi |
|---------|-------|
| Python | 3.13 |
| OpenCV | 4.13.0 |
| scikit-learn | 1.8.0 |
| scikit-image | 0.26.0 |
| PySwarms | 1.3.0 |
| NumPy | 2.4.6 |
| Pandas | 3.0.3 |

---

## 📚 Referensi

_(Tambahkan minimal 5 referensi jurnal Q1–Q3, lihat `referensi_jurnal.md`)_

1. ...
2. ...
3. ...
4. ...
5. ...

---

## 👥 Tim

| Nama | NIM |
|------|-----|
| ... | ... |
| ... | ... |

**Deadline:** 2 Juni 11:59 WITA  
**Pengumpulan:** PDF laporan + PPT + Source Code (dalam 1 ZIP)

---

*Built with ❤️ using Python · OpenCV · scikit-learn · PySwarms*
