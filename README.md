# 🥬 Lettuce Fresh Weight Prediction using ANN + PSO

> Prediksi berat segar selada (gram) dari gambar menggunakan **Artificial Neural Network (MLPRegressor)** yang dioptimalkan dengan **Particle Swarm Optimization (PSO)** dan ekstraksi fitur berbasis **classical image processing**.

---

## 📌 Latar Belakang

Monitoring berat tanaman selada secara manual membutuhkan waktu dan tenaga yang besar. Project ini membangun sistem prediksi berat selada berbasis computer vision dan machine learning yang dapat mengestimasi berat segar selada hanya dari satu foto, tanpa timbangan fisik.

---

## 🎯 Problem Statement

|            |                                                              |
| ---------- | ------------------------------------------------------------ |
| **Input**  | Gambar satu tanaman selada (JPG/JPEG)                        |
| **Output** | Prediksi berat segar dalam gram                              |
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
│   ├── augmented_images/            # 200 gambar (100 asli + 100 augmented)
│   ├── dataset.csv                  # Hasil ekstraksi 17 fitur + label (auto)
│   └── preprocessing_config.pkl     # Config preprocessing (auto)
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb       # Load, segmentasi, augmentasi
│   ├── 02_Feature_Extraction.ipynb       # Ekstraksi 17 fitur per gambar
│   ├── 03_Exploratory_Data_Analysis.ipynb # EDA, heatmap, distribusi
│   ├── 04_ANN_Model.ipynb                # Training ANN dasar (64-32-16)
│   ├── 05_PSO_Hyperparameter_Tuning.ipynb # PSO tuning 4D (baseline)
│   ├── 06_Model_Evaluation.ipynb         # Evaluasi 5-Fold CV, perbandingan
│   ├── 07_Improved_ANN_PSO.ipynb         # ⭐ Improved pipeline (model terbaik)
│   └── 08_Final_Pipeline.ipynb           # Demo presentasi (load model terbaik)
│
├── output/
│   ├── models/
│   │   ├── ann_model.pkl                 # Model ANN dasar
│   │   ├── ann_model_pso_optimized.pkl   # Model ANN+PSO baseline
│   │   ├── ann_model_improved.pkl        # ⭐ Model terbaik (ANN Improved)
│   │   ├── scaler.pkl                    # Scaler untuk model dasar
│   │   ├── scaler_improved.pkl           # Scaler untuk model improved
│   │   └── improved_metadata.pkl         # Config model improved
│   └── plots/
│       ├── actual_vs_predicted.png
│       ├── training_curve.png
│       ├── residual_plot.png
│       ├── feature_importance.png
│       ├── correlation_heatmap.png
│       ├── cv_evaluation.png
│       ├── pso_convergence.png
│       ├── weight_distribution.png
│       ├── feature_distributions.png
│       ├── outlier_boxplots.png
│       ├── color_distribution.png
│       ├── segmentation_results.png
│       ├── sample_images.png
│       ├── improved_comparison.png
│       ├── improved_distribution.png
│       ├── improved_feature_correlation.png
│       ├── improved_actual_vs_predicted.png
│       ├── demo_predictions.png
│       └── final_dashboard.png
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🗂️ Dataset

| Info                      | Detail                                 |
| ------------------------- | -------------------------------------- |
| Jumlah gambar asli        | 100 (S001–S100)                        |
| Jumlah setelah augmentasi | 200                                    |
| Format gambar             | JPEG                                   |
| Label                     | `measurements_raw.csv` (separator `;`) |
| Range berat               | 18 – 45 gram                           |
| Rata-rata berat           | ~31.97 gram                            |

### Format CSV Label

```
id_tanaman;foto;bobot_segar_gram
S001;S001.jpeg;27
S002;S002.jpeg;37
...
```

---

## 🔬 Feature Extraction (17 Fitur)

Ekstraksi fitur menggunakan **classical image processing**.

### 1. Segmentasi HSV

```python
lower_green = np.array([20, 30, 30])
upper_green = np.array([95, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
# + Morphological: OPEN (3×3) + CLOSE (7×7)
```

### 2. Fitur Morfologi (5 fitur)

| Fitur                 | Deskripsi                           |
| --------------------- | ----------------------------------- |
| `leaf_area`           | Luas area daun (pixel²)             |
| `perimeter`           | Keliling kontur                     |
| `aspect_ratio`        | Lebar / Tinggi bounding box         |
| `extent`              | Rasio area daun / area bounding box |
| `equivalent_diameter` | √(4A/π)                             |

### 3. Fitur Warna (7 fitur)

| Fitur                        | Deskripsi                          |
| ---------------------------- | ---------------------------------- |
| `mean_R`, `mean_G`, `mean_B` | Rata-rata kanal RGB pada area daun |
| `mean_H`, `mean_S`, `mean_V` | Rata-rata kanal HSV pada area daun |
| `std_G`                      | Standar deviasi kanal hijau        |

### 4. Fitur Tekstur GLCM (5 fitur)

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

|             | Baseline (NB05)   | Improved (NB08)           |
| ----------- | ----------------- | ------------------------- |
| Dimensi     | 4 (n1, n2, lr, α) | **5 (n1, n2, n3, lr, α)** |
| Partikel    | 15                | **20**                    |
| Iterasi     | 30                | **40**                    |
| Alpha range | 0.0001–0.01       | **0.001–0.1**             |
| Objective   | RMSE 5-Fold CV    | RMSE 5-Fold CV            |

---

## 🚀 Cara Menjalankan

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image \
            opencv-python tqdm pyswarms scipy Pillow joblib ipykernel
```

### Alur Notebook

```
┌─────────────────────────────────────────────────────────────┐
│  PIPELINE UTAMA (Step-by-step)                              │
│                                                             │
│  01 → 02 → 03 → 04 → 05 → 06 → 07 → 08                    │
│                                                             │
│  01: Preprocessing & Augmentasi                             │
│  02: Ekstraksi 17 Fitur → dataset.csv                       │
│  03: EDA (distribusi, heatmap, outlier)                     │
│  04: Training ANN Dasar (64-32-16)                          │
│  05: PSO Baseline (4D hyperparameter search)                │
│  06: Evaluasi & perbandingan ANN vs ANN+PSO                 │
│  07: ⭐ Improved Pipeline (feature selection + PSO 5D)      │
│  08: 🎯 Demo Presentasi (load model terbaik dari NB07)      │
└─────────────────────────────────────────────────────────────┘
```

| Notebook | Fungsi                                  | Prasyarat           |
| -------- | --------------------------------------- | ------------------- |
| `01`     | Load data, segmentasi, augmentasi       | Data di `data/raw/` |
| `02`     | Ekstrak fitur → `dataset.csv`           | NB01 selesai        |
| `03`     | EDA: distribusi, korelasi, outlier      | NB02 selesai        |
| `04`     | Training ANN dasar                      | NB02 selesai        |
| `05`     | PSO tuning baseline (4D)                | NB02 selesai        |
| `06`     | Evaluasi 5-Fold CV, perbandingan        | NB04 + NB05 selesai |
| `07`     | ⭐ Improved pipeline (model terbaik)    | NB02 selesai        |
| `08`     | 🎯 Demo presentasi (load model terbaik) | **NB07 selesai**    |

> **Shortcut presentasi:** Jalankan `07` → kemudian `08`

---

## 📊 Metrik Evaluasi

| Metrik   | Formula                 |
| -------- | ----------------------- |
| **MAE**  | Σ\|y − ŷ\| / n          |
| **RMSE** | √(Σ(y − ŷ)² / n)        |
| **R²**   | 1 − SS_res/SS_tot       |
| **MAPE** | Σ\|y − ŷ\|/\|y\| × 100% |

**Cross Validation:** `KFold(n_splits=5, shuffle=True, random_state=42)`

---

## 📈 Hasil Eksperimen

Evaluasi menggunakan **5-Fold Cross Validation**.

### Perbandingan Model (5-Fold CV, mean ± std)

| Metrik      | ANN Base        | ANN + PSO      | **ANN Improved** ⭐ |
| ----------- | --------------- | -------------- | ------------------- |
| MAE (gram)  | 4.6216 ± 1.060  | 4.0299 ± 1.138 | **2.9695 ± 0.542**  |
| RMSE (gram) | 5.7213 ± 1.181  | 5.0872 ± 1.458 | **3.7975 ± 0.541**  |
| R²          | -0.0277 ± 0.363 | 0.1003 ± 0.615 | **0.5560 ± 0.103**  |
| MAPE (%)    | 15.51 ± 4.115   | 13.24 ± 3.585  | **9.66 ± 1.965**    |

### Peningkatan ANN Improved vs ANN+PSO Baseline

| Metrik              | Δ                 | %                   |
| ------------------- | ----------------- | ------------------- |
| MAE                 | ↓ 1.0604 gram     | **+26.3%**          |
| RMSE                | ↓ 1.2897 gram     | **+25.4%**          |
| MAPE                | ↓ 3.5852%         | **+27.1%**          |
| R²                  | ↑ +0.4557         | —                   |
| Stabilitas R² (std) | 0.615 → **0.103** | **6× lebih stabil** |

### Konfigurasi Model Terbaik (ANN Improved)

| Komponen          | Detail                                      |
| ----------------- | ------------------------------------------- |
| Feature selection | 10 fitur terbaik dari 17 (korelasi Pearson) |
| Target transform  | Log-transform jika skewness > 0.5 (auto)    |
| Scaler            | Auto-select: MinMax vs StandardScaler       |
| PSO dimensi       | 5D: (n1, n2, n3, lr, alpha)                 |
| Regularisasi      | Alpha 0.001–0.1 (lebih kuat)                |

---

## 🔧 Environment

| Library      | Versi  |
| ------------ | ------ |
| Python       | 3.13   |
| OpenCV       | 4.13.0 |
| scikit-learn | 1.8.0  |
| scikit-image | 0.26.0 |
| PySwarms     | 1.3.0  |
| NumPy        | 2.4.6  |
| Pandas       | 3.0.3  |

---

## 📚 Referensi

_(Tambahkan minimal 5 referensi jurnal Q1–Q3)_

1. ...
2. ...
3. ...
4. ...
5. ...

---

## 👥 Tim

| Nama                    | NIM           |
| ----------------------- | ------------- |
| Adrian Bintang Saputera | 2310817110006 |
| Harry Pratama Yunus     | 2310817210010 |

**Deadline:** 1 Juni 23:59 WITA
**Pengumpulan:** PDF laporan + PPT + Source Code (dalam 1 ZIP)

---
