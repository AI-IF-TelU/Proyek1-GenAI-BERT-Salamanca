# IndoMiniBERT

Generasi judul berita bahasa Indonesia dengan kontrol gaya (clickbait / non-clickbait) memakai
BERT Masked Language Model kecil yang dibangun dari nol dengan PyTorch. Dataset: CLICK-ID
(William & Sari, 2020).

| Step | Pemilik | Status | Folder / file |
|---|---|---|---|
| 1. Persiapan data dan tokenizer | Anggota 1 | ✅ selesai | `config.py`, `src/data/`, `artifacts/tokenizer/`, `data/splits/` |
| 2. Arsitektur BERT | Anggota 2 | belum | `src/model.py` |
| 3. Masking dan pretraining MLM | Anggota 3 | belum | `src/collator.py`, `src/pretrain.py` |
| 4. MLM bergaya dan classifier | Anggota 4 | belum | `src/finetune_style.py`, `src/classifier.py` |
| 5. Generasi, evaluasi, dokumentasi | Anggota 5 | belum | `src/generate.py`, `src/evaluate.py` |

Nama file untuk Step 2 sampai 5 hanya saran, silakan disesuaikan. Yang penting semua ukuran dan
hyperparameter diambil dari `config.py`.

---

## 1. Setup di device masing-masing

### Laptop / PC

```bash
git clone https://github.com/AI-IF-TelU/Proyek1-GenAI-BERT-Salamanca.git
cd Proyek1-GenAI-BERT-Salamanca
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Kalau punya GPU NVIDIA, install PyTorch versi CUDA dari https://pytorch.org/get-started/locally/
sebelum `pip install -r requirements.txt`.

### Google Colab / Kaggle Notebook

```python
!git clone https://github.com/AI-IF-TelU/Proyek1-GenAI-BERT-Salamanca.git
%cd Proyek1-GenAI-BERT-Salamanca
!pip install -q -r requirements.txt
```

## 2. Download dataset

Dataset **tidak** ada di repo karena ukurannya sekitar 250 MB. Unduh dari Kaggle
([andikawilliam/clickid](https://www.kaggle.com/datasets/andikawilliam/clickid)), lalu ekstrak
**isinya** ke folder `click_id_dataset/` di root repo sampai strukturnya seperti ini:

```
click_id_dataset/
├── annotated/combined/csv/main.csv
├── raw/xlsx/*.xlsx
└── raw/csv/*.csv
```

Download manual: klik **Download** di halaman Kaggle, lalu ekstrak zip-nya ke `click_id_dataset/`.

Download lewat Kaggle CLI (Colab/Kaggle, butuh `kaggle.json` dari Settings > API di akun Kaggle):

```bash
pip install -q kaggle
kaggle datasets download -d andikawilliam/clickid
unzip -q clickid.zip -d click_id_dataset
```

Di Windows PowerShell: `Expand-Archive clickid.zip -DestinationPath click_id_dataset`.

## 3. Bikin ulang korpus pretraining (wajib sekali di tiap device)

Isi yang **sudah ada di repo**, jadi tidak perlu dibuat ulang:

- `data/splits/titles_{train,val,test}.csv`: split judul berlabel 80/10/10 (seed 42)
- `artifacts/tokenizer/`: tokenizer WordPiece 16.000 token
- `reports/*.json`: statistik dataset untuk laporan

Isi yang **tidak ada di repo** karena terlalu besar: `data/processed/pretrain_{train,val}.txt`.
Buat dengan perintah ini (sekitar 3–5 menit, kebanyakan untuk membaca file XLSX):

```bash
python -m src.data.prepare
```

Script ini memakai ulang split yang sudah ada di repo, jadi hasilnya sama di semua device.
**Jangan** jalankan `python -m src.data.tokenizer` atau `--resplit`. Kalau tokenizer dilatih
ulang, ID token berubah dan checkpoint antar anggota jadi tidak cocok.

Cek bahwa semuanya sudah siap:

```bash
python -m tests.test_step1_data
```

Semua tes harus `PASS`.

## 4. Cara memakai output Step 1

### Token khusus (ID tetap, dari `config.py`)

| Token | ID | Konstanta |
|---|---|---|
| `[PAD]` | 0 | `PAD_ID` |
| `[UNK]` | 1 | `UNK_ID` |
| `[CLS]` | 2 | `CLS_ID` |
| `[SEP]` | 3 | `SEP_ID` |
| `[MASK]` | 4 | `MASK_ID` |
| `[CB]` | 5 | `CB_ID` (gaya clickbait) |
| `[NCB]` | 6 | `NCB_ID` (gaya non-clickbait) |

Token yang **tidak boleh di-mask** (untuk collator Step 3 dan 4): `input_ids <= NCB_ID`.

### Tokenizer

```python
from src.data.tokenizer import load_tokenizer
tok = load_tokenizer()                      # objek tokenizers.Tokenizer
tok.get_vocab_size()                        # 16000, sama dengan model_cfg.vocab_size
tok.encode("presiden resmikan jalan tol", add_special_tokens=False).ids
tok.decode([2, 5, 123, 456, 3], skip_special_tokens=False)
tok.token_to_id("[MASK]")                   # 4
```

Semua teks sudah **lowercase**. Input teks baru (misalnya di demo) sebaiknya dibersihkan dulu dengan
`src.data.cleaning.clean_title(...)`.

### DataLoader pretraining (Step 3)

```python
from src.data.dataset import build_pretrain_loader
train_loader = build_pretrain_loader("train", batch_size=128)
val_loader   = build_pretrain_loader("val",   batch_size=128)

batch = next(iter(train_loader))
batch["input_ids"]       # LongTensor (B, 128)  [CLS] kalimat-kalimat satu artikel [SEP] [PAD]...
batch["attention_mask"]  # LongTensor (B, 128)  1 = token asli, 0 = padding
```

- Belum di-mask. Dynamic masking 15% (80/10/10) dikerjakan collator di Step 3.
- Kalimat dari satu artikel dipadatkan sampai 126 token per blok, tanpa melewati batas artikel.
- Pemanggilan pertama melakukan tokenisasi lalu menyimpan cache
  `data/processed/pretrain_*_blocks_128.pt`. Pemanggilan berikutnya langsung memuat cache.
- Kalau mau pakai `num_workers > 0` di Windows, kode training harus ada di dalam
  `if __name__ == "__main__":`.

### DataLoader judul berlabel (Step 4 dan 5)

```python
from src.data.dataset import build_title_loader

# MLM bergaya: [CLS] [CB]/[NCB] judul [SEP]  (token gaya ada di posisi 1, jangan di-mask)
style_loader = build_title_loader("train", batch_size=64, with_style=True)

# Classifier clickbait: [CLS] judul [SEP]  (tanpa token gaya)
clf_loader = build_title_loader("train", batch_size=64, with_style=False)

batch = next(iter(style_loader))
batch["input_ids"]       # (B, 48)
batch["attention_mask"]  # (B, 48)
batch["labels"]          # (B,)  1 = clickbait, 0 = non-clickbait
```

Split yang tersedia: `"train"`, `"val"`, `"test"`. Split `test` hanya dipakai untuk evaluasi akhir.

### Data untuk Step 5

- Distribusi panjang judul (dalam token) ada di `reports/tokenizer_stats.json` →
  `title_length_histogram` (index = jumlah token). Pakai ini untuk sampling panjang judul
  saat generasi mask-predict.
- Untuk metrik kebaruan, bandingkan hasil generasi dengan kolom `text` di
  `data/splits/titles_train.csv`.

## 5. Struktur repo

```
.
├── config.py                  # SATU sumber konfigurasi (path, token, hyperparameter semua step)
├── requirements.txt
├── src/
│   └── data/
│       ├── cleaning.py        # pembersihan teks + pemecahan kalimat
│       ├── prepare.py         # load, split, anti-leakage, tulis korpus
│       ├── tokenizer.py       # training & loading WordPiece
│       └── dataset.py         # PretrainDataset, TitleDataset, builder DataLoader
├── tests/test_step1_data.py   # cek kriteria selesai Step 1
├── artifacts/tokenizer/       # tokenizer.json, vocab.txt
├── data/splits/               # titles_train/val/test.csv
├── data/processed/            # (di-gitignore) korpus pretraining + cache
├── reports/                   # data_stats.json, tokenizer_stats.json
├── docs/llm_log/              # Log Interaksi LLM tiap anggota
└── click_id_dataset/          # (di-gitignore) dataset mentah dari Kaggle
```

## 6. Keputusan data yang perlu diketahui semua anggota

- **Sumber artikel raw = XLSX + CSV, dedup per URL.** File CSV di Kaggle terpotong (misalnya
  tribunnews hanya 1.500 dari 9.662 artikel), sedangkan XLSX lengkap (46.119 artikel, sesuai paper).
- **Pembersihan:** HTML, URL, email, mention/hashtag, dateline portal (`REPUBLIKA.CO.ID, JAKARTA --`,
  `Medan - `), tag editor detik (`(iwd/iwd) ...`), dan boilerplate (`loading...`, `Baca juga: ...`)
  dibuang. Teks diubah ke huruf kecil. Paragraf yang di-scrape tersambung koma (`.,Kalimat`) diperbaiki.
- **Catatan keterbatasan:** di data aslinya, nama entitas yang berupa hyperlink sudah hilang
  (`Presiden ketiga Indonesia , di Jakarta`). Ini tidak bisa dipulihkan, jadi sebagian kalimat
  korpus terasa "bolong". Bagian ini perlu dibahas di laporan.
- **Anti-leakage:** setiap kalimat korpus pretraining yang sama dengan atau memuat judul split test
  dibuang. Jumlahnya dicatat di `reports/data_stats.json` → `leakage`.
- **Judul duplikat** di data annotated di-dedup sebelum split. Judul yang labelnya bertentangan
  dibuang semua, supaya judul yang sama tidak muncul sekaligus di train dan test.

## 7. Aturan kerja tim

- **Branch sendiri, commit pakai akun sendiri.** Riwayat commit masuk penilaian.
  ```bash
  git checkout -b step2-model          # contoh untuk Anggota 2
  git add src/model.py
  git commit -m "Step 2: multi-head self-attention manual"
  git push -u origin step2-model
  ```
  Lalu buka Pull Request ke `main`.
- **Ambil update terbaru dari `main`** sebelum mulai kerja: `git pull origin main`.
- **Satu sumber konfigurasi.** Jangan hardcode ukuran tensor. Ambil dari `config.py`
  (`model_cfg.hidden_size`, `pretrain_cfg.max_len`, dan lain-lain). Kalau perlu nilai baru,
  tambahkan di bagian `config.py` milik step Anda.
- **Jangan commit** dataset, `data/processed/`, atau checkpoint (`*.pt`). Semuanya sudah ada di
  `.gitignore`. Bagikan checkpoint lewat Google Drive.
- **Log Interaksi LLM** dicatat masing-masing di `docs/llm_log/anggotaN.md` (template:
  `docs/llm_log/TEMPLATE.md`), lalu dikumpulkan Anggota 5 tiap minggu.
