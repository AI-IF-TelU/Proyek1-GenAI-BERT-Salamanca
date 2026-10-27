# Log Interaksi LLM: Anggota 2 (Step 2: Arsitektur BERT, Step 3: Masking dan Pretraining MLM)

LLM yang dipakai: Claude (Claude Code, VS Code)

---

## Sesi 1 · 2026-10-27 · Tahap: Inisiasi

**Prompt:**
> "https://github.com/AI-IF-TelU/Proyek1-GenAI-BERT-Salamanca. README sudah berisi panduan clone, cara
> download dataset, dan perintah python -m src.data.prepare yang perlu mereka jalankan sekali.
> baca semua file di repo tersebut lalu buatlah. Anggota 2 Arsitektur Model BERT (model.py dan hasil
> tes overfit) dan Masking dan Pretraining MLM (Checkpoint pretraining dan grafik loss)"
>
> (di tengah jalan) "wait apakah aku ga clone semua file di repo? tidak perlu atau kamu lupa?"
> "lanjutkan tugasnya"

**Temuan LLM sebelum menulis kode:**
- Repo sudah di-clone utuh. Yang tidak ada di repo (sengaja, karena di-`.gitignore`) hanya
  `click_id_dataset/` dan `data/processed/`, jadi pretraining asli belum bisa dijalankan di laptop.
- Laptop hanya punya PyTorch CPU tanpa GPU. Pretraining penuh harus dijalankan di Colab/Kaggle T4.
- Tes overfit tetap bisa dijalankan karena `data/splits/` dan tokenizer sudah ada di repo.

**Kode yang dihasilkan:**
- `src/model.py`: `BertEmbeddings`, `MultiHeadSelfAttention` manual (Q/K/V, scaled dot-product,
  padding mask, gabung head), `EncoderLayer` post-LN, MLM head dengan weight tying, classification
  head di `[CLS]`, inisialisasi normal(0, 0.02), `save_model` / `load_model`
- `src/overfit.py` + `tests/test_step2_model.py`: 9 tes (bentuk output, backward ke semua parameter,
  weight tying, padding diabaikan, atensi dua arah, jumlah parameter, simpan/muat, overfit)
- `src/collator.py`: dynamic masking 15% (80/10/10), token khusus `<= NCB_ID` tidak pernah di-mask
- `tests/test_step3_masking.py`: uji rasio masking dan token khusus
- `src/pretrain.py`: AdamW, linear warmup + decay, gradient clipping, AMP, checkpoint `last.pt` /
  `best.pt`, `--resume`, metrik loss/perplexity/top-1/top-5 per epoch, grafik loss
- `src/sweep.py`: 3 learning rate x 2 batch size, tabel dan grafik perbandingan
- `src/fill_mask.py`, `src/plots.py`, penambahan `OverfitConfig` dan field sweep di `config.py`

**Keputusan desain dari LLM:**
- Saat `labels` diberikan, logit MLM hanya dihitung di posisi yang di-mask (sekitar 15%), bukan
  128 x 16.000 logit per blok. Cara ini jauh lebih hemat memori GPU.
- Masking validasi memakai seed tetap supaya loss validasi antar-epoch bisa dibandingkan.
- Weight decay tidak diberikan pada bias dan LayerNorm.

**Debugging saat dijalankan:**
- `build_pretrain_loader` meneruskan `**kwargs` ke Dataset, bukan ke DataLoader, sehingga argumen
  `pin_memory` akan error. Argumen itu dihapus dari `pretrain.py`.
- Satu tes collator gagal (`IndexError`) karena posisi `[SEP]` di helper tes di-hardcode 100,
  padahal panjang batch tes 48. Helper diubah supaya posisi `[SEP]` mengikuti panjang batch.

**Yang saya ubah / tolak dari jawaban LLM, dan alasannya:**
- Menghentikan LLM sejenak untuk memastikan semua file repo sudah di-clone sebelum lanjut.
- (isi sendiri: perubahan lain yang Anda minta setelah membaca kode / hasil)

**Hasil setelah dijalankan:**
```
Step 2 - python -m tests.test_step2_model   : 9/9 tes lulus
  jumlah parameter                          : 7.436.930
    embeddings 4.129.280 | 4 encoder layer 3.159.040 | MLM head 82.304 | cls head 66.306
Tes overfit (8 judul, 15 token ter-mask, 300 step, lr 1e-3, dropout 0, CPU 42 detik)
  loss awal  MLM 9,4738 (teori ln 16000 = 9,68) | cls 0,6947 (teori ln 2 = 0,69)
  loss akhir MLM 0,0026                         | cls 0,00005
  akurasi akhir 100% / 100%                     -> LULUS

Step 3 - python -m tests.test_step3_masking : 5/5 tes lulus
  terpilih 14,93% | [MASK] 80,2% | acak 10,2% | tetap 9,6%
Uji asap pretrain.py, sweep.py, --resume, fill_mask.py dengan korpus sementara dari judul: berjalan
(korpus dan checkpoint uji asap sudah dihapus, bukan hasil resmi)
```

**Belum selesai (perlu GPU + dataset):** sweep dan pretraining penuh di Colab/Kaggle untuk
menghasilkan `checkpoints/pretrain/main/best.pt` dan `reports/pretrain/main/loss.png`.

**Pelajaran:**
- Loss awal yang mendekati ln(ukuran vocab) adalah cek cepat bahwa inisialisasi dan loss sudah benar.
- Tes "padding diabaikan" (isi token `[PAD]` diganti acak, output non-pad harus identik) langsung
  membuktikan padding mask di atensi bekerja.
