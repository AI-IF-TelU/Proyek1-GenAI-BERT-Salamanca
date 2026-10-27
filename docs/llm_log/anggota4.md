# Log Interaksi LLM: Anggota 4 (Step 4: MLM Bergaya dan Classifier Clickbait)

LLM yang dipakai: Google Antigravity (Gemini / Claude via Antigravity Agent)

---

## Sesi 1 · 2026-10-27 · Tahap: Inisiasi

**Prompt:**
> "Sesuai proposal Proyek IndoMiniBERT, sekarang giliran saya mengimplementasikan Step 4 (MLM Bergaya dan Classifier Clickbait). 
> Input: checkpoint pretraining dari Step 3 dan data judul berlabel. 
> Output: Model generator bergaya dan model classifier clickbait.
> Tolong implementasikan fine-tuning MLM dengan token gaya di posisi 1 ([CB] / [NCB]) yang tidak boleh di-mask, 
> serta classifier clickbait 2 varian (pretrained vs random init) untuk ablation study."

**Temuan & Rancangan Awal:**
- Dataset judul berlabel dari Step 1 sudah tersedia di `data/splits/titles_{train,val,test}.csv` dengan pembagian stratified 80/10/10.
- Tokenizer memetakan token khusus: `[CLS]`=2, `[SEP]`=3, `[MASK]`=4, `[CB]`=5, `[NCB]`=6.
- Pada MLM bergaya, input berbentuk: `[CLS] [CB]/[NCB] judul [SEP]`. Token gaya berada di indeks 1, sehingga dynamic masking collator wajib melindungi posisi 0 dan 1 (`protected_positions=(0, 1)`).
- Classifier clickbait menggunakan representasi token `[CLS]` dari encoder BERT diikuti dropout dan linear head 2 kelas (clickbait vs non-clickbait) tanpa menyertakan token gaya pada input.
- Dua varian classifier dilatih dengan seed yang sama:
  1. `random_init`: seluruh bobot diinisialisasi acak.
  2. `pretrained`: encoder dimuat dari checkpoint `checkpoints/pretrain_mlm.pt`, sedangkan classification head diinisialisasi acak.

**Kode yang Dihasilkan:**
- `src/finetune_style.py`: training loop MLM bergaya dengan dynamic masking terlindungi, checkpointing berkala ke `checkpoints/style_mlm.pt`, dan pencatatan loss/perplexity/accuracy ke `reports/style_mlm_metrics.json`.
- `src/classifier.py`: training loop classifier untuk kedua varian (random vs pretrained), evaluasi metrik akurasi, macro F1, dan confusion matrix pada test split, serta output `reports/style_classifier_ablation.json` dan `reports/classifier_loss.png`.

**Yang Saya Ubah / Arahkan:**
- Memastikan token gaya tidak bocor ke classifier saat training classifier (menggunakan `with_style=False` pada `build_title_loader`).
- Memastikan metrik evaluasi akhir dihitung pada split test independen (1.496 sampel) tanpa menyentuh data latih.

---

## Sesi 2 · 2026-10-27 · Tahap: Debugging

**Prompt:**
> "Saat mencoba memuat checkpoint dan menjalankan evaluasi, terjadi error mismatch arsitektur pada state_dict: 
> checkpoint menggunakan kunci q_proj/k_proj/v_proj, ffn.net.0, ffn_norm, bert.norm, sedangkan definisi model lama menggunakan query/key/value, intermediate, output_norm. 
> Selain itu, checkpoint Step 3 pretrain_mlm.pt menggunakan struktur lama sedangkan style_mlm.pt dan classifier_*.pt menggunakan struktur baru. Tolong sinkronkan arsitektur ini."

**Pemeriksaan & Analisis Masalah:**
- Model hasil training Step 4 (checkpoint `classifier_random_init.pt`, `classifier_pretrained.pt`, dan `style_mlm.pt`) ternyata telah dilatih dengan arsitektur Pre-LayerNorm (norm sebelum sub-layer + final `bert.norm`) dan penamaan proyeksi `q_proj`, `k_proj`, `v_proj`, `out_proj`, serta `ffn.net` (`nn.Sequential`).
- Checkpoint `style_mlm.pt` menggunakan `mlm_head.weight` langsung terikat (*weight-tied*) tanpa `mlm_transform` terpisah.
- Checkpoint `pretrain_mlm.pt` dari Step 3 lama menggunakan konvensi nama `query`/`key`/`value` dan `output`.

**Perbaikan yang Diterapkan:**
- Memperbarui `src/model.py` agar selaras secara presisi dengan checkpoint yang tersimpan (arsitektur Pre-LayerNorm, proyeksi terpisah, final norm `bert.norm`, dan weight-tied `mlm_head`).
- Menambahkan mekanisme key-remapping di `src/classifier.py` (`load_pretrained_encoder`) dan `src/finetune_style.py` agar mampu membaca checkpoint pretraining lama maupun baru secara otomatis.
- Menambahkan property `mlm_bias` di `BertForMaskedLM` sebagai fallback kompatibilitas backward agar suite unit test lama (`tests/test_step3_training.py`) tetap lolos tanpa mengotori state dictionary PyTorch.

**Hasil Setelah Debugging:**
- Seluruh 8 unit test yang ada (`test_step1_data.py` dan `test_step3_training.py`) lolos 100%.

---

## Sesi 3 · 2026-10-28 · Tahap: Optimasi

**Prompt:**
> "Buatkan pipeline evaluasi komprehensif untuk Step 4 yang menghasilkan visualisasi confusion matrix, 
> perbandingan ablation F1 dan akurasi, contoh generasi fill-mask, serta Colab notebook siap pakai untuk training di GPU T4."

**Kode & Fitur yang Dihasilkan:**
- `src/evaluate_step4.py`: script evaluasi terpadu yang memuat kedua checkpoint classifier dan generator bergaya, menghitung metrik test lengkap, menghasilkan plot `reports/step4_confusion_matrix.png`, menyimpan `reports/step4_evaluation.json`, dan menjalankan inferensi fill-mask terpandu.
- `tests/test_step4.py`: unit & integration test untuk memvalidasi:
  1. Token gaya `[CB]`/`[NCB]` dan `[CLS]` tidak pernah di-mask.
  2. Arsitektur forward & backward classifier menghasilkan logits 2-kelas.
  3. Checkpoint `style_mlm.pt`, `classifier_random_init.pt`, dan `classifier_pretrained.pt` dapat dimuat sempurna.
  4. Laporan ablation berisi tabel perbandingan F1 & akurasi sesuai kriteria selesai proposal.
- `notebooks/step4_train_and_evaluate.ipynb`: Jupyter notebook siap pakai di Google Colab dengan akselerasi GPU T4 yang menjalankan training style MLM, training classifier ablation, evaluasi, hingga download hasil.

**Hasil Pengujian & Angka Evaluasi (Test Split = 1.496 sampel):**
```
=======================================================
  STEP 4 - HASIL EVALUASI TEST SPLIT
=======================================================
Variant                Accuracy   Macro F1
--------------------------------------------
random_init              0.7360     0.7172
pretrained               0.7313     0.7002
--------------------------------------------
Delta F1 (pretrained - random): -0.0171
Confusion matrix plot: reports/step4_confusion_matrix.png
Laporan JSON         : reports/step4_evaluation.json
Pytest suite         : 13/13 tests PASSED
```

**Pelajaran:**
1. **Perlindungan Token Gaya**: Dynamic masking harus secara eksplisit mengecualikan token gaya di posisi 1 (`protected_positions=(0, 1)`), karena jika token gaya di-mask, model tidak akan belajar conditioning asosiasi gaya terhadap kata-kata khas clickbait/non-clickbait.
2. **Ablation Study Pretraining**: Pada konfigurasi baseline dengan sedikit epoch, varian random-init menghasilkan F1 yang bersaing ketat (0.7172 vs 0.7002). Temuan ini penting untuk didokumentasikan di laporan: pretraining memerlukan korpus/epoch yang lebih besar untuk memperlebar keunggulan representasi semantik pada downstream task klasifikasi judul pendek.
3. **Kompatibilitas Arsitektur Checkpoint**: Perubahan kecil pada struktur modul (misalnya `nn.Sequential` vs modul bernama terpisah, atau Pre-LN vs Post-LN) mengubah nama kunci `state_dict`. Memasukkan remapping otomatis di modul pemuat checkpoint mencegah kegagalan pipeline antar-anggota tim.
