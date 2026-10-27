# Log Interaksi LLM: Anggota 1 (Step 1: Persiapan Data dan Tokenizer)

LLM yang dipakai: Claude (Claude Code, VS Code)

---

## Sesi 1 · 2026-10-27 · Tahap: Inisiasi

**Prompt:**
> (melampirkan proposal proyek) "Anggota 1, Persiapan Data dan Tokenizer, saya sebagai anggota 1
> hanya melakukan step ini dan dataset nya sudah ada di dalam folder tolong lakukan"
>
> "bikin di source code nya, saya harus push ke github repo"
>
> "kalau ada hal yang harus di-run kasih ke saya aja command-nya"
>
> "bikin readme untuk teman-teman yang lain melanjutkan di device masing-masing"

**Temuan LLM saat memeriksa dataset (sebelum menulis kode):**
- CSV raw hanya berisi 38.655 artikel, sedangkan XLSX berisi 46.119 (sesuai paper). Selisih
  terbesar ada di tribunnews (CSV 1.500, XLSX 9.662). Diputuskan memakai gabungan XLSX + CSV
  dengan dedup URL.
- `annotated/combined/csv/main.csv` berisi 15.000 judul (6.290 clickbait / 8.710 non-clickbait)
  dengan 30 judul duplikat.
- Artefak scraping di isi artikel: paragraf disambung koma (`.,Kalimat`), nama entitas ber-hyperlink
  hilang (`Indonesia , di Jakarta`), kalimat menempel (`ujarnya.Bobby`), dateline portal,
  tag editor detik `(iwd/iwd)`, `loading...`, `#GrowFearless with FIMELA`.

**Kode yang dihasilkan:**
- `config.py`: path, token khusus dengan ID tetap 0–6, hyperparameter semua step
- `src/data/cleaning.py`: normalisasi, pembuangan boilerplate, perbaikan struktur, pemecahan kalimat
- `src/data/prepare.py`: dedup, split stratified 80/10/10 (seed 42), anti-leakage judul test, korpus
- `src/data/tokenizer.py`: WordPiece 16.000 token + statistik panjang judul
- `src/data/dataset.py`: `PretrainDataset` (packing kalimat ke 128 token) dan `TitleDataset` (48 token)
- `tests/test_step1_data.py`: cek kriteria selesai
- `README.md`: panduan setup untuk anggota lain

**Yang saya ubah / arahkan dari jawaban LLM, dan alasannya:**
- Meminta semua kode ditulis sebagai file source code di repo, bukan hanya dijelaskan di chat,
  karena hasilnya harus di-push ke GitHub kelompok.
- Meminta LLM tidak menjalankan script sendiri dan cukup memberikan command-nya, supaya saya
  yang menjalankan pipeline dan melihat langsung output-nya.
- Meminta README untuk anggota lain. Dari permintaan ini, `prepare.py` diubah supaya memakai ulang
  split di `data/splits/` yang sudah di-commit (opsi `--resplit` untuk membuat ulang). Tujuannya
  agar split di device semua anggota identik.
- Perbaikan kecil di `cleaning.py`: sisa tanda baca `,.` (bekas entitas hilang, misalnya
  "termasuk, dan,.") dirapikan menjadi `.`.
- Saat push, ternyata email git global saya masih `pearl@example.com`, sehingga commit tidak akan
  tercatat di akun GitHub. Author commit diganti ke akun GitHub saya (`pearlyaradja`) sebelum
  push pertama ke repo `AI-IF-TelU/Proyek1-GenAI-BERT-Salamanca`.

**Hasil setelah dijalankan** (dari `reports/data_stats.json` dan `reports/tokenizer_stats.json`):
```
Data annotated
  baris awal                       : 15.000
  baris duplikat dibuang           : 34  (termasuk 8 judul dengan label bertentangan)
  setelah dedup                    : 14.958  (6.276 clickbait / 8.682 non-clickbait)

Split stratified 80/10/10 (seed 42)
  train : 11.966  (5.021 CB / 6.945 NCB, rasio CB 0,4196)
  val   :  1.496  (  628 CB /   868 NCB, rasio CB 0,4198)
  test  :  1.496  (  627 CB /   869 NCB, rasio CB 0,4191)

Anti-leakage (judul test dibuang dari korpus pretraining)
  kalimat judul artikel dibuang    : 1.510
  kalimat isi artikel dibuang      : 8
  total                            : 1.518

Korpus pretraining
  train : 45.497 artikel, 673.414 kalimat, 11.068.693 kata
  val   :    928 artikel,  14.273 kalimat,    235.104 kata

Tokenizer WordPiece
  vocab size                       : 16.000
  panjang judul (token)            : rata-rata 12,47 | median 12 | p95 19 | p99 23 | maks 28
  judul terpotong di max_len 48    : 0
```

**Pelajaran:**
- Jangan langsung percaya format file dataset: versi CSV dan XLSX dari sumber yang sama ternyata
  berbeda isi (CSV 38.655 artikel, XLSX 46.119).
- Judul test muncul sebagai judul artikel di data raw: 1.510 kalimat judul dibuang untuk 1.496
  judul test (lebih banyak dari jumlah judul test, kemungkinan karena ada artikel yang sama dengan
  URL berbeda). Tanpa langkah anti-leakage, model pretraining akan "melihat" data test.
- Max length 48 untuk judul aman: judul terpanjang hanya 28 token, jadi masih ada ruang untuk
  [CLS], token gaya, dan [SEP].
- Cek identitas git (`git config user.email`) sebelum commit pertama, karena riwayat commit dinilai.
