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

**Yang saya ubah / tolak dari jawaban LLM, dan alasannya:**
- ... (isi sendiri)

**Hasil setelah dijalankan:**
```
(tempel output python -m src.data.prepare, python -m src.data.tokenizer, python -m tests.test_step1_data)
```

**Pelajaran:**
- Jangan langsung percaya format file dataset: versi CSV dan XLSX dari sumber yang sama ternyata
  berbeda isi.
- ...
